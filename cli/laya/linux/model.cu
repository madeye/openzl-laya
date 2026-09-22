// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "cli/laya/linux/model.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>

#include "cli/laya/linux/safetensors.h"
#include "tools/json.hpp"

namespace openzl::laya {
namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            throw std::runtime_error(                                         \
                    std::string("CUDA: ") + cudaGetErrorString(err_) + " at " \
                    + #call);                                                 \
    } while (0)

#define LT_CHECK(call)                                                        \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                     \
            throw std::runtime_error(                                         \
                    std::string("cuBLASLt error ") + std::to_string(int(st_)) \
                    + " at " + #call);                                        \
    } while (0)

constexpr int kHidden       = 768;
constexpr int kHeads        = 12;
constexpr int kHeadDim      = 64;
constexpr int kIntermediate = 1152;
constexpr int kMaxOptions   = 32;
constexpr float kEps        = 1e-5f;

// ------------------------------------------------------------------ kernels

__device__ inline float warpSum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ inline float blockSum(float v, float* shared)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warpSum(v);
    __syncthreads();
    if (lane == 0)
        shared[warp] = v;
    __syncthreads();
    const int warps = (blockDim.x + 31) >> 5;
    v               = lane < warps ? shared[lane] : 0.f;
    if (warp == 0)
        v = warpSum(v);
    if (threadIdx.x == 0)
        shared[0] = v;
    __syncthreads();
    return shared[0];
}

__device__ inline float geluExact(float x)
{
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
}

// LayerNorm over kHidden values per row. One block per row. `input` is
// either fp32 activations or, when `ids` is given, embedding rows gathered
// from an fp16 table. Writes fp32 (`outF`) and/or fp16 (`outH`) outputs.
__global__ void layerNormKernel(
        const float* __restrict__ input,
        const int* __restrict__ ids,
        const __half* __restrict__ table,
        const __half* __restrict__ weight,
        const __half* __restrict__ bias,
        float* __restrict__ outF,
        __half* __restrict__ outH)
{
    __shared__ float shared[32];
    const int row = blockIdx.x;
    float v[3];
    const __half* embed = ids ? table + size_t(ids[row]) * kHidden : nullptr;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = threadIdx.x + j * 256;
        v[j]        = embed ? __half2float(embed[c])
                            : input[size_t(row) * kHidden + c];
    }
    const float mean = blockSum(v[0] + v[1] + v[2], shared) / kHidden;
    float sq         = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        v[j] -= mean;
        sq += v[j] * v[j];
    }
    const float inv = rsqrtf(blockSum(sq, shared) / kHidden + kEps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = threadIdx.x + j * 256;
        float y     = v[j] * inv * __half2float(weight[c]);
        if (bias)
            y += __half2float(bias[c]);
        if (outF)
            outF[size_t(row) * kHidden + c] = y;
        if (outH)
            outH[size_t(row) * kHidden + c] = __float2half(y);
    }
}

// Splits a [L, 3*kHidden] projection into Q, K, V as [heads, L, 64] fp16,
// applying rotary embeddings to Q and K when `cosine` is given.
__global__ void splitHeadsKernel(
        const float* __restrict__ qkv,
        const float* __restrict__ cosine,
        const float* __restrict__ sine,
        __half* __restrict__ q,
        __half* __restrict__ k,
        __half* __restrict__ v,
        int length)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length * kHeads * (kHeadDim / 2))
        return;
    const int d      = idx % (kHeadDim / 2);
    const int h      = (idx / (kHeadDim / 2)) % kHeads;
    const int i      = idx / (kHeadDim / 2) / kHeads;
    const float* row = qkv + size_t(i) * 3 * kHidden;
    const size_t out = (size_t(h) * length + i) * kHeadDim;
    const float qa = row[h * kHeadDim + d], qb = row[h * kHeadDim + d + 32];
    const float ka = row[kHidden + h * kHeadDim + d],
                kb = row[kHidden + h * kHeadDim + d + 32];
    float qo0 = qa, qo1 = qb, ko0 = ka, ko1 = kb;
    if (cosine) {
        const float c = cosine[i * (kHeadDim / 2) + d],
                    s = sine[i * (kHeadDim / 2) + d];
        qo0           = qa * c - qb * s;
        qo1           = qb * c + qa * s;
        ko0           = ka * c - kb * s;
        ko1           = kb * c + ka * s;
    }
    q[out + d]      = __float2half(qo0);
    q[out + d + 32] = __float2half(qo1);
    k[out + d]      = __float2half(ko0);
    k[out + d + 32] = __float2half(ko1);
    v[out + d]      = __float2half(row[2 * kHidden + h * kHeadDim + d]);
    v[out + d + 32] = __float2half(row[2 * kHidden + h * kHeadDim + d + 32]);
}

// Fused attention on tensor cores (WMMA m16n16k16, fp16 inputs, fp32
// accumulation): a block of two warps handles 32 queries of one head, each
// warp 16 queries. K and V are staged in 64-key tiles; per tile the warp
// computes S = Q K^T (16x64) with 16 mma ops, applies the online softmax in
// fp32 on shared memory (one lane per half row), then O += P V with 16 more.
// Keys beyond `valid` and, when window > 0, keys with |i-j| > window are
// masked. Writes the head's 64 columns of the row-major [L, 768] output.
constexpr int kAttnQueries = 32;
constexpr int kAttnTile    = 64;
constexpr int kAttnWarps   = kAttnQueries / 16;
constexpr int kSLd         = kAttnTile + 8; // padded leading dimensions
constexpr int kOLd         = kHeadDim + 4;

__global__ void __launch_bounds__(kAttnWarps * 32) attentionKernel(
        const __half* __restrict__ q,
        const __half* __restrict__ k,
        const __half* __restrict__ v,
        __half* __restrict__ out,
        int length,
        const int* __restrict__ validPtr,
        int window,
        float scale)
{
    using namespace nvcuda;
    __shared__ __align__(32) __half kTile[kAttnTile * kHeadDim];
    __shared__ __align__(32) __half vTile[kAttnTile * kHeadDim];
    __shared__ __align__(32) float sTile[kAttnWarps][16 * kSLd];
    __shared__ __align__(32) __half pTile[kAttnWarps][16 * kSLd];
    __shared__ __align__(32) float oTile[kAttnWarps][16 * kOLd];
    const int valid       = *validPtr;
    const int head        = blockIdx.y;
    const int i0          = blockIdx.x * kAttnQueries;
    const int warp        = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int rowBase     = i0 + warp * 16;
    const int row         = lane >> 1;       // this lane's row within the warp
    const int colBase     = (lane & 1) * 32; // this lane's half row
    const size_t headBase = size_t(head) * length * kHeadDim;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
            qFrag[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
        wmma::load_matrix_sync(
                qFrag[kk],
                q + headBase + size_t(rowBase) * kHeadDim + kk * 16,
                kHeadDim);
    for (int c = lane; c < 16 * kOLd; c += 32)
        oTile[warp][c] = 0.f;
    float m = -INFINITY, l = 0.f;

    int jStart = 0, jEnd = valid;
    if (window > 0) {
        jStart = max(0, i0 - window);
        jEnd   = min(valid, i0 + kAttnQueries - 1 + window + 1);
    }
    jStart -= jStart % kAttnTile;
    for (int tile = jStart; tile < jEnd; tile += kAttnTile) {
        __syncthreads();
        for (int chunk = threadIdx.x; chunk < kAttnTile * kHeadDim / 8;
             chunk += kAttnWarps * 32) {
            const int r   = chunk / (kHeadDim / 8),
                      col = (chunk % (kHeadDim / 8)) * 8;
            const int j   = tile + r;
            int4 kk = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (j < length) {
                kk = *reinterpret_cast<const int4*>(
                        k + headBase + size_t(j) * kHeadDim + col);
                vv = *reinterpret_cast<const int4*>(
                        v + headBase + size_t(j) * kHeadDim + col);
            }
            *reinterpret_cast<int4*>(kTile + r * kHeadDim + col) = kk;
            *reinterpret_cast<int4*>(vTile + r * kHeadDim + col) = vv;
        }
        __syncthreads();
        // S = Q K^T for this warp's 16 rows and the tile's 64 keys.
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<
                        wmma::matrix_b,
                        16,
                        16,
                        16,
                        __half,
                        wmma::col_major>
                        kFrag;
                wmma::load_matrix_sync(
                        kFrag, kTile + (n * 16) * kHeadDim + kk * 16, kHeadDim);
                wmma::mma_sync(acc, qFrag[kk], kFrag, acc);
            }
            wmma::store_matrix_sync(
                    sTile[warp] + n * 16, acc, kSLd, wmma::mem_row_major);
        }
        __syncwarp();
        // Online softmax for row `row`, columns [colBase, colBase + 32).
        const int i   = rowBase + row;
        float* sRow   = sTile[warp] + row * kSLd;
        float tileMax = -INFINITY;
        float sc[32];
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            const int j   = tile + colBase + c;
            const bool ok = j < valid && (window == 0 || abs(i - j) <= window);
            sc[c]         = ok ? sRow[colBase + c] * scale : -INFINITY;
            tileMax       = fmaxf(tileMax, sc[c]);
        }
        tileMax      = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, 1));
        __half* pRow = pTile[warp] + row * kSLd;
        float* oRow  = oTile[warp] + row * kOLd;
        if (tileMax == -INFINITY) {
#pragma unroll
            for (int c = 0; c < 32; ++c)
                pRow[colBase + c] = __float2half(0.f);
        } else {
            const float mNew  = fmaxf(m, tileMax);
            const float alpha = expf(m - mNew);
            float sum         = 0.f;
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                const float pj = sc[c] == -INFINITY ? 0.f : expf(sc[c] - mNew);
                pRow[colBase + c] = __float2half(pj);
                sum += pj;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            l = l * alpha + sum;
            m = mNew;
#pragma unroll
            for (int c = 0; c < 32; ++c)
                oRow[colBase + c] *= alpha;
        }
        __syncwarp();
        // O += P V
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(
                    acc, oTile[warp] + n * 16, kOLd, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<
                        wmma::matrix_a,
                        16,
                        16,
                        16,
                        __half,
                        wmma::row_major>
                        pFrag;
                wmma::fragment<
                        wmma::matrix_b,
                        16,
                        16,
                        16,
                        __half,
                        wmma::row_major>
                        vFrag;
                wmma::load_matrix_sync(pFrag, pTile[warp] + kk * 16, kSLd);
                wmma::load_matrix_sync(
                        vFrag, vTile + (kk * 16) * kHeadDim + n * 16, kHeadDim);
                wmma::mma_sync(acc, pFrag, vFrag, acc);
            }
            wmma::store_matrix_sync(
                    oTile[warp] + n * 16, acc, kOLd, wmma::mem_row_major);
        }
        __syncwarp();
    }
    const float inv   = l > 0.f ? 1.f / l : 0.f;
    const float* oRow = oTile[warp] + row * kOLd;
    __half* dst =
            out + size_t(rowBase + row) * kHidden + head * kHeadDim + colBase;
#pragma unroll
    for (int c = 0; c < 32; ++c)
        dst[c] = __float2half(oRow[c + colBase] * inv);
}

// GLU with exact GELU: out = gelu(x[:, :I]) * x[:, I:].
__global__ void
gluKernel(const float* __restrict__ wi, __half* __restrict__ out, int rows)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * kIntermediate)
        return;
    const int r = idx / kIntermediate, c = idx % kIntermediate;
    const float* row = wi + size_t(r) * 2 * kIntermediate;
    out[idx]         = __float2half(geluExact(row[c]) * row[kIntermediate + c]);
}

template <int kMode> // 0: identity, 1: relu, 2: gelu
__global__ void activationKernel(
        const float* __restrict__ in,
        __half* __restrict__ out,
        size_t count)
{
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float v = in[idx];
    if (kMode == 1)
        v = fmaxf(v, 0.f);
    else if (kMode == 2)
        v = geluExact(v);
    out[idx] = __float2half(v);
}

__global__ void
addRowKernel(float* __restrict__ x, const float* __restrict__ row, int rows)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * kHidden)
        x[idx] += row[idx % kHidden];
}

__global__ void gatherRowsKernel(
        const float* __restrict__ x,
        const int* __restrict__ positions,
        float* __restrict__ out,
        int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count * kHidden)
        out[idx] =
                x[size_t(positions[idx / kHidden]) * kHidden + idx % kHidden];
}

// ----------------------------------------------------------------- helpers

template <typename T>
T* deviceAlloc(size_t count)
{
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

template <typename T>
T* upload(const std::vector<T>& host)
{
    T* p = deviceAlloc<T>(host.size());
    CUDA_CHECK(cudaMemcpy(
            p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

__half* uploadHalf(const std::vector<uint16_t>& bits)
{
    __half* p = deviceAlloc<__half>(bits.size());
    CUDA_CHECK(cudaMemcpy(
            p, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
    return p;
}

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace         = nullptr;
    size_t workspaceSize    = 32 << 20;
    cudaStream_t stream     = nullptr;
    struct Key {
        bool transA;
        int m, n, k, batch, outType, lda, ldb, ldc;
        long strideA, strideB, strideC;
        bool bias;
        bool operator<(const Key& o) const
        {
            return std::tie(
                           transA,
                           m,
                           n,
                           k,
                           batch,
                           outType,
                           lda,
                           ldb,
                           ldc,
                           strideA,
                           strideB,
                           strideC,
                           bias)
                    < std::tie(
                            o.transA,
                            o.m,
                            o.n,
                            o.k,
                            o.batch,
                            o.outType,
                            o.lda,
                            o.ldb,
                            o.ldc,
                            o.strideA,
                            o.strideB,
                            o.strideC,
                            o.bias);
        }
    };
    struct Plan {
        cublasLtMatmulDesc_t desc;
        cublasLtMatrixLayout_t a, b, c;
        cublasLtMatmulAlgo_t algo;
    };
    std::map<Key, Plan> plans;

    void init(cudaStream_t s)
    {
        stream = s;
        LT_CHECK(cublasLtCreate(&handle));
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    ~Gemm()
    {
        for (auto& [key, plan] : plans) {
            cublasLtMatmulDescDestroy(plan.desc);
            cublasLtMatrixLayoutDestroy(plan.a);
            cublasLtMatrixLayoutDestroy(plan.b);
            cublasLtMatrixLayoutDestroy(plan.c);
        }
        if (workspace)
            cudaFree(workspace);
        if (handle)
            cublasLtDestroy(handle);
    }

    // Column-major cuBLAS view of row-major data: computes
    // C[n x m] = op(A)[m x k] * B[k x n] (+ beta * C) (+ bias[m]).
    // transA=true means A is a row-major [m, k] matrix used transposed (a
    // Linear weight); transA=false means A is stored k-major with lda.
    void run(
            bool transA,
            const __half* A,
            int lda,
            long strideA,
            const __half* B,
            int ldb,
            long strideB,
            void* C,
            cudaDataType outType,
            int ldc,
            long strideC,
            int m,
            int n,
            int k,
            int batch,
            float beta,
            const float* bias)
    {
        Key key{ transA, m,   n,       k,       batch,   int(outType),   lda,
                 ldb,    ldc, strideA, strideB, strideC, bias != nullptr };
        auto it = plans.find(key);
        if (it == plans.end()) {
            Plan plan{};
            LT_CHECK(cublasLtMatmulDescCreate(
                    &plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                              opB = CUBLAS_OP_N;
            LT_CHECK(cublasLtMatmulDescSetAttribute(
                    plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
            LT_CHECK(cublasLtMatmulDescSetAttribute(
                    plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
            if (bias) {
                cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
                LT_CHECK(cublasLtMatmulDescSetAttribute(
                        plan.desc,
                        CUBLASLT_MATMUL_DESC_EPILOGUE,
                        &epilogue,
                        sizeof(epilogue)));
                cudaDataType biasType = CUDA_R_32F;
                LT_CHECK(cublasLtMatmulDescSetAttribute(
                        plan.desc,
                        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                        &biasType,
                        sizeof(biasType)));
            }
            LT_CHECK(cublasLtMatrixLayoutCreate(
                    &plan.a, CUDA_R_16F, transA ? k : m, transA ? m : k, lda));
            LT_CHECK(
                    cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16F, k, n, ldb));
            LT_CHECK(cublasLtMatrixLayoutCreate(&plan.c, outType, m, n, ldc));
            if (batch > 1) {
                for (auto [layout, stride] : { std::pair{ plan.a, strideA },
                                               std::pair{ plan.b, strideB },
                                               std::pair{ plan.c, strideC } }) {
                    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
                            layout,
                            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                            &batch,
                            sizeof(batch)));
                    int64_t s = stride;
                    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
                            layout,
                            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                            &s,
                            sizeof(s)));
                }
            }
            cublasLtMatmulPreference_t preference;
            LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
            LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
                    preference,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspaceSize,
                    sizeof(workspaceSize)));
            // Time the heuristic's candidates once (like PyTorch's
            // max-autotune, but at load) and keep the fastest.
            cublasLtMatmulHeuristicResult_t results[8];
            int found = 0;
            LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    preference,
                    8,
                    results,
                    &found));
            cublasLtMatmulPreferenceDestroy(preference);
            if (!found)
                throw std::runtime_error("no cuBLASLt algorithm for GEMM");
            const float alpha = 1.f, zero = 0.f;
            if (bias)
                LT_CHECK(cublasLtMatmulDescSetAttribute(
                        plan.desc,
                        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                        &bias,
                        sizeof(bias)));
            cudaEvent_t begin, end;
            CUDA_CHECK(cudaEventCreate(&begin));
            CUDA_CHECK(cudaEventCreate(&end));
            float best = 1e30f;
            for (int i = 0; i < found; ++i) {
                // Benchmark into the workspace-free scratch: beta 0 keeps C
                // untouched semantics irrelevant here since C is overwritten
                // by the real call afterwards.
                if (cublasLtMatmul(
                            handle,
                            plan.desc,
                            &alpha,
                            A,
                            plan.a,
                            B,
                            plan.b,
                            &zero,
                            C,
                            plan.c,
                            C,
                            plan.c,
                            &results[i].algo,
                            workspace,
                            workspaceSize,
                            stream)
                    != CUBLAS_STATUS_SUCCESS)
                    continue;
                CUDA_CHECK(cudaEventRecord(begin, stream));
                for (int r = 0; r < 5; ++r)
                    cublasLtMatmul(
                            handle,
                            plan.desc,
                            &alpha,
                            A,
                            plan.a,
                            B,
                            plan.b,
                            &zero,
                            C,
                            plan.c,
                            C,
                            plan.c,
                            &results[i].algo,
                            workspace,
                            workspaceSize,
                            stream);
                CUDA_CHECK(cudaEventRecord(end, stream));
                CUDA_CHECK(cudaEventSynchronize(end));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
                if (ms < best) {
                    best      = ms;
                    plan.algo = results[i].algo;
                }
            }
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            if (best == 1e30f)
                throw std::runtime_error(
                        "no runnable cuBLASLt algorithm for GEMM");
            it = plans.emplace(key, plan).first;
        }
        const Plan& plan  = it->second;
        const float alpha = 1.f;
        if (bias)
            LT_CHECK(cublasLtMatmulDescSetAttribute(
                    plan.desc,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &bias,
                    sizeof(bias)));
        LT_CHECK(cublasLtMatmul(
                handle,
                plan.desc,
                &alpha,
                A,
                plan.a,
                B,
                plan.b,
                &beta,
                C,
                plan.c,
                C,
                plan.c,
                &plan.algo,
                workspace,
                workspaceSize,
                stream));
    }

    bool nnLayout =
            !(getenv("OPENZL_LAYA_NN_LAYOUT")
              && !atoi(getenv("OPENZL_LAYA_NN_LAYOUT")));
    // out[M, N] (+)= X[M, K] * W[N, K]^T + bias[N]; row-major. With nnLayout
    // the weight is stored as W^T [K, N].
    void linear(
            const __half* W,
            const __half* X,
            void* out,
            cudaDataType outType,
            int M,
            int N,
            int K,
            float beta        = 0.f,
            const float* bias = nullptr)
    {
        if (nnLayout)
            run(false,
                W,
                N,
                0,
                X,
                K,
                0,
                out,
                outType,
                N,
                0,
                N,
                M,
                K,
                1,
                beta,
                bias);
        else
            run(true,
                W,
                K,
                0,
                X,
                K,
                0,
                out,
                outType,
                N,
                0,
                N,
                M,
                K,
                1,
                beta,
                bias);
    }
};

struct Layer {
    __half* attnNorm = nullptr; // null for layer 0 (identity)
    __half* wqkv;
    __half* wo;
    __half* mlpNorm;
    __half* wi;
    __half* wo2;
    bool global;
};

struct HeadLayer {
    __half *norm1W, *norm1B, *inProj, *outProj, *norm2W, *norm2B, *linear1,
            *linear2;
    float *inProjB, *outProjB, *linear1B, *linear2B;
};

} // namespace

struct Model::Impl {
    // configuration
    int maxLen = 1024, headMaxLen = 256, layers = 22, headLayers = 2,
        slidingWindow = 64, vocab = 256000, padId = 0;
    float ropeTheta = 160000.f;
    std::vector<float> temperatureByType;
    std::map<std::string, float> temperatureByOptions;
    std::string device;

    // weights
    __half* tokEmbeddings;
    __half* embNorm;
    std::vector<Layer> layer;
    __half* finalNorm;
    float* typeEmbedding; // row 0 (choice)
    std::vector<HeadLayer> head;
    __half *scorerNormW, *scorerNormB, *scorerW1, *scorerW3;
    float *scorerB1, *scorerB3;
    std::vector<float> actW, actB, actW2, actB2; // host, fp32
    float *cosTable, *sinTable;                  // [maxLen, 32]

    // activations (sized for maxLen)
    int* ids;
    int* markers;
    int* valid;        // device-side real token count read by the softmax masks
    float* h;          // residual stream [L, 768]
    __half* x16;       // normalized input to GEMMs [L, 768]
    float* qkv;        // [L, 2304]
    __half *q, *k, *v; // [heads, L, 64]
    __half* attn;      // [L, 768]
    float* wiOut;      // [L, 2304] also reused for head FF [L, 3072]
    __half* glu;       // [L, 1152] / [L, 3072]
    float* gathered;   // [32, 768]
    float* scorerHidden; // [32, 768]
    __half* scorerHalf;  // [32, 768]
    float* logits;       // [32]
    float* pinnedOut;    // host: 32 logits + 768 pooled
    int* pinnedIds;      // host: maxLen + 32

    Gemm gemm;
    cudaStream_t stream;
    cudaEvent_t start, stop;
    std::map<int, cudaGraphExec_t> graphs;
    size_t bytes = 0;

    void loadConfig(const std::string& dir)
    {
        std::ifstream agent(dir + "/rl_agent_config.json");
        nlohmann::json cfg = nlohmann::json::parse(agent);
        maxLen             = cfg.value("max_len", 1024);
        headMaxLen         = cfg.value("head_max_len", 256);
        headLayers         = cfg.value("head_layers", 2);
        temperatureByType =
                cfg.value("temperature", std::vector<float>{ 1, 1, 1 });
        if (cfg.contains("temperature_by_options"))
            for (auto it = cfg["temperature_by_options"].begin();
                 it != cfg["temperature_by_options"].end();
                 ++it)
                temperatureByOptions[it.key()] = it.value().get<float>();
        std::ifstream enc(dir + "/encoder/config.json");
        nlohmann::json e = nlohmann::json::parse(enc);
        if (e.value("hidden_size", 768) != kHidden
            || e.value("num_attention_heads", 12) != kHeads
            || e.value("intermediate_size", 1152) != kIntermediate)
            throw std::runtime_error("unsupported encoder geometry");
        layers = e.value("num_hidden_layers", 22);
        vocab  = e.value("vocab_size", 256000);
        padId  = e.value("pad_token_id", 0);
        if (e.contains("rope_parameters")
            && e["rope_parameters"].contains("full_attention"))
            ropeTheta = e["rope_parameters"]["full_attention"].value(
                    "rope_theta", 160000.f);
        else
            ropeTheta = e.value("global_rope_theta", 160000.f);
        if (e.contains("rope_parameters")
            && e["rope_parameters"].contains("sliding_attention")
            && e["rope_parameters"]["sliding_attention"].value(
                       "rope_theta", ropeTheta)
                    != ropeTheta)
            throw std::runtime_error(
                    "distinct local/global rope theta is unsupported");
        slidingWindow = e.value("local_attention", 128) / 2;
        if (e.value("norm_eps", 1e-5) != 1e-5)
            throw std::runtime_error("unsupported norm_eps");
    }

    __half* loadHalf(SafeTensors& st, const std::string& name)
    {
        auto bits = st.halves(name);
        bytes += bits.size() * 2;
        return uploadHalf(bits);
    }
    // Linear weight [N, K] stored transposed as [K, N] when nnLayout is set,
    // so cuBLASLt sees a plain (N, N) problem.
    __half* loadLinear(SafeTensors& st, const std::string& name)
    {
        if (!gemm.nnLayout)
            return loadHalf(st, name);
        const auto& info = st.info(name);
        auto bits        = st.halves(name);
        const size_t N = info.shape[0], K = info.shape[1];
        std::vector<uint16_t> t(bits.size());
        for (size_t n = 0; n < N; ++n)
            for (size_t kk = 0; kk < K; ++kk)
                t[kk * N + n] = bits[n * K + kk];
        bytes += t.size() * 2;
        return uploadHalf(t);
    }
    float* loadFloat(SafeTensors& st, const std::string& name)
    {
        auto f = st.floats(name);
        bytes += f.size() * 4;
        return upload(f);
    }

    void loadWeights(const std::string& dir)
    {
        SafeTensors st(dir + "/model.safetensors");
        tokEmbeddings =
                loadHalf(st, "encoder.embeddings.tok_embeddings.weight");
        embNorm = loadHalf(st, "encoder.embeddings.norm.weight");
        for (int i = 0; i < layers; ++i) {
            const std::string p = "encoder.layers." + std::to_string(i) + ".";
            Layer l{};
            l.attnNorm =
                    i == 0 ? nullptr : loadHalf(st, p + "attn_norm.weight");
            l.wqkv    = loadLinear(st, p + "attn.Wqkv.weight");
            l.wo      = loadLinear(st, p + "attn.Wo.weight");
            l.mlpNorm = loadHalf(st, p + "mlp_norm.weight");
            l.wi      = loadLinear(st, p + "mlp.Wi.weight");
            l.wo2     = loadLinear(st, p + "mlp.Wo.weight");
            l.global  = i % 3 == 0;
            layer.push_back(l);
        }
        finalNorm = loadHalf(st, "encoder.final_norm.weight");
        {
            auto t = st.floats("type_emb.weight");
            typeEmbedding =
                    upload(std::vector<float>(t.begin(), t.begin() + kHidden));
        }
        for (int i = 0; i < headLayers; ++i) {
            const std::string p = "head.layers." + std::to_string(i) + ".";
            HeadLayer hl{};
            hl.norm1W   = loadHalf(st, p + "norm1.weight");
            hl.norm1B   = loadHalf(st, p + "norm1.bias");
            hl.inProj   = loadLinear(st, p + "self_attn.in_proj_weight");
            hl.inProjB  = loadFloat(st, p + "self_attn.in_proj_bias");
            hl.outProj  = loadLinear(st, p + "self_attn.out_proj.weight");
            hl.outProjB = loadFloat(st, p + "self_attn.out_proj.bias");
            hl.norm2W   = loadHalf(st, p + "norm2.weight");
            hl.norm2B   = loadHalf(st, p + "norm2.bias");
            hl.linear1  = loadLinear(st, p + "linear1.weight");
            hl.linear1B = loadFloat(st, p + "linear1.bias");
            hl.linear2  = loadLinear(st, p + "linear2.weight");
            hl.linear2B = loadFloat(st, p + "linear2.bias");
            head.push_back(hl);
        }
        scorerNormW = loadHalf(st, "scorer.0.weight");
        scorerNormB = loadHalf(st, "scorer.0.bias");
        scorerW1    = loadLinear(st, "scorer.1.weight");
        scorerB1    = loadFloat(st, "scorer.1.bias");
        scorerW3    = loadLinear(st, "scorer.3.weight");
        scorerB3    = loadFloat(st, "scorer.3.bias");
        actW        = st.floats("act_head.0.weight");
        actB        = st.floats("act_head.0.bias");
        actW2       = st.floats("act_head.2.weight");
        actB2       = st.floats("act_head.2.bias");
        if (actW.size() != size_t(256) * (kHidden + 4) || actW2.size() != 512)
            throw std::runtime_error("unexpected action head shape");
        // Rotary tables, computed like the reference: fp32 position times
        // fp32 inverse frequency, then cos/sin.
        std::vector<float> cosT(size_t(maxLen) * 32), sinT(size_t(maxLen) * 32);
        for (int d = 0; d < 32; ++d) {
            const float invFreq =
                    1.f / powf(ropeTheta, float(2 * d) / float(kHeadDim));
            for (int i = 0; i < maxLen; ++i) {
                const float angle        = float(i) * invFreq;
                cosT[size_t(i) * 32 + d] = cosf(angle);
                sinT[size_t(i) * 32 + d] = sinf(angle);
            }
        }
        cosTable = upload(cosT);
        sinTable = upload(sinT);
    }

    void allocate()
    {
        const size_t L = size_t(maxLen);
        ids            = deviceAlloc<int>(L);
        markers        = deviceAlloc<int>(kMaxOptions);
        valid          = deviceAlloc<int>(1);
        h              = deviceAlloc<float>(L * kHidden);
        x16            = deviceAlloc<__half>(L * kHidden);
        qkv            = deviceAlloc<float>(L * 3 * kHidden);
        q              = deviceAlloc<__half>(size_t(kHeads) * L * kHeadDim);
        k              = deviceAlloc<__half>(size_t(kHeads) * L * kHeadDim);
        v              = deviceAlloc<__half>(size_t(kHeads) * L * kHeadDim);
        attn           = deviceAlloc<__half>(L * kHidden);
        wiOut          = deviceAlloc<float>(L * 4 * kHidden);
        glu            = deviceAlloc<__half>(L * 4 * kHidden);
        gathered       = deviceAlloc<float>(size_t(kMaxOptions) * kHidden);
        scorerHidden   = deviceAlloc<float>(size_t(kMaxOptions) * kHidden);
        scorerHalf     = deviceAlloc<__half>(size_t(kMaxOptions) * kHidden);
        logits         = deviceAlloc<float>(kMaxOptions);
        CUDA_CHECK(cudaMallocHost(
                &pinnedOut, (kMaxOptions + kHidden) * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
                &pinnedIds, (L + kMaxOptions + 1) * sizeof(int)));
        bytes += L
                * (kHidden * 4 + kHidden * 2 + 3 * kHidden * 4
                   + 3 * kHeads * kHeadDim * 2 + kHidden * 2 + 4 * kHidden * 6);
    }

    // Attention over Q/K/V [heads, L, 64] writing [L, 768] fp16 into `attn`.
    void attention(int L, int window)
    {
        const dim3 grid(L / kAttnQueries, kHeads);
        attentionKernel<<<grid, kAttnWarps * 32, 0, stream>>>(
                q, k, v, attn, L, valid, window, 0.125f);
    }

    void forward(int L, int markerCount)
    {
        const int rows        = L;
        const int splitBlocks = (L * kHeads * (kHeadDim / 2) + 255) / 256;
        // embeddings + norm -> h (fp32) and x16 (layer 0 attention input)
        layerNormKernel<<<rows, 256, 0, stream>>>(
                nullptr, ids, tokEmbeddings, embNorm, nullptr, h, x16);
        for (int i = 0; i < layers; ++i) {
            const Layer& l = layer[size_t(i)];
            if (l.attnNorm)
                layerNormKernel<<<rows, 256, 0, stream>>>(
                        h, nullptr, nullptr, l.attnNorm, nullptr, nullptr, x16);
            gemm.linear(l.wqkv, x16, qkv, CUDA_R_32F, L, 3 * kHidden, kHidden);
            splitHeadsKernel<<<splitBlocks, 256, 0, stream>>>(
                    qkv, cosTable, sinTable, q, k, v, L);
            attention(L, l.global ? 0 : slidingWindow);
            gemm.linear(
                    l.wo,
                    attn,
                    h,
                    CUDA_R_32F,
                    L,
                    kHidden,
                    kHidden,
                    1.f); // residual
            layerNormKernel<<<rows, 256, 0, stream>>>(
                    h, nullptr, nullptr, l.mlpNorm, nullptr, nullptr, x16);
            gemm.linear(
                    l.wi,
                    x16,
                    wiOut,
                    CUDA_R_32F,
                    L,
                    2 * kIntermediate,
                    kHidden);
            gluKernel<<<(L * kIntermediate + 255) / 256, 256, 0, stream>>>(
                    wiOut, glu, L);
            gemm.linear(
                    l.wo2,
                    glu,
                    h,
                    CUDA_R_32F,
                    L,
                    kHidden,
                    kIntermediate,
                    1.f); // residual
        }
        // final norm into the head's residual stream (reuse qkv storage as x)
        float* x = qkv; // [L, 768] fp32, distinct from h
        layerNormKernel<<<rows, 256, 0, stream>>>(
                h, nullptr, nullptr, finalNorm, nullptr, x, nullptr);
        addRowKernel<<<(L * kHidden + 255) / 256, 256, 0, stream>>>(
                x, typeEmbedding, L);
        float* projection = wiOut; // [L, 2304] / [L, 3072]
        for (const HeadLayer& hl : head) {
            layerNormKernel<<<rows, 256, 0, stream>>>(
                    x, nullptr, nullptr, hl.norm1W, hl.norm1B, nullptr, x16);
            gemm.linear(
                    hl.inProj,
                    x16,
                    projection,
                    CUDA_R_32F,
                    L,
                    3 * kHidden,
                    kHidden,
                    0.f,
                    hl.inProjB);
            splitHeadsKernel<<<splitBlocks, 256, 0, stream>>>(
                    projection, nullptr, nullptr, q, k, v, L);
            attention(L, 0);
            gemm.linear(
                    hl.outProj,
                    attn,
                    x,
                    CUDA_R_32F,
                    L,
                    kHidden,
                    kHidden,
                    1.f,
                    hl.outProjB);
            layerNormKernel<<<rows, 256, 0, stream>>>(
                    x, nullptr, nullptr, hl.norm2W, hl.norm2B, nullptr, x16);
            gemm.linear(
                    hl.linear1,
                    x16,
                    projection,
                    CUDA_R_32F,
                    L,
                    4 * kHidden,
                    kHidden,
                    0.f,
                    hl.linear1B);
            activationKernel<1>
                    <<<(L * 4 * kHidden + 255) / 256, 256, 0, stream>>>(
                            projection, glu, size_t(L) * 4 * kHidden);
            gemm.linear(
                    hl.linear2,
                    glu,
                    x,
                    CUDA_R_32F,
                    L,
                    kHidden,
                    4 * kHidden,
                    1.f,
                    hl.linear2B);
        }
        // scorer on the marker rows
        gatherRowsKernel<<<
                (markerCount * kHidden + 255) / 256,
                256,
                0,
                stream>>>(x, markers, gathered, markerCount);
        layerNormKernel<<<markerCount, 256, 0, stream>>>(
                gathered,
                nullptr,
                nullptr,
                scorerNormW,
                scorerNormB,
                nullptr,
                scorerHalf);
        gemm.linear(
                scorerW1,
                scorerHalf,
                scorerHidden,
                CUDA_R_32F,
                markerCount,
                kHidden,
                kHidden,
                0.f,
                scorerB1);
        activationKernel<2>
                <<<(markerCount * kHidden + 255) / 256, 256, 0, stream>>>(
                        scorerHidden,
                        scorerHalf,
                        size_t(markerCount) * kHidden);
        gemm.linear(
                scorerW3,
                scorerHalf,
                logits,
                CUDA_R_32F,
                markerCount,
                1,
                kHidden,
                0.f,
                scorerB3);
        CUDA_CHECK(cudaMemcpyAsync(
                pinnedOut,
                logits,
                size_t(markerCount) * 4,
                cudaMemcpyDeviceToHost,
                stream));
        CUDA_CHECK(cudaMemcpyAsync(
                pinnedOut + kMaxOptions,
                x,
                kHidden * 4,
                cudaMemcpyDeviceToHost,
                stream));
    }
};

Model::Model(const std::string& directory) : impl_(std::make_unique<Impl>())
{
    auto& m = *impl_;
    CUDA_CHECK(cudaFree(nullptr)); // initialize the context early
    cudaDeviceProp prop{};
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    m.device = prop.name;
    CUDA_CHECK(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&m.start));
    CUDA_CHECK(cudaEventCreate(&m.stop));
    m.gemm.init(m.stream);
    m.loadConfig(directory);
    m.loadWeights(directory);
    m.allocate();
}

Model::~Model()
{
    auto& m = *impl_;
    for (auto& [length, exec] : m.graphs)
        cudaGraphExecDestroy(exec);
    cudaStreamSynchronize(m.stream);
}

int Model::maxLength() const
{
    return impl_->maxLen;
}
int Model::headMaxLength() const
{
    return impl_->headMaxLen;
}
int Model::maxOptions() const
{
    return kMaxOptions;
}
std::string Model::deviceName() const
{
    return impl_->device;
}
std::string Model::precision() const
{
    return "fp16-tensorcore-fp32-accumulate";
}
size_t Model::deviceBytes() const
{
    return impl_->bytes;
}

float Model::temperature(int options) const
{
    const char* size = options <= 2 ? "2"
            : options <= 5          ? "3-5"
            : options <= 10         ? "6-10"
                                    : "11+";
    auto it = impl_->temperatureByOptions.find(std::string("choice:") + size);
    if (it != impl_->temperatureByOptions.end())
        return it->second;
    return impl_->temperatureByType.empty() ? 1.f : impl_->temperatureByType[0];
}

ModelOutput Model::infer(
        const std::vector<int>& ids,
        const std::vector<int>& markers,
        int paddedLength,
        bool useGraph)
{
    auto& m = *impl_;
    if (ids.empty() || int(ids.size()) > paddedLength || paddedLength > m.maxLen
        || paddedLength % kAttnQueries)
        throw std::runtime_error("invalid sequence length");
    if (markers.empty() || int(markers.size()) > kMaxOptions)
        throw std::runtime_error("invalid marker count");
    const int L     = paddedLength;
    const int valid = int(ids.size());
    const int count = int(markers.size());
    for (int i = 0; i < L; ++i)
        m.pinnedIds[i] = i < valid ? ids[size_t(i)] : m.padId;
    for (int i = 0; i < count; ++i)
        m.pinnedIds[m.maxLen + i] = markers[size_t(i)];
    m.pinnedIds[m.maxLen + kMaxOptions] = valid;
    CUDA_CHECK(cudaMemcpyAsync(
            m.ids,
            m.pinnedIds,
            size_t(L) * 4,
            cudaMemcpyHostToDevice,
            m.stream));
    CUDA_CHECK(cudaMemcpyAsync(
            m.markers,
            m.pinnedIds + m.maxLen,
            size_t(count) * 4,
            cudaMemcpyHostToDevice,
            m.stream));
    CUDA_CHECK(cudaMemcpyAsync(
            m.valid,
            m.pinnedIds + m.maxLen + kMaxOptions,
            4,
            cudaMemcpyHostToDevice,
            m.stream));
    CUDA_CHECK(cudaEventRecord(m.start, m.stream));
    // Graphs are keyed by padded length and marker count; the real token
    // count lives in device memory, so one graph serves a whole bucket.
    const int key = (L << 6) | count;
    if (useGraph) {
        auto it = m.graphs.find(key);
        if (it == m.graphs.end()) {
            // Warm plans (cuBLASLt heuristics) outside capture, then capture.
            m.forward(L, count);
            CUDA_CHECK(cudaStreamSynchronize(m.stream));
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(
                    m.stream, cudaStreamCaptureModeThreadLocal));
            m.forward(L, count);
            CUDA_CHECK(cudaStreamEndCapture(m.stream, &graph));
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            it = m.graphs.emplace(key, exec).first;
        }
        CUDA_CHECK(cudaGraphLaunch(it->second, m.stream));
    } else {
        m.forward(L, count);
    }
    CUDA_CHECK(cudaEventRecord(m.stop, m.stream));
    CUDA_CHECK(cudaStreamSynchronize(m.stream));
    ModelOutput out;
    CUDA_CHECK(cudaEventElapsedTime(&out.milliseconds, m.start, m.stop));
    out.logits.assign(m.pinnedOut, m.pinnedOut + count);
    // Action head on the host in fp32: features from the softmax of the
    // logits (temperature 1, as in the reference), then two small linears.
    std::vector<double> p(static_cast<size_t>(count));
    double peak = -1e30, total = 0;
    for (float z : out.logits)
        peak = std::max(peak, double(z));
    for (int i = 0; i < count; ++i) {
        p[size_t(i)] = std::exp(double(out.logits[size_t(i)]) - peak);
        total += p[size_t(i)];
    }
    for (auto& value : p)
        value /= total;
    double top1 = 0, top2 = 0, entropy = 0;
    for (double value : p) {
        if (value > top1) {
            top2 = top1;
            top1 = value;
        } else if (value > top2) {
            top2 = value;
        }
        entropy -= value * std::log(std::max(value, 1e-9));
    }
    const double kk = std::max(count, 2);
    std::array<float, 4> feats{ float(top1),
                                float(top1 - top2),
                                float(entropy / std::log(kk)),
                                float(kk / 255.0) };
    std::vector<float> hidden(256);
    for (int o = 0; o < 256; ++o) {
        float acc      = m.actB[size_t(o)];
        const float* w = m.actW.data() + size_t(o) * (kHidden + 4);
        for (int c = 0; c < kHidden; ++c)
            acc += w[c] * m.pinnedOut[kMaxOptions + c];
        for (int c = 0; c < 4; ++c)
            acc += w[kHidden + c] * feats[size_t(c)];
        hidden[size_t(o)] =
                0.5f * acc * (1.f + erff(acc * 0.70710678118654752440f));
    }
    float act[2];
    for (int o = 0; o < 2; ++o) {
        float acc = m.actB2[size_t(o)];
        for (int c = 0; c < 256; ++c)
            acc += m.actW2[size_t(o) * 256 + size_t(c)] * hidden[size_t(c)];
        act[o] = acc;
    }
    const float actPeak   = std::max(act[0], act[1]);
    const float e0        = std::exp(act[0] - actPeak),
                e1        = std::exp(act[1] - actPeak);
    out.actionProbability = e0 / (e0 + e1);
    return out;
}

} // namespace openzl::laya
