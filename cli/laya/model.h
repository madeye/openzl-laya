// Copyright (c) Meta Platforms, Inc. and affiliates.
// Laya decision model (mmBERT-base encoder + typed decision head). The worker
// is shared; each platform provides one backend implementing this interface:
// - cuda/model.cu (Linux): native CUDA kernels over the upstream safetensors
//   checkpoint. GEMMs run on tensor cores through cuBLASLt with fp16 inputs
//   and fp32 accumulation; each padded prompt length replays one CUDA graph.
// - coreml/model.mm (macOS): the pinned Core ML conversion, fixed-length
//   buckets (512 and 1024 tokens) run on the CPU and Neural Engine.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace openzl::laya {

struct ModelOutput {
    std::vector<float> logits;   // one per marker
    float actionProbability = 0; // P(keep decision)
    float milliseconds      = 0; // time of the forward pass
};

class Model {
   public:
    /// Loads the backend's assets from the asset directory. `padId` is the
    /// tokenizer's pad token; the CUDA backend pads with its checkpoint's
    /// pad_token_id instead, while Core ML bundles carry no pad id.
    Model(const std::string& directory, int padId);
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    /// `ids` holds the real tokens (at most maxLength()); the sequence is
    /// padded to `paddedLength` tokens. `markers` are marker positions.
    /// `useGraph` selects CUDA graph replay and is ignored by Core ML.
    ModelOutput infer(
            const std::vector<int>& ids,
            const std::vector<int>& markers,
            int paddedLength,
            bool useGraph = true);

    /// Padded length the backend runs for a prompt of `tokens` tokens.
    int paddedLength(int tokens) const;
    /// Padded lengths to run before readiness (CUDA graphs to record, Core ML
    /// buckets to compile).
    std::vector<int> warmupLengths() const;
    int maxLength() const;
    int headMaxLength() const;
    int maxOptions() const;
    std::string deviceName() const;
    /// Compute units reported to the CLI: "cuda:<device>" or "cpu_and_ne".
    std::string computeUnits() const;
    /// Backend reported to the CLI: "native-cuda" or "coreml".
    std::string backend() const;
    std::string precision() const;
    /// Temperature for a choice question with `options` options.
    float temperature(int options) const;
    size_t deviceBytes() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace openzl::laya
