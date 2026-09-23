// Copyright (c) Meta Platforms, Inc. and affiliates.
// Core ML backend of the Laya worker (macOS). Runs the pinned
// FluidInference/laya-coreml buckets: compiled .mlmodelc bundles with
// fixed-length inputs (input_ids, attention_mask, marker_map, question_type)
// and calibration metadata, one per sequence length. A prompt runs on the
// smallest bucket that holds it, as in FluidUse; the length-512 bucket serves
// the usual 300-450 token statistics prompts about 3x faster than 1024.
// Both run on the CPU and Neural Engine.
// Inputs are encoded exactly as FluidUse 0.2.0 does, so the same prompt
// yields the same prediction; each bucket's input arrays are allocated once
// and reused across calls.
#include "cli/laya/model.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <sys/sysctl.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "tools/json.hpp"

namespace openzl::laya {
namespace {

// Ascending bucket lengths; the largest one bounds the prompt.
constexpr int kBuckets[]     = { 512, 1024 };
constexpr int kMaxOptions    = 32;
constexpr int kQuestionTypes = 3; // choice, score, noul

std::string describe(NSError* error)
{
    return error ? std::string(error.localizedDescription.UTF8String)
                 : std::string("unknown error");
}

MLMultiArray* array(NSArray<NSNumber*>* shape, MLMultiArrayDataType type)
{
    NSError* error      = nil;
    MLMultiArray* value = [[MLMultiArray alloc] initWithShape:shape
                                                     dataType:type
                                                        error:&error];
    if (!value)
        throw std::runtime_error(
                "cannot allocate Core ML input: " + describe(error));
    return value;
}

bool hasShape(
        MLFeatureDescription* feature,
        const std::vector<int>& shape,
        MLMultiArrayDataType type)
{
    MLMultiArrayConstraint* constraint = feature.multiArrayConstraint;
    if (!constraint || constraint.dataType != type
        || constraint.shape.count != shape.size())
        return false;
    for (size_t i = 0; i < shape.size(); ++i)
        if (constraint.shape[i].intValue != shape[i])
            return false;
    return true;
}

/// First `count` float32 values of a model output.
std::vector<float>
readOutput(id<MLFeatureProvider> output, NSString* name, int count)
{
    MLMultiArray* value = [output featureValueForName:name].multiArrayValue;
    if (!value || value.count != count
        || value.dataType != MLMultiArrayDataTypeFloat32)
        throw std::runtime_error(
                std::string(name.UTF8String) + " must be float32 with "
                + std::to_string(count) + " values");
    std::vector<float> values(static_cast<size_t>(count));
    // Blocks capture C++ objects by copy, so hand the block a raw pointer.
    float* destination   = values.data();
    const size_t payload = static_cast<size_t>(count) * sizeof(float);
    __block bool copied  = false;
    [value getBytesWithHandler:^(const void* bytes, NSInteger size) {
      if (size >= NSInteger(payload)) {
          std::memcpy(destination, bytes, payload);
          copied = true;
      }
    }];
    if (!copied)
        throw std::runtime_error(
                std::string(name.UTF8String) + " is truncated");
    for (float v : values)
        if (!std::isfinite(v))
            throw std::runtime_error("Model returned non-finite values");
    return values;
}

std::string chipName()
{
    char name[256] = {};
    size_t size    = sizeof(name) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", name, &size, nullptr, 0))
        return "Apple";
    return name;
}

std::string bundleName(int length)
{
    return "laya_multilingual_e8_L" + std::to_string(length)
            + "_options32.mlmodelc";
}

/// Checkpoint-level calibration; identical in every bucket of one conversion.
struct Calibration {
    int headMaxLen = 0;
    std::vector<float> temperatureByType;
    std::map<std::string, float> temperatureByOptions;

    bool operator==(const Calibration& other) const
    {
        return headMaxLen == other.headMaxLen
                && temperatureByType == other.temperatureByType
                && temperatureByOptions == other.temperatureByOptions;
    }
};

/// One loaded sequence-length bucket with its reused inputs; predictions are
/// serialized by the worker's single inference thread.
struct Bucket {
    int length                            = 0;
    MLModel* model                        = nil;
    size_t bytes                          = 0;
    MLMultiArray* inputIds                = nil;
    MLMultiArray* attentionMask           = nil;
    MLMultiArray* markerMap               = nil;
    MLMultiArray* questionType            = nil;
    MLDictionaryFeatureProvider* features = nil;

    void load(const std::string& directory, int expectedLength)
    {
        NSString* path =
                [NSString stringWithUTF8String:(directory + "/"
                                                + bundleName(expectedLength))
                                                       .c_str()];
        MLModelConfiguration* configuration =
                [[MLModelConfiguration alloc] init];
        // CPU + Neural Engine: as fast as MLComputeUnitsAll for these
        // buckets on an M4, but the compiled Neural Engine program is cached
        // by the system across processes, while the GPU path recompiles on
        // every worker start (about 11 s per bucket). The weights also stay
        // out of the worker's resident memory.
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError* error             = nil;
        model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                  configuration:configuration
                                          error:&error];
        if (!model)
            throw std::runtime_error(
                    "cannot load Core ML model: " + describe(error));
        NSNumber* size = nil;
        [[NSURL fileURLWithPath:[path stringByAppendingPathComponent:
                                                @"weights/weight.bin"]]
                getResourceValue:&size
                          forKey:NSURLFileSizeKey
                           error:nil];
        bytes = size_t(size.unsignedLongLongValue);
    }

    Calibration loadMetadata(int expectedLength)
    {
        NSDictionary* metadata = model.modelDescription.metadata;
        NSDictionary<NSString*, NSString*>* creator =
                metadata[MLModelCreatorDefinedKey];
        if (![creator isKindOfClass:[NSDictionary class]])
            throw std::runtime_error(
                    "Core ML model lacks creator-defined metadata");
        auto text = [&](NSString* key) -> std::string {
            NSString* value = creator[key];
            if (![value isKindOfClass:[NSString class]])
                throw std::runtime_error(
                        std::string("Core ML metadata ") + key.UTF8String
                        + " is missing");
            return value.UTF8String;
        };
        Calibration calibration;
        length                 = std::stoi(text(@"length"));
        calibration.headMaxLen = std::stoi(text(@"head_max_len"));
        if (std::stoi(text(@"max_options")) != kMaxOptions)
            throw std::runtime_error(
                    "expected 32 option slots in the Core ML model");
        calibration.temperatureByType =
                nlohmann::json::parse(text(@"temperature"))
                        .get<std::vector<float>>();
        if (calibration.temperatureByType.size() != kQuestionTypes)
            throw std::runtime_error(
                    "Core ML metadata temperature must list three values");
        calibration.temperatureByOptions =
                nlohmann::json::parse(text(@"temperature_by_options"))
                        .get<std::map<std::string, float>>();
        if (length != expectedLength || calibration.headMaxLen <= 0
            || calibration.headMaxLen >= length)
            throw std::runtime_error("invalid Core ML bucket metadata");
        return calibration;
    }

    void validateInterface()
    {
        MLModelDescription* description = model.modelDescription;
        NSDictionary<NSString*, MLFeatureDescription*>* inputs =
                description.inputDescriptionsByName;
        NSDictionary<NSString*, MLFeatureDescription*>* outputs =
                description.outputDescriptionsByName;
        const bool ok = inputs.count == 4
                && hasShape(inputs[@"input_ids"],
                            { 1, length },
                            MLMultiArrayDataTypeInt32)
                && hasShape(inputs[@"attention_mask"],
                            { 1, length },
                            MLMultiArrayDataTypeInt32)
                && hasShape(inputs[@"marker_map"],
                            { 1, kMaxOptions, length },
                            MLMultiArrayDataTypeFloat32)
                && hasShape(inputs[@"question_type"],
                            { 1, kQuestionTypes },
                            MLMultiArrayDataTypeFloat32)
                && hasShape(outputs[@"logits"],
                            { 1, kMaxOptions },
                            MLMultiArrayDataTypeFloat32)
                && hasShape(outputs[@"probabilities"],
                            { 1, kMaxOptions },
                            MLMultiArrayDataTypeFloat32)
                && hasShape(outputs[@"action_probabilities"],
                            { 1, 2 },
                            MLMultiArrayDataTypeFloat32);
        if (!ok)
            throw std::runtime_error("unexpected Core ML model interface");
    }

    void allocate()
    {
        NSNumber* n   = @(length);
        inputIds      = array(@[ @1, n ], MLMultiArrayDataTypeInt32);
        attentionMask = array(@[ @1, n ], MLMultiArrayDataTypeInt32);
        markerMap =
                array(@[ @1, @(kMaxOptions), n ], MLMultiArrayDataTypeFloat32);
        questionType =
                array(@[ @1, @(kQuestionTypes) ], MLMultiArrayDataTypeFloat32);
        NSError* error = nil;
        features =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                    @"input_ids" : inputIds,
                    @"attention_mask" : attentionMask,
                    @"marker_map" : markerMap,
                    @"question_type" : questionType,
                }
                                                                  error:&error];
        if (!features)
            throw std::runtime_error(
                    "cannot build Core ML inputs: " + describe(error));
    }
};

} // namespace

struct Model::Impl {
    std::vector<Bucket> buckets; // ascending length
    Calibration calibration;
    int padId = 0;
    std::string device;

    const Bucket& bucketFor(int tokens) const
    {
        for (const auto& bucket : buckets)
            if (tokens <= bucket.length)
                return bucket;
        return buckets.back();
    }
};

Model::Model(const std::string& directory, int padId)
        : impl_(std::make_unique<Impl>())
{
    @autoreleasepool {
        auto& m  = *impl_;
        m.padId  = padId;
        m.device = chipName();
        for (int length : kBuckets) {
            Bucket bucket;
            bucket.load(directory, length);
            const auto calibration = bucket.loadMetadata(length);
            // Buckets must come from one checkpoint conversion.
            if (!m.buckets.empty() && !(calibration == m.calibration))
                throw std::runtime_error(
                        "Core ML buckets come from different checkpoints");
            m.calibration = calibration;
            bucket.validateInterface();
            bucket.allocate();
            m.buckets.push_back(bucket);
        }
    }
}

Model::~Model() = default;

ModelOutput Model::infer(
        const std::vector<int>& ids,
        const std::vector<int>& markers,
        int paddedLength,
        bool /*useGraph*/)
{
    auto& m         = *impl_;
    const int valid = int(ids.size());
    const int count = int(markers.size());
    const Bucket& b = m.bucketFor(valid);
    if (ids.empty() || valid > b.length || paddedLength != b.length)
        throw std::runtime_error("invalid sequence length");
    if (markers.empty() || count > kMaxOptions)
        throw std::runtime_error("invalid marker count");
    for (int marker : markers)
        if (marker < 0 || marker >= valid)
            throw std::runtime_error("marker position out of range");
    @autoreleasepool {
        // Freshly allocated MLMultiArrays are contiguous, so the flat
        // row-major layout below matches their strides.
        auto* idPointer   = static_cast<int32_t*>(b.inputIds.dataPointer);
        auto* maskPointer = static_cast<int32_t*>(b.attentionMask.dataPointer);
        for (int i = 0; i < b.length; ++i) {
            idPointer[i]   = i < valid ? ids[size_t(i)] : m.padId;
            maskPointer[i] = i < valid ? 1 : 0;
        }
        auto* markerPointer = static_cast<float*>(b.markerMap.dataPointer);
        std::fill_n(markerPointer, size_t(kMaxOptions) * size_t(b.length), 0.f);
        for (int row = 0; row < count; ++row)
            markerPointer
                    [size_t(row) * size_t(b.length)
                     + size_t(markers[size_t(row)])] = 1.f;
        auto* typePointer = static_cast<float*>(b.questionType.dataPointer);
        typePointer[0]    = 1.f; // choice
        typePointer[1]    = 0.f;
        typePointer[2]    = 0.f;

        const auto start = std::chrono::steady_clock::now();
        NSError* error   = nil;
        id<MLFeatureProvider> prediction =
                [b.model predictionFromFeatures:b.features error:&error];
        const float milliseconds =
                std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
        if (!prediction)
            throw std::runtime_error(
                    "Core ML prediction failed: " + describe(error));
        auto logits = readOutput(prediction, @"logits", kMaxOptions);
        readOutput(prediction, @"probabilities", kMaxOptions);
        auto action = readOutput(prediction, @"action_probabilities", 2);
        logits.resize(size_t(count));
        return { std::move(logits), action[0], milliseconds };
    }
}

int Model::paddedLength(int tokens) const
{
    return impl_->bucketFor(tokens).length;
}
std::vector<int> Model::warmupLengths() const
{
    // Core ML compiles each bucket's device kernels on its first prediction
    // (seconds), so warm every bucket before readiness.
    std::vector<int> lengths;
    for (const auto& bucket : impl_->buckets)
        lengths.push_back(bucket.length);
    return lengths;
}
int Model::maxLength() const
{
    return impl_->buckets.back().length;
}
int Model::headMaxLength() const
{
    return impl_->calibration.headMaxLen;
}
int Model::maxOptions() const
{
    return kMaxOptions;
}
std::string Model::deviceName() const
{
    return impl_->device;
}
std::string Model::computeUnits() const
{
    return "cpu_and_ne";
}
std::string Model::backend() const
{
    return "coreml";
}
std::string Model::precision() const
{
    return "e8";
}
size_t Model::deviceBytes() const
{
    size_t bytes = 0;
    for (const auto& bucket : impl_->buckets)
        bytes += bucket.bytes;
    return bytes;
}

float Model::temperature(int options) const
{
    // FluidUse: the per-cardinality entry, else the choice-type temperature.
    const char* size        = options <= 2 ? "2"
                   : options <= 5          ? "3-5"
                   : options <= 10         ? "6-10"
                                           : "11+";
    const auto& calibration = impl_->calibration;
    auto it                 = calibration.temperatureByOptions.find(
            std::string("choice:") + size);
    if (it != calibration.temperatureByOptions.end())
        return it->second;
    return calibration.temperatureByType[0];
}

} // namespace openzl::laya
