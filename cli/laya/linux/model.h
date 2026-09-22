// Copyright (c) Meta Platforms, Inc. and affiliates.
// Laya decision model (mmBERT-base encoder + typed decision head) on CUDA.
// GEMMs run on tensor cores through cuBLASLt with fp16 inputs (the checkpoint
// stores fp16 weights, so they are exact) and fp32 accumulation; layer norms,
// the residual stream, softmax and the heads stay in fp32. Each padded prompt
// length replays one CUDA graph.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace openzl::laya {

struct ModelOutput {
    std::vector<float> logits;   // one per marker
    float actionProbability = 0; // P(keep decision)
    float milliseconds      = 0; // device time of the forward pass
};

class Model {
   public:
    /// Loads rl_agent_config.json, encoder/config.json and model.safetensors
    /// from the asset directory onto the current CUDA device.
    explicit Model(const std::string& directory);
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    /// `ids` holds the real tokens (at most maxLength()); the sequence is
    /// padded to `paddedLength` tokens. `markers` are marker positions.
    ModelOutput infer(
            const std::vector<int>& ids,
            const std::vector<int>& markers,
            int paddedLength,
            bool useGraph = true);

    int maxLength() const;
    int headMaxLength() const;
    int maxOptions() const;
    std::string deviceName() const;
    std::string precision() const;
    /// Temperature for a choice question with `options` options.
    float temperature(int options) const;
    size_t deviceBytes() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace openzl::laya
