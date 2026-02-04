#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct PiVisionConfig {
    std::string model_path;    // Path to the LLM GGUF
    std::string vision_path;   // Path to the vision projector GGUF
    int         n_ctx        = 2048;
    float       temperature  = 0.1f;
    bool        verbose      = false;
};

struct RunResult {
    std::string content;           // Full model response
    std::string model_desc;        // e.g. "gemma-3-4b"
    int         images_processed = 0;
    int         prompt_tokens    = 0;  // tokens in the prompt
    int         gen_tokens       = 0;  // tokens generated
    int         total_tokens     = 0;  // prompt + generated
    double      tokens_per_sec   = 0.0;
    double      prompt_ms        = 0.0;  // prompt eval time (ms)
    double      gen_ms           = 0.0;  // generation time (ms)
    double      ttft_ms          = 0.0;  // time to first token (ms)
    double      wall_ms          = 0.0;  // total wall clock time (ms)
};

class PiVision {
public:
    explicit PiVision(const PiVisionConfig& config);
    ~PiVision();

    PiVision(const PiVision&)            = delete;
    PiVision& operator=(const PiVision&) = delete;

    // Validate that image files exist, are readable JPG/PNG, and
    // that the vision projector is compatible with the loaded LLM.
    // Returns empty string on success, or a human-readable error.
    std::string validate(const std::vector<std::string>& image_paths) const;

    bool load_image(const std::string& path);

    // Streaming interface (unchanged).
    void run(const std::string& prompt,
             std::function<void(const std::string&)> stream_cb);

    // Batch interface – runs inference and returns the full result
    // with metadata.  Preferred when --json is used by the CLI.
    RunResult run_collect(const std::string& prompt);

    // --- Multi-turn chat (KV cache persists between turns) ----------------

    // Run one chat turn. Images loaded via load_image() apply to this turn.
    RunResult chat_turn(const std::string& user_message,
                        std::function<void(const std::string&)> stream_cb);
    RunResult chat_turn_collect(const std::string& user_message);

    // Reset conversation (clears KV cache + history).
    void chat_clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
