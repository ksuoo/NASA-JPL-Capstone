#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct PiVisionConfig {
    std::string model_path;    // Path to gguf
    std::string vision_path;   // Path to mmproj
    int         n_ctx        = 2048;
    float       temperature  = 0.1f;
    bool        verbose      = false;
};

struct RunResult {
    std::string content;           // Model response
    std::string model_desc;        // Model name
    int         images_processed = 0;
    int         prompt_tokens    = 0;  
    int         gen_tokens       = 0;  
    int         total_tokens     = 0;  // Sum of prompt tokens and generated tokens
    double      tokens_per_sec   = 0.0;
    double      prompt_ms        = 0.0;  // prompt eval time (ms)
    double      gen_ms           = 0.0;  // generation time (ms)
    double      ttft_ms          = 0.0;  // time to first token (ms)
    double      wall_ms          = 0.0;  // total time from start to finish (wall time) (ms)
};

class PiVision {
public:
    explicit PiVision(const PiVisionConfig& config);
    ~PiVision();

    PiVision(const PiVision&)            = delete;
    PiVision& operator=(const PiVision&) = delete;

    std::string validate(const std::vector<std::string>& image_paths) const;

    bool load_image(const std::string& path);

    // Streaming interface
    void run(const std::string& prompt,
             std::function<void(const std::string&)> stream_cb);

    // Batch interface – runs inference and returns the full result w/ metadata
    RunResult run_collect(const std::string& prompt);

    // Multi-turn chat using KV cache

    // Run one chat turn. Images loaded via load_image() apply to this turn
    RunResult chat_turn(const std::string& user_message,
                        std::function<void(const std::string&)> stream_cb);
    RunResult chat_turn_collect(const std::string& user_message);

    // Reset the conversation (clears KV cache + history)
    void chat_clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
