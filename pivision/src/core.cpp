// pivision – multimodal inference engine
// Single translation unit for all llama.cpp internals.

#include "pivision.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "llama.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void quiet_log_callback(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR)
        fputs(text, stderr);
}

static bool is_valid_image_format(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    unsigned char hdr[4] = {};
    f.read(reinterpret_cast<char *>(hdr), 4);
    if (!f) return false;

    if (hdr[0] == 0xFF && hdr[1] == 0xD8 && hdr[2] == 0xFF) return true; // JPEG
    if (hdr[0] == 0x89 && hdr[1] == 0x50 && hdr[2] == 0x4E && hdr[3] == 0x47) return true; // PNG
    return false;
}

// Formats a single-shot user prompt using the model's embedded chat template.
static std::string format_chat_prompt(const char *chat_template,
                                      const std::string &user_prompt,
                                      int n_images,
                                      const std::string &media_marker) {
    std::string content;
    content.reserve(256 + user_prompt.size());

    for (int i = 0; i < n_images; ++i) {
        content += media_marker;
        content += '\n';
    }
    content += user_prompt;

    llama_chat_message msg = { "user", content.c_str() };
    int32_t len = llama_chat_apply_template(chat_template, &msg, 1, true, nullptr, 0);

    if (len <= 0) {
        // Fallback if the template isn't usable
        return "user\n" + content + "\nassistant\n";
    }

    std::vector<char> buf(static_cast<size_t>(len) + 1);
    llama_chat_apply_template(chat_template, &msg, 1, true, buf.data(), buf.size());
    return std::string(buf.data(), static_cast<size_t>(len));
}

struct PiVision::Impl {
    PiVisionConfig config;

    llama_model       *model    = nullptr;
    llama_context     *ctx      = nullptr;
    const llama_vocab *vocab    = nullptr;
    mtmd_context      *mtmd_ctx = nullptr;
    llama_sampler     *sampler  = nullptr;

    std::vector<mtmd::bitmap> bitmaps;
    std::string model_desc;
    std::string chat_template;

    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;

    llama_pos n_past = 0;

    explicit Impl(const PiVisionConfig &cfg) : config(cfg) {
        // Suppress llama.cpp and vision encoder log spam globally
        llama_log_set(quiet_log_callback, nullptr);
        mtmd_helper_log_set(quiet_log_callback, nullptr);

        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        model = llama_model_load_from_file(config.model_path.c_str(), mparams);
        if (!model)
            throw std::runtime_error("pivision: failed to load LLM from " + config.model_path);

        vocab = llama_model_get_vocab(model);

        char desc_buf[256] = {};
        llama_model_desc(model, desc_buf, sizeof(desc_buf));
        model_desc = desc_buf;

        const char *tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl)
            chat_template = tmpl;

        tmpls = common_chat_templates_init(model, "");

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx    = static_cast<uint32_t>(config.n_ctx);
        cparams.n_batch  = 512;
        cparams.n_ubatch = 512;
        cparams.no_perf  = false;

        ctx = llama_init_from_model(model, cparams);
        if (!ctx)
            throw std::runtime_error("pivision: failed to create llama context");

        if (!config.vision_path.empty()) {
            mtmd_context_params mp = mtmd_context_params_default();
            mp.use_gpu       = false;
            mp.n_threads     = 4;
            mp.print_timings = false;

            mtmd_ctx = mtmd_init_from_file(config.vision_path.c_str(), model, mp);
            if (!mtmd_ctx)
                throw std::runtime_error("pivision: failed to load vision projector from "
                                         + config.vision_path);
        }

        build_sampler();
    }

    ~Impl() {
        if (sampler)  llama_sampler_free(sampler);
        if (mtmd_ctx) mtmd_free(mtmd_ctx);
        if (ctx)      llama_free(ctx);
        if (model)    llama_model_free(model);
        llama_backend_free();
    }

    void build_sampler() {
        auto sparams = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));
    }

    std::string validate(const std::vector<std::string> &paths) const {
        if (!mtmd_ctx)
            return "vision projector not loaded – provide --vision to use images";

        if (!mtmd_support_vision(mtmd_ctx))
            return "vision projector does not support vision input – is it compatible with this LLM?";

        for (const auto &p : paths) {
            std::ifstream f(p);
            if (!f.good())
                return "image file not found: " + p;
            if (!is_valid_image_format(p))
                return "unsupported image format (expected JPG or PNG): " + p;
        }
        return {};
    }

    bool load_image(const std::string &path) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx, path.c_str());
        if (!bmp) {
            fprintf(stderr, "[pivision] failed to load image: %s\n", path.c_str());
            return false;
        }
        bitmaps.emplace_back(bmp);
        return true;
    }

    // Tokenise and eval a formatted prompt. Pass add_bos=true for the first turn.
    void eval_prompt(const std::string &formatted, bool add_bos) {
        const int n_images = static_cast<int>(bitmaps.size());

        if (n_images > 0 && mtmd_ctx) {
            mtmd_input_text text;
            text.text          = formatted.c_str();
            text.add_special   = add_bos;
            text.parse_special = true;

            mtmd::input_chunks chunks(mtmd_input_chunks_init());

            std::vector<const mtmd_bitmap *> bmp_ptrs;
            bmp_ptrs.reserve(bitmaps.size());
            for (auto &b : bitmaps)
                bmp_ptrs.push_back(b.ptr.get());

            int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks.ptr.get(),
                                            &text, bmp_ptrs.data(), bmp_ptrs.size());
            if (tok_res != 0)
                throw std::runtime_error("pivision: mtmd_tokenize failed (code "
                                         + std::to_string(tok_res) + ")");

            bitmaps.clear();

            llama_pos new_n_past = 0;
            int32_t eval_res = mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks.ptr.get(),
                                                       n_past, 0, 512, true, &new_n_past);
            if (eval_res != 0)
                throw std::runtime_error("pivision: mtmd_helper_eval_chunks failed (code "
                                         + std::to_string(eval_res) + ")");

            n_past = new_n_past;
        } else {
            std::vector<llama_token> tokens(formatted.size() + 64);
            int n = llama_tokenize(vocab, formatted.c_str(), formatted.size(),
                                   tokens.data(), tokens.size(), add_bos, true);
            if (n < 0) {
                tokens.resize(-n);
                n = llama_tokenize(vocab, formatted.c_str(), formatted.size(),
                                   tokens.data(), tokens.size(), add_bos, true);
            }
            tokens.resize(n);

            llama_batch batch = llama_batch_get_one(tokens.data(), n);
            if (llama_decode(ctx, batch))
                throw std::runtime_error("pivision: failed to eval text prompt");

            n_past += n;
        }
    }

    std::string sample_response(std::function<void(const std::string &)> stream_cb) {
        std::string content;
        const int max_tokens = config.n_ctx - static_cast<int>(n_past);

        for (int i = 0; i < max_tokens; ++i) {
            llama_token id = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;

            char buf[256];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                std::string piece(buf, static_cast<size_t>(n));
                content += piece;
                if (stream_cb) stream_cb(piece);
            }

            llama_batch batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "[pivision] decode failed at token %d\n", i);
                break;
            }
            ++n_past;
        }
        return content;
    }

    void run_inner(const std::string &prompt,
                   std::function<void(const std::string &)> stream_cb,
                   RunResult &out) {
        namespace chr = std::chrono;
        auto wall_start = chr::steady_clock::now();

        const int n_images = static_cast<int>(bitmaps.size());

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_perf_context_reset(ctx);
        n_past = 0;

        const std::string marker = mtmd_ctx ? std::string(mtmd_default_marker()) : std::string();
        const char *tmpl = chat_template.empty() ? nullptr : chat_template.c_str();
        std::string full_prompt = format_chat_prompt(tmpl, prompt, n_images, marker);

        eval_prompt(full_prompt, true);

        auto ttft_start = chr::steady_clock::now();
        bool ttft_recorded = false;
        double ttft_ms = 0.0;

        auto wrapped_cb = [&](const std::string &piece) {
            if (!ttft_recorded) {
                ttft_ms = chr::duration<double, std::milli>(
                    chr::steady_clock::now() - ttft_start).count();
                ttft_recorded = true;
            }
            if (stream_cb) stream_cb(piece);
        };

        out.content = sample_response(wrapped_cb);

        auto wall_end = chr::steady_clock::now();
        auto perf = llama_perf_context(ctx);

        out.model_desc       = model_desc;
        out.images_processed = n_images;
        out.prompt_tokens    = perf.n_p_eval;
        out.gen_tokens       = perf.n_eval;
        out.total_tokens     = perf.n_p_eval + perf.n_eval;
        out.prompt_ms        = perf.t_p_eval_ms;
        out.gen_ms           = perf.t_eval_ms;
        out.ttft_ms          = ttft_ms;
        out.wall_ms          = chr::duration<double, std::milli>(wall_end - wall_start).count();

        double gen_sec     = perf.t_eval_ms / 1000.0;
        out.tokens_per_sec = gen_sec > 0.0 ? static_cast<double>(perf.n_eval) / gen_sec : 0.0;
    }

    void chat_turn_inner(const std::string &user_message,
                         std::function<void(const std::string &)> stream_cb,
                         RunResult &out) {
        namespace chr = std::chrono;
        auto wall_start = chr::steady_clock::now();

        const int n_images = static_cast<int>(bitmaps.size());
        bool is_first = chat_history.empty();

        const std::string marker = mtmd_ctx ? std::string(mtmd_default_marker()) : std::string();
        std::string content;
        for (int i = 0; i < n_images; ++i) {
            content += marker;
            content += '\n';
        }
        content += user_message;

        common_chat_msg user_msg;
        user_msg.role    = "user";
        user_msg.content = content;

        std::string formatted = common_chat_format_single(
            tmpls.get(), chat_history, user_msg, true, false);

        chat_history.push_back(user_msg);
        llama_perf_context_reset(ctx);
        eval_prompt(formatted, is_first);

        auto ttft_start = chr::steady_clock::now();
        bool ttft_recorded = false;
        double ttft_ms = 0.0;

        auto wrapped_cb = [&](const std::string &piece) {
            if (!ttft_recorded) {
                ttft_ms = chr::duration<double, std::milli>(
                    chr::steady_clock::now() - ttft_start).count();
                ttft_recorded = true;
            }
            if (stream_cb) stream_cb(piece);
        };

        out.content = sample_response(wrapped_cb);

        auto wall_end = chr::steady_clock::now();

        common_chat_msg asst_msg;
        asst_msg.role    = "assistant";
        asst_msg.content = out.content;
        chat_history.push_back(asst_msg);

        auto perf = llama_perf_context(ctx);

        out.model_desc       = model_desc;
        out.images_processed = n_images;
        out.prompt_tokens    = perf.n_p_eval;
        out.gen_tokens       = perf.n_eval;
        out.total_tokens     = perf.n_p_eval + perf.n_eval;
        out.prompt_ms        = perf.t_p_eval_ms;
        out.gen_ms           = perf.t_eval_ms;
        out.ttft_ms          = ttft_ms;
        out.wall_ms          = chr::duration<double, std::milli>(wall_end - wall_start).count();

        double gen_sec     = perf.t_eval_ms / 1000.0;
        out.tokens_per_sec = gen_sec > 0.0 ? static_cast<double>(perf.n_eval) / gen_sec : 0.0;
    }

    void chat_clear_inner() {
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past = 0;
        chat_history.clear();
    }
};

// ---------------------------------------------------------------------------

PiVision::PiVision(const PiVisionConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

PiVision::~PiVision() = default;

std::string PiVision::validate(const std::vector<std::string> &image_paths) const {
    return impl_->validate(image_paths);
}

bool PiVision::load_image(const std::string &path) {
    return impl_->load_image(path);
}

void PiVision::run(const std::string &prompt,
                   std::function<void(const std::string &)> stream_cb) {
    RunResult unused;
    impl_->run_inner(prompt, std::move(stream_cb), unused);
}

RunResult PiVision::run_collect(const std::string &prompt) {
    RunResult result;
    impl_->run_inner(prompt, nullptr, result);
    return result;
}

RunResult PiVision::chat_turn(const std::string &user_message,
                               std::function<void(const std::string &)> stream_cb) {
    RunResult result;
    impl_->chat_turn_inner(user_message, std::move(stream_cb), result);
    return result;
}

RunResult PiVision::chat_turn_collect(const std::string &user_message) {
    RunResult result;
    impl_->chat_turn_inner(user_message, nullptr, result);
    return result;
}

void PiVision::chat_clear() {
    impl_->chat_clear_inner();
}
