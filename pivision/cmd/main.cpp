#include "pivision.h"

#include <getopt.h>
#include <sys/utsname.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Built-in fallback defaults (Gemma 3 4B)
static const char * BUILTIN_MODEL  = "/home/jplpi/llama.cpp/models/gemma-3-4b-it-q4_k_m/gemma-3-4b-it-Q4_K_M.gguf";
static const char * BUILTIN_VISION = "/home/jplpi/llama.cpp/models/mmproj-model-f16-4B.gguf";
static const int    BUILTIN_N_CTX  = 4096;

// Configuration loaded from file
struct Config {
    std::string model_path;
    std::string vision_path;
    std::string default_image_path;
    int         default_n_ctx = 0;
    std::string log_directory;
    std::string source;  // which config file was loaded
};

// Simple JSON string value extractor (no external dependencies)
static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    size_t start = pos + 1;
    size_t end = json.find('"', start);
    if (end == std::string::npos) return "";

    return json.substr(start, end - start);
}

// Simple JSON integer value extractor
static int json_get_int(const std::string & json, const std::string & key, int default_val = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;

    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    // Parse integer
    if (pos >= json.size() || !isdigit(json[pos])) return default_val;

    return std::stoi(json.substr(pos));
}

// Parse a single config file
static Config parse_config_file(const fs::path & path) {
    Config cfg;
    std::ifstream f(path);
    if (!f) return cfg;

    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    cfg.model_path = json_get_string(json, "model_path");
    cfg.vision_path = json_get_string(json, "vision_path");
    cfg.default_image_path = json_get_string(json, "default_image_path");
    cfg.default_n_ctx = json_get_int(json, "default_n_ctx", 0);
    cfg.log_directory = json_get_string(json, "log_directory");
    cfg.source = path.string();

    return cfg;
}

// Load config with priority: explicit path > ./pivision.json > ~/.config > /etc
static Config load_config(const std::string & explicit_path = "") {
    // Priority 1: Explicit --config path
    if (!explicit_path.empty()) {
        if (!fs::exists(explicit_path)) {
            std::cerr << "warning: config file not found: " << explicit_path << "\n";
        } else {
            return parse_config_file(explicit_path);
        }
    }

    // Priority 2: Local directory ./pivision.json
    if (fs::exists("./pivision.json")) {
        return parse_config_file("./pivision.json");
    }

    // Priority 3: User config ~/.config/pivision/config.json
    const char * home = getenv("HOME");
    if (home) {
        fs::path user_cfg = fs::path(home) / ".config" / "pivision" / "config.json";
        if (fs::exists(user_cfg)) {
            return parse_config_file(user_cfg);
        }
    }

    // Priority 4: System config /etc/pivision/config.json
    if (fs::exists("/etc/pivision/config.json")) {
        return parse_config_file("/etc/pivision/config.json");
    }

    return Config{};  // No config found
}

static void usage(const char * prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --prompt <text> [options]          Single-shot mode\n"
        << "  " << prog << " --chat [--prompt <text>] [options] Interactive chat\n"
        << "  " << prog << " --check-health                     Verify system health\n"
        << "\nOptions:\n"
        << "  --model <llm.gguf>     LLM model\n"
        << "  --vision <proj.gguf>   Vision projector\n"
        << "  --image <img>          Image file (repeatable)\n"
        << "  --prompt <text>        Initial prompt (in chat mode, processed first)\n"
        << "  --config <file>        Config file path\n"
        << "  --json                 JSON output (single-shot only)\n"
        << "  --verbose              Print stats (wall time, TTFT, tok/s)\n"
        << "  --check-health         Check system thermal, RAM, and library status\n"
        << "\nConfig file priority:\n"
        << "  1. --config <path>              (explicit)\n"
        << "  2. ./pivision.json              (local directory)\n"
        << "  3. ~/.config/pivision/config.json (user)\n"
        << "  4. /etc/pivision/config.json    (system)\n"
        << "\nChat commands:\n"
        << "  /image <path>          Load an image for the next message\n"
        << "  /clear                 Reset conversation\n"
        << "  /quit                  Exit\n";
}

// Minimal JSON helpers – avoids pulling in a library.
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x",
                             static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void print_json_error(const std::string & msg) {
    std::cout << "{\"error\":\"" << json_escape(msg) << "\"}\n";
}

// Global log directory (set from config)
static std::string g_log_directory;

// Save a comprehensive session log
static void save_log(const std::string & prompt,
                     const std::vector<std::string> & images,
                     const RunResult & r) {
    // Determine log directory
    fs::path log_dir;
    if (!g_log_directory.empty()) {
        log_dir = g_log_directory;
    } else {
        const char * home = getenv("HOME");
        if (!home) return;
        log_dir = fs::path(home) / "pivision_logs";
    }

    fs::create_directories(log_dir);

    // timestamp filename: session_YYYYMMDD_HHMMSS.log
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    char fname[48];
    std::strftime(fname, sizeof(fname), "session_%Y%m%d_%H%M%S.log", &tm);

    std::ofstream f(log_dir / fname);
    if (!f) return;

    // Session header
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm);
    f << "================================================================================\n";
    f << "PiVision Session Log\n";
    f << "Timestamp: " << timestamp << "\n";
    f << "================================================================================\n\n";

    // Model info
    f << "[MODEL]\n";
    f << "Description: " << r.model_desc << "\n";
    f << "Images processed: " << r.images_processed << "\n\n";

    // Images
    if (!images.empty()) {
        f << "[IMAGES]\n";
        for (size_t i = 0; i < images.size(); ++i) {
            f << "  " << (i + 1) << ". " << images[i] << "\n";
        }
        f << "\n";
    }

    // Prompt
    f << "[PROMPT]\n";
    f << prompt << "\n\n";

    // Performance metrics
    f << "[PERFORMANCE]\n";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f", r.tokens_per_sec);
    f << "Tokens/sec (generation): " << buf << "\n";
    f << "Prompt tokens: " << r.prompt_tokens << "\n";
    f << "Generated tokens: " << r.gen_tokens << "\n";
    f << "Total tokens: " << r.total_tokens << "\n";
    snprintf(buf, sizeof(buf), "%.1f", r.prompt_ms);
    f << "Prompt eval time: " << buf << " ms\n";
    snprintf(buf, sizeof(buf), "%.1f", r.gen_ms);
    f << "Generation time: " << buf << " ms\n";
    snprintf(buf, sizeof(buf), "%.1f", r.ttft_ms);
    f << "Time to first token: " << buf << " ms\n";
    snprintf(buf, sizeof(buf), "%.1f", r.wall_ms / 1000.0);
    f << "Total wall time: " << buf << " s\n\n";

    // Response
    f << "[RESPONSE]\n";
    f << r.content << "\n\n";

    f << "================================================================================\n";
}

static void print_stats(const RunResult & r) {
    fprintf(stderr,
        "\n--- stats -----------------------------------------------\n"
        "  model:          %s\n"
        "  images:         %d\n"
        "  prompt tokens:  %d  (%.1f ms, %.1f tok/s)\n"
        "  gen tokens:     %d  (%.1f ms, %.1f tok/s)\n"
        "  ttft:           %.0f ms\n"
        "  wall time:      %.1f s\n"
        "---------------------------------------------------------\n",
        r.model_desc.c_str(),
        r.images_processed,
        r.prompt_tokens, r.prompt_ms,
        (r.prompt_ms > 0.0) ? r.prompt_tokens / (r.prompt_ms / 1000.0) : 0.0,
        r.gen_tokens, r.gen_ms, r.tokens_per_sec,
        r.ttft_ms,
        r.wall_ms / 1000.0);
}

static void print_json_result(const RunResult & r) {
    char tok_sec[32], wall_sec[32];
    snprintf(tok_sec,  sizeof(tok_sec),  "%.1f", r.tokens_per_sec);
    snprintf(wall_sec, sizeof(wall_sec), "%.1f", r.wall_ms / 1000.0);

    std::cout
        << "{\n"
        << "  \"content\": \"" << json_escape(r.content) << "\",\n"
        << "  \"metadata\": {\n"
        << "    \"model\": \""            << json_escape(r.model_desc) << "\",\n"
        << "    \"images_processed\": "   << r.images_processed  << ",\n"
        << "    \"prompt_tokens\": "      << r.prompt_tokens     << ",\n"
        << "    \"gen_tokens\": "         << r.gen_tokens        << ",\n"
        << "    \"total_tokens\": "       << r.total_tokens      << ",\n"
        << "    \"tokens_per_sec\": "     << tok_sec             << ",\n"
        << "    \"ttft_ms\": "            << static_cast<int>(r.ttft_ms) << ",\n"
        << "    \"wall_time_sec\": "      << wall_sec            << "\n"
        << "  }\n"
        << "}\n";
}

// Check system health: thermal, RAM, libraries
static int check_health() {
    std::cout << "PiVision Health Check\n";
    std::cout << "=====================\n\n";

    bool all_ok = true;

    // 0. Detect device
    std::cout << "System:\n";
    std::ifstream model_file("/proc/device-tree/model");
    if (model_file) {
        std::string model;
        std::getline(model_file, model);
        // Remove null bytes
        model.erase(std::remove(model.begin(), model.end(), '\0'), model.end());
        std::cout << "  Device: " << model << "\n";
    }
    struct utsname uts;
    if (uname(&uts) == 0) {
        std::cout << "  Arch:   " << uts.machine << "\n";
    }

    // 1. Check thermal
    std::cout << "\nThermal Status:\n";
    std::ifstream thermal("/sys/class/thermal/thermal_zone0/temp");
    if (thermal) {
        int temp_milli;
        thermal >> temp_milli;
        float temp = temp_milli / 1000.0f;
        std::cout << "  CPU Temperature: " << temp << " C";
        if (temp > 80) {
            std::cout << " [WARNING: High temperature!]\n";
            all_ok = false;
        } else if (temp > 70) {
            std::cout << " [Warm]\n";
        } else {
            std::cout << " [OK]\n";
        }
    } else {
        std::cout << "  Unable to read thermal sensor (not available on all systems)\n";
    }

    // 2. Check RAM
    std::cout << "\nMemory Status:\n";
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo) {
        std::string line;
        long total_kb = 0, avail_kb = 0;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                sscanf(line.c_str(), "MemTotal: %ld", &total_kb);
            } else if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld", &avail_kb);
            }
        }
        float total_gb = total_kb / 1024.0f / 1024.0f;
        float avail_gb = avail_kb / 1024.0f / 1024.0f;
        std::cout << "  Total RAM:     " << total_gb << " GB\n";
        std::cout << "  Available RAM: " << avail_gb << " GB";
        if (avail_gb < 2.0) {
            std::cout << " [WARNING: Low memory]\n";
            all_ok = false;
        } else {
            std::cout << " [OK]\n";
        }
    }

    // 3. Check llama.cpp libraries
    std::cout << "\nLibrary Status:\n";
    const char * ld_path = getenv("LD_LIBRARY_PATH");
    if (ld_path) {
        std::cout << "  LD_LIBRARY_PATH: " << ld_path << "\n";
    } else {
        std::cout << "  LD_LIBRARY_PATH: (not set)\n";
    }

    // Check common library locations
    std::vector<std::string> lib_paths = {
        "/home/jplpi/llama.cpp/build/bin/libllama.so",
        "/usr/local/lib/libllama.so",
        "/usr/lib/libllama.so"
    };
    bool lib_found = false;
    for (const auto & p : lib_paths) {
        if (fs::exists(p)) {
            std::cout << "  libllama.so: " << p << " [FOUND]\n";
            lib_found = true;
            break;
        }
    }
    if (!lib_found) {
        std::cout << "  libllama.so: [NOT FOUND in standard locations]\n";
        all_ok = false;
    }

    // 4. Check model files
    std::cout << "\nModel Status:\n";
    Config cfg = load_config();
    if (!cfg.source.empty()) {
        std::cout << "  Config loaded: " << cfg.source << "\n";
    }
    if (!cfg.model_path.empty()) {
        std::cout << "  Model: " << cfg.model_path;
        if (fs::exists(cfg.model_path)) {
            auto size = fs::file_size(cfg.model_path);
            std::cout << " [" << (size / 1024 / 1024) << " MB]\n";
        } else {
            std::cout << " [NOT FOUND]\n";
            all_ok = false;
        }
    }

    std::cout << "\n";
    if (all_ok) {
        std::cout << "Status: All checks passed!\n";
        return 0;
    } else {
        std::cout << "Status: Some issues detected.\n";
        return 1;
    }
}

int main(int argc, char * argv[]) {
    std::string model, vision, prompt, config_path;
    std::vector<std::string> images;
    bool json_mode    = false;
    bool verbose      = false;
    bool chat_mode    = false;
    bool check_health_mode = false;

    static struct option long_opts[] = {
        {"model",        required_argument, nullptr, 'm'},
        {"vision",       required_argument, nullptr, 'v'},
        {"image",        required_argument, nullptr, 'i'},
        {"prompt",       required_argument, nullptr, 'p'},
        {"config",       required_argument, nullptr, 'C'},
        {"chat",         no_argument,       nullptr, 'c'},
        {"json",         no_argument,       nullptr, 'j'},
        {"verbose",      no_argument,       nullptr, 'V'},
        {"check-health", no_argument,       nullptr, 'H'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:v:i:p:C:cjVHh",
                              long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model  = optarg;               break;
            case 'v': vision = optarg;               break;
            case 'i': images.emplace_back(optarg);   break;
            case 'p': prompt = optarg;               break;
            case 'C': config_path = optarg;          break;
            case 'c': chat_mode = true;              break;
            case 'j': json_mode = true;              break;
            case 'V': verbose   = true;              break;
            case 'H': check_health_mode = true;      break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    // Handle --check-health mode
    if (check_health_mode) {
        return check_health();
    }

    // If prompt looks like a file path and the file exists, read its contents
    if (!prompt.empty() && fs::is_regular_file(prompt)) {
        std::ifstream pf(prompt);
        if (pf) {
            prompt.assign(std::istreambuf_iterator<char>(pf),
                          std::istreambuf_iterator<char>());
        }
    }

    // Load config file and apply priority: CLI > config > built-in
    Config file_cfg = load_config(config_path);

    if (verbose && !file_cfg.source.empty()) {
        std::cerr << "config loaded: " << file_cfg.source << "\n";
    }

    // Set global log directory from config
    if (!file_cfg.log_directory.empty()) {
        g_log_directory = file_cfg.log_directory;
    }

    if (model.empty()) {
        if (!file_cfg.model_path.empty() && fs::exists(file_cfg.model_path)) {
            model = file_cfg.model_path;
            if (!json_mode) std::cerr << "using config model: " << model << "\n";
        } else if (fs::exists(BUILTIN_MODEL)) {
            model = BUILTIN_MODEL;
            if (!json_mode) std::cerr << "using default model: " << model << "\n";
        }
    }

    if (vision.empty()) {
        if (!file_cfg.vision_path.empty() && fs::exists(file_cfg.vision_path)) {
            vision = file_cfg.vision_path;
            if (!json_mode) std::cerr << "using config vision: " << vision << "\n";
        } else if (fs::exists(BUILTIN_VISION)) {
            vision = BUILTIN_VISION;
            if (!json_mode) std::cerr << "using default vision: " << vision << "\n";
        }
    }

    // Apply default image from config if no images specified
    if (images.empty() && !file_cfg.default_image_path.empty()
        && fs::exists(file_cfg.default_image_path)) {
        images.push_back(file_cfg.default_image_path);
        if (!json_mode) std::cerr << "using config image: " << file_cfg.default_image_path << "\n";
    }

    if (!chat_mode && prompt.empty()) {
        if (json_mode) {
            print_json_error("missing required argument: --prompt");
            return 1;
        }
        usage(argv[0]);
        return 1;
    }

    if (chat_mode && json_mode) {
        std::cerr << "error: --chat and --json cannot be combined\n";
        return 1;
    }

    // Auto-detect vision projector if --vision omitted but images given
    // or chat mode (where images may be loaded later via /image)
    if ((!images.empty() || chat_mode) && vision.empty()) {
        fs::path model_dir = fs::path(model).parent_path();
        std::vector<std::string> candidates;
        for (const auto & entry : fs::directory_iterator(model_dir)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("mmproj", 0) == 0 && name.size() > 5
                && name.substr(name.size() - 5) == ".gguf") {
                candidates.push_back(entry.path().string());
            }
        }
        if (candidates.size() == 1) {
            vision = candidates[0];
            if (!json_mode) {
                std::cerr << "auto-detected vision projector: "
                          << vision << "\n";
            }
        } else {
            std::string msg = candidates.empty()
                ? "no mmproj*.gguf found in " + model_dir.string()
                  + " – provide --vision explicitly"
                : "multiple mmproj*.gguf found in " + model_dir.string()
                  + " – provide --vision to pick one";
            if (chat_mode && candidates.empty()) {
                // No projector available — vision just won't work, that's ok
                std::cerr << "note: " << msg << "\n";
            } else {
                if (json_mode) { print_json_error(msg); }
                else           { std::cerr << "error: " << msg << "\n"; }
                return 1;
            }
        }
    }

    try {
        // -- Construct engine ------------------------------------------------
        PiVisionConfig cfg;
        cfg.model_path  = model;
        cfg.vision_path = vision;
        cfg.verbose     = verbose;

        PiVision pv(cfg);

        if (chat_mode) {
            // -- Interactive chat REPL ------------------------------------------
            std::vector<std::string> turn_images;

            // Load any images passed via --image on the command line
            if (!images.empty()) {
                std::string err = pv.validate(images);
                if (!err.empty()) {
                    std::cerr << "error: " << err << "\n";
                    return 1;
                }
                for (size_t idx = 0; idx < images.size(); ++idx) {
                    const auto & img = images[idx];
                    if (!pv.load_image(img)) {
                        std::cerr << "failed to load image: " << img << "\n";
                        return 1;
                    }
                    if (verbose) {
                        std::cerr << "Image " << (idx + 1) << ": " << img << "\n";
                    }
                }
                turn_images = images;
            }

            std::cout << "pivision chat (type /quit to exit, /help for commands)\n\n";

            // Process initial prompt if provided via --prompt
            if (!prompt.empty()) {
                std::cout << "> " << prompt << "\n";
                RunResult turn_result = pv.chat_turn(prompt,
                    [](const std::string & piece) {
                        std::cout << piece << std::flush;
                    });
                std::cout << "\n\n";

                if (verbose) print_stats(turn_result);
                save_log(prompt, turn_images, turn_result);
                turn_images.clear();
            }

            std::string line;
            while (true) {
                std::cout << "> " << std::flush;
                if (!std::getline(std::cin, line)) break;  // EOF

                // Trim leading/trailing whitespace
                size_t start = line.find_first_not_of(" \t");
                if (start == std::string::npos) continue;  // blank line
                line = line.substr(start);

                // -- Chat commands ----------------------------------------------
                if (line == "/quit" || line == "/exit") break;

                if (line == "/clear") {
                    pv.chat_clear();
                    turn_images.clear();
                    std::cout << "conversation cleared\n\n";
                    continue;
                }

                if (line == "/help") {
                    std::cout << "Commands:\n"
                              << "  /image <path>  Load an image for the next message\n"
                              << "  /clear         Reset conversation\n"
                              << "  /quit          Exit\n\n";
                    continue;
                }

                if (line.rfind("/image ", 0) == 0) {
                    std::string img_path = line.substr(7);
                    // Trim whitespace from path
                    size_t ps = img_path.find_first_not_of(" \t");
                    if (ps != std::string::npos) img_path = img_path.substr(ps);

                    std::string err = pv.validate({img_path});
                    if (!err.empty()) {
                        std::cerr << "error: " << err << "\n";
                        continue;
                    }

                    if (!pv.load_image(img_path)) {
                        std::cerr << "failed to load image: " << img_path << "\n";
                        continue;
                    }
                    turn_images.push_back(img_path);
                    std::cout << "loaded: " << img_path << "\n\n";
                    continue;
                }

                // -- Send user message ------------------------------------------
                RunResult turn_result = pv.chat_turn(line,
                    [](const std::string & piece) {
                        std::cout << piece << std::flush;
                    });
                std::cout << "\n\n";

                if (verbose) print_stats(turn_result);
                save_log(line, turn_images, turn_result);
                turn_images.clear();
            }
        } else {
            // -- Single-shot mode -----------------------------------------------
            if (!images.empty()) {
                std::string err = pv.validate(images);
                if (!err.empty()) {
                    if (json_mode) { print_json_error(err); }
                    else           { std::cerr << "error: " << err << "\n"; }
                    return 1;
                }

                for (size_t idx = 0; idx < images.size(); ++idx) {
                    const auto & img = images[idx];
                    if (!pv.load_image(img)) {
                        std::string msg = "failed to load image: " + img;
                        if (json_mode) { print_json_error(msg); }
                        else           { std::cerr << msg << "\n"; }
                        return 1;
                    }
                    if (verbose) {
                        std::cerr << "Image " << (idx + 1) << ": " << img << "\n";
                    }
                }
            }

            RunResult result;
            if (json_mode) {
                result = pv.run_collect(prompt);
                print_json_result(result);
            } else {
                result = pv.run_collect(prompt);
                std::cout << result.content << "\n";
            }

            if (verbose) print_stats(result);
            save_log(prompt, images, result);
        }

    } catch (const std::exception & e) {
        if (json_mode) { print_json_error(e.what()); }
        else           { std::cerr << "error: " << e.what() << "\n"; }
        return 1;
    }

    return 0;
}
