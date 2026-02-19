// log_to_csv.cpp – Scrape PiVision session log directory and export to CSV.
// Standalone: no dependency on pivision library or llama.cpp.
// Usage: log_to_csv [--log-dir <path>] [--config <path>] [--output <file.csv>]

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Config / log directory (same priority as pivision main)
// ---------------------------------------------------------------------------

static std::string json_get_string(const std::string& json, const std::string& key) {
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

static std::string get_default_log_dir() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
#else
    const char* home = std::getenv("HOME");
#endif
    if (!home) return "";
    return (fs::path(home) / "pivision_logs").string();
}

static std::string load_log_directory_from_config(const std::string& config_path) {
    if (config_path.empty() || !fs::exists(config_path)) return "";
    std::ifstream f(config_path);
    if (!f) return "";
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return json_get_string(json, "log_directory");
}

static std::string resolve_log_dir(const std::string& explicit_log_dir,
                                   const std::string& config_path) {
    if (!explicit_log_dir.empty()) return explicit_log_dir;
    std::string from_config = load_log_directory_from_config(config_path);
    if (!from_config.empty()) return from_config;
    // Try config in standard locations
    if (config_path.empty()) {
        if (fs::exists("./pivision.json"))
            from_config = load_log_directory_from_config("./pivision.json");
        if (from_config.empty()) {
#ifdef _WIN32
            const char* home = std::getenv("USERPROFILE");
            if (!home) home = std::getenv("HOME");
#else
            const char* home = std::getenv("HOME");
#endif
            if (home) {
                fs::path user_cfg = fs::path(home) / ".config" / "pivision" / "config.json";
                if (fs::exists(user_cfg))
                    from_config = load_log_directory_from_config(user_cfg.string());
            }
        }
        if (!from_config.empty()) return from_config;
    }
    return get_default_log_dir();
}

// ---------------------------------------------------------------------------
// Parsed session record (matches save_log() format in main.cpp)
// ---------------------------------------------------------------------------

struct SessionRecord {
    std::string timestamp;
    std::string model_description;
    int         images_processed = 0;
    std::string image_paths;      // semicolon-separated
    std::string prompt;
    double      tokens_per_sec    = 0.0;
    int         prompt_tokens     = 0;
    int         gen_tokens        = 0;
    int         total_tokens      = 0;
    double      prompt_ms         = 0.0;
    double      gen_ms            = 0.0;
    double      ttft_ms           = 0.0;
    double      wall_sec          = 0.0;
    std::string response;
};

// Trim leading/trailing whitespace
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end == std::string::npos ? std::string::npos : end - start + 1);
}

// Parse "Key: value" line; returns value or empty (handles leading spaces)
static std::string parse_value_line(const std::string& line, const std::string& key) {
    std::string t = trim(line);
    if (t.size() < key.size() + 1) return "";
    if (t.compare(0, key.size(), key) != 0) return "";
    if (t[key.size()] != ':') return "";
    return trim(t.substr(key.size() + 1));
}

static bool parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    try {
        out = std::stod(s);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_int(const std::string& s, int& out) {
    if (s.empty()) return false;
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

// Parse one session log file into a record. Returns false on critical parse failure.
static bool parse_log_file(const fs::path& path, SessionRecord& out) {
    std::ifstream f(path);
    if (!f) return false;

    out = SessionRecord{};
    std::string line;
    enum Section { None, Model, Images, Prompt, Performance, Response };
    Section section = None;
    std::ostringstream prompt_accum;
    std::ostringstream response_accum;
    bool in_prompt = false;
    bool in_response = false;

    while (std::getline(f, line)) {
        std::string trimmed = trim(line);

        if (trimmed == "[MODEL]") {
            section = Model;
            in_prompt = false;
            in_response = false;
            continue;
        }
        if (trimmed == "[IMAGES]") {
            section = Images;
            in_prompt = false;
            in_response = false;
            continue;
        }
        if (trimmed == "[PROMPT]") {
            section = Prompt;
            prompt_accum.str("");
            prompt_accum.clear();
            in_prompt = true;
            in_response = false;
            continue;
        }
        if (trimmed == "[PERFORMANCE]") {
            section = Performance;
            in_prompt = false;
            in_response = false;
            continue;
        }
        if (trimmed == "[RESPONSE]") {
            section = Response;
            response_accum.str("");
            response_accum.clear();
            in_prompt = false;
            in_response = true;
            continue;
        }

        if (section == None && trimmed.find("Timestamp:") == 0) {
            size_t colon = trimmed.find(':');
            if (colon != std::string::npos)
                out.timestamp = trim(trimmed.substr(colon + 1));
            continue;
        }

        if (section == Model) {
            std::string v;
            if (!(v = parse_value_line(line, "Description")).empty())
                out.model_description = v;
            else if (!(v = parse_value_line(line, "Images processed")).empty())
                parse_int(v, out.images_processed);
            continue;
        }

        if (section == Images) {
            // Lines like "  1. /path/to/img.png"
            if (trimmed.empty()) continue;
            size_t dot = trimmed.find('.');
            if (dot != std::string::npos) {
                std::string path_part = trim(trimmed.substr(dot + 1));
                if (!path_part.empty()) {
                    if (!out.image_paths.empty()) out.image_paths += "; ";
                    out.image_paths += path_part;
                }
            }
            continue;
        }

        if (in_prompt) {
            if (trimmed.empty() && prompt_accum.str().size() > 0)
                continue; // keep reading until we hit [PERFORMANCE] or next section
            if (trimmed.find('[') == 0) {
                in_prompt = false;
                section = None;
                continue;
            }
            if (prompt_accum.tellp() > 0) prompt_accum << "\n";
            prompt_accum << line;
            continue;
        }

        if (section == Performance) {
            std::string v;
            if (!(v = parse_value_line(line, "Tokens/sec (generation)")).empty())
                parse_double(v, out.tokens_per_sec);
            else if (!(v = parse_value_line(line, "Prompt tokens")).empty())
                parse_int(v, out.prompt_tokens);
            else if (!(v = parse_value_line(line, "Generated tokens")).empty())
                parse_int(v, out.gen_tokens);
            else if (!(v = parse_value_line(line, "Total tokens")).empty())
                parse_int(v, out.total_tokens);
            else if (!(v = parse_value_line(line, "Prompt eval time")).empty()) {
                // "123.4 ms"
                size_t sp = v.find(' ');
                if (sp != std::string::npos) v = v.substr(0, sp);
                parse_double(v, out.prompt_ms);
            } else if (!(v = parse_value_line(line, "Generation time")).empty()) {
                size_t sp = v.find(' ');
                if (sp != std::string::npos) v = v.substr(0, sp);
                parse_double(v, out.gen_ms);
            } else if (!(v = parse_value_line(line, "Time to first token")).empty()) {
                size_t sp = v.find(' ');
                if (sp != std::string::npos) v = v.substr(0, sp);
                parse_double(v, out.ttft_ms);
            } else if (!(v = parse_value_line(line, "Total wall time")).empty()) {
                size_t sp = v.find(' ');
                if (sp != std::string::npos) v = v.substr(0, sp);
                parse_double(v, out.wall_sec);
            }
            continue;
        }

        if (in_response) {
            if (trimmed.find("====") == 0) break;
            if (response_accum.tellp() > 0) response_accum << "\n";
            response_accum << line;
            continue;
        }
    }

    out.prompt = trim(prompt_accum.str());
    out.response = trim(response_accum.str());
    return true;
}

// Escape a CSV field: wrap in quotes, double internal quotes.
static std::string csv_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else if (c == '\r') continue;
        else out += c;
    }
    out += '"';
    return out;
}

// Write one row of CSV (all fields escaped for safety).
static void write_csv_row(std::ostream& out, const SessionRecord& r) {
    out << csv_escape(r.timestamp)
        << "," << csv_escape(r.model_description)
        << "," << r.images_processed
        << "," << csv_escape(r.image_paths)
        << "," << csv_escape(r.prompt)
        << "," << r.tokens_per_sec
        << "," << r.prompt_tokens
        << "," << r.gen_tokens
        << "," << r.total_tokens
        << "," << r.prompt_ms
        << "," << r.gen_ms
        << "," << r.ttft_ms
        << "," << r.wall_sec
        << "," << csv_escape(r.response)
        << "\n";
}

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  Scrape PiVision session logs from the log directory and write a CSV.\n\n"
        << "Options:\n"
        << "  --log-dir <path>   Log directory (default: from config or ~/pivision_logs)\n"
        << "  --config <path>    Config file to read log_directory from\n"
        << "  --output <file>    Output CSV path (default: <log-dir>/pivision_sessions.csv)\n"
        << "  --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string log_dir;
    std::string config_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--log-dir" || arg == "-l") {
            if (i + 1 >= argc) { std::cerr << "error: --log-dir requires an argument\n"; return 1; }
            log_dir = argv[++i];
            continue;
        }
        if (arg == "--config" || arg == "-C") {
            if (i + 1 >= argc) { std::cerr << "error: --config requires an argument\n"; return 1; }
            config_path = argv[++i];
            continue;
        }
        if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) { std::cerr << "error: --output requires an argument\n"; return 1; }
            output_path = argv[++i];
            continue;
        }
        std::cerr << "error: unknown option " << arg << "\n";
        usage(argv[0]);
        return 1;
    }

    std::string resolved = resolve_log_dir(log_dir, config_path);
    if (resolved.empty()) {
        std::cerr << "error: could not determine log directory. Set --log-dir or ensure HOME/USERPROFILE and pivision log_directory are set.\n";
        return 1;
    }

    fs::path log_path(resolved);
    if (!fs::exists(log_path) || !fs::is_directory(log_path)) {
        std::cerr << "error: log directory does not exist or is not a directory: " << resolved << "\n";
        return 1;
    }

    if (output_path.empty())
        output_path = (log_path / "pivision_sessions.csv").string();

    std::vector<fs::path> log_files;
    for (const auto& entry : fs::directory_iterator(log_path)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() > 8 && name.compare(0, 8, "session_") == 0) {
            size_t ext = name.rfind(".log");
            if (ext != std::string::npos && ext + 4 == name.size())
                log_files.push_back(entry.path());
        }
    }
    std::sort(log_files.begin(), log_files.end());

    std::vector<SessionRecord> records;
    for (const auto& p : log_files) {
        SessionRecord r;
        if (parse_log_file(p, r))
            records.push_back(std::move(r));
        else
            std::cerr << "warning: skipped or failed to parse: " << p << "\n";
    }

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "error: cannot open output file: " << output_path << "\n";
        return 1;
    }

    // Header row
    csv << "timestamp,model_description,images_processed,image_paths,prompt,"
        << "tokens_per_sec,prompt_tokens,gen_tokens,total_tokens,"
        << "prompt_ms,gen_ms,ttft_ms,wall_sec,response\n";

    for (const auto& r : records)
        write_csv_row(csv, r);

    std::cerr << "Wrote " << records.size() << " session(s) to " << output_path << "\n";
    return 0;
}
