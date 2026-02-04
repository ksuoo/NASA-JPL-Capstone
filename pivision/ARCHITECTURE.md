# PiVision Architecture

Lightweight C++ CLI and library for VLM (Vision-Language Model) inference on Linux ARM64/x86_64, backed by llama.cpp. Optimized for single-board computers.

## Directory Layout

```
/
├── CMakeLists.txt              # Build system; links against external llama.cpp
├── README.md                   # Quick-start usage guide
├── ARCHITECTURE.md             # This file
├── install.sh                  # One-line installer
├── include/
│   └── pivision.h              # Public API (PIMPL, no llama.cpp leakage)
├── src/
│   └── core.cpp                # Library implementation (talks to llama.cpp)
├── cmd/
│   └── main.cpp                # CLI entry point (only sees pivision.h)
└── third_party/
    └── stb/
        └── stb_image.h         # Header-only image loader (used inside mtmd)
```

## Key Design Decisions

### Clean API boundary (PIMPL)
`pivision.h` exposes an opaque `PiVision` class whose `Impl` struct lives
entirely in `core.cpp`. Consumers (including the CLI) never `#include` any
llama.cpp header.

### Linking model
llama.cpp is **not** vendored. The build expects a pre-compiled llama.cpp
tree via `-DLLAMA_DIR`. The CMake file imports shared and static libraries:
- `libllama.so`    — main LLM inference engine
- `libggml.so`     — tensor/backend library
- `libggml-base.so`— ggml base operations
- `libmtmd.so`     — multimodal support (CLIP encoder + projector + helpers)
- `libcommon.a`    — llama.cpp common utilities (chat templates, formatting)

### Multimodal via `mtmd`
The library uses llama.cpp's high-level `mtmd` API (`mtmd.h` +
`mtmd-helper.h`) instead of raw `clip.h` calls. This handles:
- Image decoding (delegates to stb_image internally)
- CLIP encoding and preprocessing
- Prompt tokenization with `<__media__>` markers
- Embedding injection into the LLM context
- Correct handling of model-specific features (Gemma 3, LLaVA, etc.)

### Prompt Formatting

Two prompt formatting paths exist:

**Single-shot** (`run()` / `run_collect()`): Uses `format_chat_prompt()`
(a static helper in `core.cpp`) which calls `llama_chat_apply_template()`
to format a single user message. The template is auto-detected from GGUF
metadata.

**Multi-turn chat** (`chat_turn()` / `chat_turn_collect()`): Uses
`common_chat_format_single()` from llama.cpp's libcommon. This takes the
full chat history and formats only the new turn incrementally, which is
needed because the KV cache retains previous turns.

Both paths auto-detect the model's chat template from GGUF metadata.
Image markers (`<__media__>`) are prepended to user content before
template application. The `<__media__>` markers are consumed by
`mtmd_tokenize()` which replaces them with image embedding chunks.

### CPU-only / NEON
All GPU offloading is disabled (`n_gpu_layers = 0`, `use_gpu = false`).
Built with architecture-specific optimizations (`-march=armv8-a -O3` on ARM64, `-O3` on x86_64) and LTO.

## Inference Data Flow

```
CLI (cmd/main.cpp)
  │  --model, --vision, --image (×N), --prompt [--json]
  ▼
pv.validate(images)
  ├─ mtmd_support_vision() — projector compatible?
  └─ for each path: exists? magic bytes = JPG/PNG?
  │
pv.load_image(path)  ×N
  └─ mtmd_helper_bitmap_init_from_file() → stored in bitmaps vector
  │
pv.run(prompt, cb)                    ← streaming mode
pv.run_collect(prompt) → RunResult    ← JSON / batch mode
  │
  ├─ 1. llama_memory_clear() + llama_perf_context_reset()
  │
  ├─ 2. format_chat_prompt(cached_template, prompt, n_images, marker)
  │     → llama_chat_apply_template() formats for any model
  │
  ├─ 3. mtmd_tokenize()  → interleaved text + image chunks
  │
  ├─ 4. mtmd_helper_eval_chunks()
  │     → CLIP-encodes each image once, injects embeddings into
  │       KV cache at correct offsets, then runs llama_decode
  │
  └─ 5. Sampling loop
        ┌──────────────────────────────┐
        │ llama_sampler_sample()       │
        │ → llama_vocab_is_eog check   │
        │ → llama_token_to_piece()     │
        │ → stream_cb / accumulate     │
        │ → llama_decode(next token)   │
        └──────────────────────────────┘
        Repeats until EOG or n_ctx exhausted.
  │
  ├─ 6. llama_perf_context() → fill RunResult metadata
  │
  ▼
stdout  (streaming tokens  OR  single JSON object)
```

## Interactive Chat Mode

`--chat` enters an interactive REPL with persistent KV cache across turns.
Can be combined with `--prompt` to process an initial message before entering
interactive mode.

```
$ pivision_cli --chat

pivision chat (type /quit to exit, /help for commands)

> Hello, what can you do?
I can answer questions, describe images, ...

> /image photo.jpg
loaded: photo.jpg

> What's in this image?
This image shows a sunset over...

> Tell me more about the colors
The sky transitions from deep orange...

> /clear
conversation cleared

> /quit
```

**With initial prompt** (processes first, then enters interactive mode):
```
$ pivision_cli --chat --image photo.jpg --prompt "What's in this image?"

pivision chat (type /quit to exit, /help for commands)

> What's in this image?
This image shows a mountain landscape with...

> Tell me more about the sky
The sky appears to be...

> /quit
```

**Chat state** (in `core.cpp` Impl):
- `common_chat_templates_ptr tmpls` — parsed chat template from model
- `std::vector<common_chat_msg> chat_history` — grows each turn
- `llama_pos n_past` — KV cache cursor (not reset between turns)

**Per-turn flow**:
1. Format only the new turn via `common_chat_format_single()`
2. Tokenize + eval (advancing `n_past`)
3. Sample response tokens (streaming to stdout)
4. Push both user and assistant messages onto `chat_history`

`/clear` calls `chat_clear()` which resets the KV cache, `n_past`, and
history vector.

## Verbose Stats (`--verbose`)

When `--verbose` is passed, a clean stats block is printed to stderr after
each response (works in both single-shot and chat modes):

```
--- stats -----------------------------------------------
  model:          gemma-3-4B Q4_0
  images:         1
  prompt tokens:  842  (3254.1 ms, 258.7 tok/s)
  gen tokens:     128  (9876.5 ms, 13.0 tok/s)
  ttft:           24 ms
  wall time:      13.1 s
---------------------------------------------------------
```

llama.cpp's internal debug logging is always suppressed — `--verbose` only
shows the useful summary above.

## JSON Output Mode

When `--json` is passed the CLI collects all tokens via `run_collect()` and
prints a single JSON object:

```json
{
  "content": "The model response text...",
  "metadata": {
    "model": "gemma-3-4b",
    "images_processed": 2,
    "prompt_tokens": 842,
    "gen_tokens": 128,
    "total_tokens": 970,
    "tokens_per_sec": 13.0,
    "ttft_ms": 24,
    "wall_time_sec": 13.1
  }
}
```

Errors also emit JSON when `--json` is active:
```json
{"error": "image file not found: missing.jpg"}
```

## Pre-Inference Validation

`pv.validate(image_paths)` runs before any model inference:

| Check | How |
|---|---|
| Vision projector compatible with LLM | `mtmd_support_vision()` |
| Image file exists | `std::ifstream` open check |
| Image is JPG or PNG | Magic-byte comparison (first 3–4 bytes) |

Returns an empty string on success, or a human-readable error message.

## Multi-Image Support

Multiple images are loaded via repeated `load_image()` calls. The prompt
engine inserts one `<__media__>` marker per image.
`mtmd_helper_eval_chunks` processes all chunks in order, encoding each
image's CLIP embeddings exactly once and injecting them at the correct
KV-cache offset.

## Sampler Chain

```
top_k(40) → top_p(0.95) → temperature(config.temperature) → dist(seed=42)
```

## Configuration System

PiVision searches for config files in priority order:

1. `--config <path>` (explicit CLI argument)
2. `./pivision.json` (current directory)
3. `~/.config/pivision/config.json` (user)
4. `/etc/pivision/config.json` (system)

```json
{
  "model_path": "/path/to/model.gguf",
  "vision_path": "/path/to/mmproj.gguf",
  "default_image_path": "",
  "default_n_ctx": 4096,
  "log_directory": "~/pivision_logs"
}
```

CLI arguments always override config values. If a path doesn't exist, falls
back to built-in defaults (Gemma 3 4B).

## Health Check

`--check-health` verifies system readiness:
- CPU thermal status (warns > 70°C)
- Available RAM (warns < 2GB)
- llama.cpp library availability
- Model file presence and size

## Installation

### Production Install

```bash
curl -sSL https://raw.githubusercontent.com/jplpi/pivision/main/install.sh | bash
```

Installs to `/usr/local/bin/pivision` with auto-configured library paths.

### Development Build

```bash
mkdir build && cd build
cmake .. -DLLAMA_DIR=/home/jplpi/llama.cpp
make -j4

export LD_LIBRARY_PATH=/home/jplpi/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

## Usage Examples

```bash
# Simple usage (uses config/defaults)
pivision --image photo.jpg --prompt "Describe this image."

# Interactive chat
pivision --chat

# Chat with initial prompt
pivision --chat --image photo.jpg --prompt "What's in this image?"

# JSON mode (script-friendly)
pivision --image a.jpg --image b.jpg --prompt "Compare" --json

# Health check
pivision --check-health
```

## Status

### Core Features
- [x] Project skeleton and build system
- [x] Real llama.cpp model loading (LLM + vision projector)
- [x] Image loading via mtmd (stb_image internally)
- [x] Multi-image support with `<__media__>` markers
- [x] CLIP encoding + embedding injection via mtmd_helper_eval_chunks
- [x] Streaming token generation with sampler chain
- [x] Auto-detecting chat template (works with any model)
- [x] JSON output mode (`--json`)
- [x] Pre-inference validation (file existence, format, projector compatibility)
- [x] Performance metadata (tokens/sec, TTFT, wall time, prompt/gen split)
- [x] Text-only mode (no --vision/--image required)
- [x] Auto-detect vision projector from model directory
- [x] Interactive multi-turn chat mode (`--chat`) with persistent KV cache
- [x] Chat mode with initial prompt (`--chat --prompt "..."`)

### Production Features
- [x] Configuration file with priority chain (CLI > local > user > system)
- [x] Comprehensive session logging with performance metrics
- [x] Health check (`--check-health`) for thermal, RAM, libraries
- [x] One-line installer (`install.sh`)
- [x] Release build with IPO/LTO and binary stripping
- [x] System-wide installation to `/usr/local/bin/pivision`

### Planned
- [ ] Configurable sampler parameters from CLI
- [ ] Model presets/aliases
