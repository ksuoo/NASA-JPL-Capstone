# PiVision - Quick Start

PiVision runs vision and language models on Linux using llama.cpp. Optimized for ARM64 single-board computers (Raspberry Pi 5, Orange Pi, Rock Pi, Jetson, etc.) but also works on x86_64.

## Installation

### One-Line Install (Production)

```bash
curl -sSL https://raw.githubusercontent.com/jplpi/pivision/main/install.sh | bash
```

This will:
- Detect your system architecture (ARM64 or x86_64)
- Install dependencies (build-essential, cmake, libcurl)
- Clone and build llama.cpp if needed
- Build PiVision with architecture-optimized flags
- Install `pivision` to `/usr/local/bin`
- Set up config directories

### Development Build

If you want to modify the code:

```bash
git clone https://github.com/jplpi/pivision.git
cd pivision
./install.sh --dev
```

Or manually:

```bash
cd ~/build
cmake .. -DLLAMA_DIR=~/llama.cpp
make -j4

# Set library path (add to ~/.bashrc for persistence)
export LD_LIBRARY_PATH=~/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

## Verify Installation

```bash
pivision --check-health
```

Shows thermal status, available RAM, library paths, and model files.

## Usage

### Simple Commands (using defaults)

```bash
# Text question
pivision --prompt "What is the capital of France?"

# Describe an image
pivision --image photo.jpg --prompt "What do you see?"

# Interactive chat
pivision --chat

# Chat with initial prompt
pivision --chat --image photo.jpg --prompt "Describe this"
```

### Explicit Model Paths

```bash
pivision \
  --model ~/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  --vision ~/llama.cpp/models/mmproj-model-f16-4B.gguf \
  --image ~/images/mars_surface.jpeg \
  --prompt "What do you see?"
```

### Interactive Chat

```bash
pivision --chat
```

Then type messages at the `>` prompt:

```
> What's 2 + 2?
4

> /image ~/images/mars_surface.jpeg
loaded: ~/images/mars_surface.jpeg

> What's in this image?
This appears to be the surface of Mars...

> Tell me more about the rocks
The rocks show layered sedimentary patterns...

> /clear
conversation cleared

> /quit
```

### Chat Commands

| Command | What it does |
|---------|-------------|
| `/image <path>` | Load an image for your next message |
| `/clear` | Reset the conversation |
| `/quit` | Exit |
| `/help` | Show commands |

## Configuration

PiVision searches for config files in this order:

1. `--config <path>` (explicit)
2. `./pivision.json` (current directory)
3. `~/.config/pivision/config.json` (user)
4. `/etc/pivision/config.json` (system)

### Config File Format

```json
{
  "model_path": "/path/to/model.gguf",
  "vision_path": "/path/to/mmproj.gguf",
  "default_image_path": "",
  "default_n_ctx": 4096,
  "log_directory": "~/pivision_logs"
}
```

## JSON Output

For scripts, add `--json` to get structured output:

```bash
pivision --image photo.jpg --prompt "Describe this" --json
```

Returns:
```json
{
  "content": "This image shows...",
  "metadata": {
    "model": "gemma-3-4b",
    "images_processed": 1,
    "tokens_per_sec": 12.5,
    "total_tokens": 42
  }
}
```

## Options Reference

| Flag | Description |
|------|-------------|
| `--model <path>` | LLM model file |
| `--vision <path>` | Vision projector file |
| `--image <path>` | Image file (repeatable) |
| `--prompt <text>` | Question or path to prompt file |
| `--config <path>` | Config file path |
| `--chat` | Interactive multi-turn chat mode |
| `--json` | JSON output (single-shot only) |
| `--verbose` | Show performance stats |
| `--check-health` | Verify system status |

## Logs

Every run saves a session log to `~/pivision_logs/` (or config `log_directory`).

Each log contains:
- Timestamp and model info
- Full prompt and images used
- Performance metrics (tokens/sec, TTFT, wall time)
- Complete response

## Project Structure

```
install.sh           One-line installer
include/pivision.h   Public API
src/core.cpp         Engine (llama.cpp interaction)
cmd/main.cpp         CLI tool
third_party/stb/     Image loading (header-only)
```

## Troubleshooting

### Libraries not found
```bash
export LD_LIBRARY_PATH=~/llama.cpp/build/bin:$LD_LIBRARY_PATH
```
Add to `~/.bashrc` for persistence.

### Model not found
Check your config file or specify `--model` explicitly:
```bash
pivision --check-health  # Shows which config is loaded
```

### High temperature warning
SBCs may throttle at high temps. Ensure adequate cooling:
```bash
pivision --check-health  # Shows CPU temperature
```
