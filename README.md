# NASA JPL Capstone
## Vision Language Models for Deep Space Spacecraft

Lightweight C++ command line interface (CLI) tool and library enabling VLM (Vision Language Model) inference on Linux ARM64/x86_64, backed by [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Installation

Run the installation script located at: pivision/install.sh.
More info on running that within the pivision folder.

## Custom Flags
`--model <path>`
- Uses a GGUF LLM model from the selected path.

`--vision <path>`
- Uses a GGUF VLM model from the selected path.

`--image <path>`
- Attaches an image from the selected path. Multiple images can be loaded in a single call. 

`--prompt <str>`
- Prompts the model using the attached string.

`--chat`
- Enables interactive chat mode with the model. Can be combined with `--prompt` to process an initial image before interaction.

`--verbose`
- Outputs model benchmark statistics to stderr after each response.
```
<Example output>
```

`--json`
- Enables JSON output mode. CLI collects all tokens and outputs statistics in a JSON object. 
```
<Example response>
```

`--check-health` 
- Determines system readiness based on hardware availability/usage.

## Usage Examples
```
# Uses default model & vision from config
pivision --image photo.jpg --prompt "Describe this image."

# Interactive chat
pivision --chat

# Chat with initial prompt
pivision --chat --image photo.jpg --prompt "What's in this image?"

# Health check
pivision --check-health
```
