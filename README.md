# NASA JPL Capstone
## Vision Language Models for Deep Space Spacecraft

Lightweight C++ command line interface (CLI) tool and library enabling VLM (Vision Language Model) inference on Linux ARM64/x86_64, backed by [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Installation & Set Up

1. Run the installation script located at: pivision/install.sh. 
2. Upload your models and projectors to the directory llama.cpp/models
3. Run config_script.sh in testing/scripts to generate the configuration files to benchmark using the preset use cases.

## Custom Flags
`--model <path>`  Uses a GGUF LLM model from the selected path.

`--vision <path>`  Uses a GGUF VLM model from the selected path.

`--image <path>`  Attaches an image from the selected path. Multiple images can be loaded in a single call. 

`--prompt <str>`  Prompts the model using the attached string.

`--chat`  Enables interactive chat mode with the model. Can be combined with `--prompt` to process an initial image before interaction.

`--verbose`  Outputs model benchmark statistics to stderr after each response.
```
--- stats -----------------------------------------------
  model:          mistral3 3B Q4_K - Medium
  images:         0
  prompt tokens:  4  (1139.7 ms, 3.5 tok/s)
  gen tokens:     12  (9199.5 ms, 1.3 tok/s)
  ttft:           2 ms
  wall time:      10.4 s
---------------------------------------------------------

```

`--json`  Enables JSON output mode. CLI collects all tokens and outputs statistics in a JSON object. 
```
<Example response>
```

`--check-health`   Determines system readiness based on hardware availability/usage.

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

## Testing

Bash scripts are provided to help run prompt/image test cases across models

Detailed testing documentation can be found in [testing/README.md](testing/README.md).

### Quick Start
Download the desired models and their mmproj files, and store them in:
- 'llama.cpp/models/{model_name}/model'
- 'llama.cpp/models/{model_name}/mmproj'

Then run:
```bash
cd testing/scripts
./config_script.sh #generates JSON config files for each selected model
./test_all.sh #runs all generated config files for each selected model
