# Testing

This folder contains the prompts, images, scripts, generated configs used for benchmarking Pivision against various use cases.

## Prompt and Image Layout

Prompts and images are matched by basename. For example:
- `PCat3_A.txt`
- `PCat3_A.jpg`

Supported image extensions:
- '.jpg' 
- '.jpeg'
- '.png'

A config file only gets generated if both the prompt and its matching image exists.


## Model Layout 
Models are expected to live under:

```text
llama.cpp/models/
`-- <model_name>/
    |-- model/
    |   `-- <model_file>.gguf
    `-- mmproj/
        `-- <mmproj_file>.gguf
```
Script currently uses first match if multiple files are found.

## Script 1: `config_script.sh`

Generates config files for each prompt/image test case

Script:
- reads and searches for models listed in `MODELS` array
- loops through every prompt in `../prompts`
- looks for corresponding image in `../images`
- generates JSON config file

### Example Generated Config

```json
{
  "comment": "gemma-3-12b-it-Q4_K_M PCat3_A",
  "model_path": "../../llama.cpp/models/gemma-3-12b-it-Q4_K_M/model/gemma-3-12b-it-Q4_K_M.gguf",
  "vision_path": "../../llama.cpp/models/gemma-3-12b-it-Q4_K_M/mmproj/mmproj-BF16.gguf",
  "default_image_path": "../../testing/images/PCat3/PCat3_A.jpg",
  "default_n_ctx": 4096,
  "prompt": "../../testing/prompts/PCat3/PCat3_A.txt",
  "log_directory": "../../pivision_logs/gemma-3-12b-it-Q4_K_M/PCat3_A"
}
```

### Dependency
config_script uses `jq` to generate JSON objects

Install on Debian:
```bash
sudo apt install jq
```

## Script 2: `test_all.sh`

Runs all config files for each model listed in `MODELS` array

Script:
- runs each JSON config with pivision_cli using `--config` flag
- writes failure details to `../errors`
- running the cli tool creates a log file that captures the output


