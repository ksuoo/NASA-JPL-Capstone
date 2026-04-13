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
