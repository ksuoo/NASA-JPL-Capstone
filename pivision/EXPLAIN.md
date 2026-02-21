# How PiVision Works

A technical explanation for developers.

---

## The Big Picture

PiVision lets you ask questions about images using AI, running entirely on a Raspberry Pi with no internet required.

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐
│  Image  │ +  │ Prompt  │ -> │  PiVision   │ -> │ Response │
│ (JPEG)  │    │ (Text)  │    │  (on Pi 5)  │    │  (Text)  │
└─────────┘    └─────────┘    └─────────────┘    └──────────┘
```

**Example:**
- Input image: `mars_rock.jpg`
- Input prompt: `"What type of rock is this?"`
- Output: `"This appears to be a sedimentary rock with layered striations..."`

---

## The Two Models

PiVision uses **two AI models** that work together:

### 1. Vision Encoder (CLIP)
- **What it does:** Converts an image into numbers (embeddings)
- **File:** `mmproj-*.gguf` (~600MB - 1GB)
- **Architecture:** Based on CLIP (Contrastive Language-Image Pre-training)

```
┌─────────────┐      ┌──────────────────┐
│   Image     │  ->  │  Vision Encoder  │  ->  [0.23, -0.45, 0.12, ...]
│ 1024x768px  │      │     (CLIP)       │      (768-4096 numbers)
└─────────────┘      └──────────────────┘
```

The vision encoder breaks the image into patches (like a grid), processes each patch, and outputs a sequence of embedding vectors that represent what's in the image.

### 2. Language Model (LLM)
- **What it does:** Generates text based on the prompt + image embeddings
- **File:** `gemma-3-4b-*.gguf` (~3GB)
- **Architecture:** Transformer-based autoregressive language model

```
┌─────────────────────┐      ┌─────────────────┐
│  Image Embeddings   │      │                 │
│         +           │  ->  │   Language      │  ->  "This rock shows..."
│   Text Prompt       │      │   Model (LLM)   │
└─────────────────────┘      └─────────────────┘
```

---

## The Complete Data Flow

Here's what happens when you run:
```bash
pivision --image rock.jpg --prompt "Describe this rock"
```

### Step 1: Image Loading
```
rock.jpg  ->  stb_image library  ->  RGB pixel array (width x height x 3)
```

### Step 2: Image Preprocessing
```
RGB pixels  ->  Resize to model's expected size (e.g., 448x448)
            ->  Normalize pixel values to [-1, 1]
            ->  Convert to tensor format
```

### Step 3: Vision Encoding (CLIP)
```
Image tensor  ->  Split into patches (e.g., 14x14 pixel patches)
              ->  Each patch -> embedding vector
              ->  Output: sequence of ~256-1024 embedding vectors
```

### Step 4: Prompt Formatting
The prompt is formatted using the model's chat template:
```
<bos><start_of_turn>user
<__media__>
Describe this rock<end_of_turn>
<start_of_turn>model
```

The `<__media__>` marker tells the model where to inject the image embeddings.

### Step 5: Tokenization
```
Formatted prompt  ->  Tokenizer  ->  [2, 106, 1645, 108, 255999, 108, ...]
                                     (list of integer token IDs)
```

Each word/subword becomes a number. The `<__media__>` marker is a special token.

### Step 6: Embedding Injection
```
Token embeddings:     [emb_0] [emb_1] [emb_2] [MEDIA] [emb_4] ...
                                        ↓
Image embeddings:                 [img_0, img_1, img_2, ... img_255]
                                        ↓
Combined:            [emb_0] [emb_1] [emb_2] [img_0...img_255] [emb_4] ...
```

The image embeddings are inserted where `<__media__>` was.

### Step 7: Transformer Forward Pass
The combined embeddings go through the transformer layers:
```
Input embeddings
       ↓
┌─────────────────────────────────┐
│  Attention Layer 1              │  <- Each token attends to all others
│  Feed-Forward Layer 1           │
├─────────────────────────────────┤
│  Attention Layer 2              │
│  Feed-Forward Layer 2           │
├─────────────────────────────────┤
│         ... (32 layers)         │
├─────────────────────────────────┤
│  Attention Layer 32             │
│  Feed-Forward Layer 32          │
└─────────────────────────────────┘
       ↓
Output: probability distribution over vocabulary
```

### Step 8: Token Sampling (Autoregressive Generation)
```
While not end-of-sequence:
    1. Get probability for each possible next token
    2. Apply sampling (top-k, top-p, temperature)
    3. Pick a token
    4. Add to output
    5. Feed back into model for next token
```

Example generation:
```
Step 1: [prompt + image] -> "This"
Step 2: [prompt + image + "This"] -> "appears"
Step 3: [prompt + image + "This appears"] -> "to"
Step 4: [prompt + image + "This appears to"] -> "be"
...continues until <eos> token
```

### Step 9: Detokenization
```
[1596, 8412, 577, 614, ...]  ->  "This appears to be a sedimentary rock..."
```

---

## Key Concepts

### Embeddings
Numbers that represent meaning. Similar concepts have similar numbers.
```
"cat"  -> [0.2, 0.8, -0.3, ...]
"dog"  -> [0.3, 0.7, -0.2, ...]  <- Similar to cat
"car"  -> [-0.5, 0.1, 0.9, ...]  <- Different
```

### Attention
How the model relates different parts of the input. When processing the word "rock", the model can "look at" the image embeddings to understand what rock you mean.

### KV Cache
Stores intermediate computations so we don't recalculate everything for each new token. This is why chat mode is faster after the first message.

### Quantization (GGUF)
Models are compressed from 16-bit floats to 4-bit integers to fit on the Pi.
```
Original:  13 billion parameters × 2 bytes = 26 GB
Quantized: 13 billion parameters × 0.5 bytes = 6.5 GB
```

Some accuracy is lost, but it's usually acceptable.

---

## Code Architecture

```
pivision/
├── cmd/main.cpp        # CLI interface (argument parsing, user interaction)
├── src/core.cpp        # Engine (all llama.cpp interaction)
├── include/pivision.h  # Public API (what cmd/main.cpp sees)
└── third_party/stb/    # Image loading library
```

### The PIMPL Pattern

`main.cpp` never sees llama.cpp internals. It only uses the clean `PiVision` class:

```cpp
// main.cpp - Simple and clean
PiVision pv(config);
pv.load_image("rock.jpg");
RunResult result = pv.run_collect("Describe this rock");
std::cout << result.content;
```

```cpp
// core.cpp - All the llama.cpp complexity hidden here
struct PiVision::Impl {
    llama_model * model;
    llama_context * ctx;
    mtmd_context * mtmd_ctx;  // Vision encoder
    // ... all the complex stuff
};
```

This separation means:
- CLI code is easy to understand
- llama.cpp details don't leak out
- We can change the engine without touching the CLI

---

## Why llama.cpp?

llama.cpp is a C++ library for running LLMs efficiently on CPUs.

**Key features we use:**
- `llama_model_load_from_file()` - Load GGUF model
- `llama_decode()` - Run transformer forward pass
- `llama_sampler_sample()` - Pick next token
- `mtmd_*` functions - Handle vision encoding

**Why it's fast on Pi:**
- Written in C++ with SIMD optimizations (NEON on ARM)
- Quantized models (4-bit instead of 16-bit)
- Efficient memory management
- No Python overhead

---

## Performance Characteristics

On Raspberry Pi 5 (8GB) with Gemma 3 4B Q4:

| Metric | Typical Value |
|--------|---------------|
| Model load time | ~30 seconds |
| Image encoding | ~2-5 seconds |
| Prompt processing | ~5-15 seconds |
| Token generation | ~10-15 tokens/sec |
| Memory usage | ~4-5 GB |

**Bottlenecks:**
1. **Prompt processing** - Must process all tokens before generating
2. **Memory bandwidth** - Moving weights from RAM to CPU
3. **Image encoding** - CLIP is compute-intensive

---

## File Formats

### GGUF (GPT-Generated Unified Format)
Binary format for storing quantized models.
```
┌────────────────────────────────┐
│  Header (magic, version)       │
├────────────────────────────────┤
│  Metadata (architecture,       │
│  tokenizer, chat template)     │
├────────────────────────────────┤
│  Tensor data (weights,         │
│  quantized to 4-bit)           │
└────────────────────────────────┘
```

### Model Files We Use
```
gemma-3-4b-it-q4_0.gguf     # Main language model (3GB)
mmproj-model-f16-4B.gguf    # Vision encoder/projector (1GB)
```

---

## Sampling Parameters

Control how the model picks tokens:

```cpp
// Current settings in core.cpp
top_k(40)         // Only consider top 40 most likely tokens
top_p(0.95)       // Only consider tokens until cumulative prob > 95%
temperature(0.7)  // Higher = more random, Lower = more deterministic
```

**Temperature example:**
```
Prompt: "The sky is"

Temperature 0.1 (deterministic):
  -> "blue" (always picks highest probability)

Temperature 1.0 (random):
  -> "blue" / "clear" / "beautiful" / "gray" (varies each time)
```

---

## Multi-Turn Chat

In chat mode, we keep the KV cache between turns:

```
Turn 1: "What's in this image?"
        [process image + prompt, generate response]
        KV cache now contains: image + Q1 + A1

Turn 2: "Tell me more about the rocks"
        [only process new prompt, reuse cached context]
        KV cache now contains: image + Q1 + A1 + Q2 + A2
```

This is why follow-up questions are faster - we don't re-encode the image.

---

## Summary

1. **Image** goes through **CLIP** to become **embeddings**
2. **Prompt** gets **tokenized** and formatted with chat template
3. **Image embeddings** are injected where `<__media__>` marker is
4. **Transformer** processes everything and predicts next token
5. **Sampling** picks tokens one by one until done
6. **Tokens** are converted back to **text**

All of this runs on a $80 Raspberry Pi with no internet connection.
