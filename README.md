> "I can't believe it's not Unsloth!"
> <br>— Grandma

# GGUF Clone

Create optimized GGUF quantizations by _cloning_ from any GGUF of the same architecture.

Whether you just fine-tuned your own model or stumbled upon a new hidden gem on Hugging Face, this tool allows you to quickly quantize it with the same settings as an existing, high-quality quant.

In theory, fine-tunes should benefit from the same imatrix data and optimizations. In practice... it's hard to tell for sure. But at least it's now dead simple to try.

## Key Features

- Simple YAML config
- Use any Hugging Face GGUF as **template** (model to copy from)
- Use any Hugging Face model as **target** (model to quantize)
- Optionally copy GGUF metatdata and loose files (mmproj, etc)
- Run multiple quantizations from the same template
- Use split GGUFs as template input and/or target output
- Works with your existing Hugging Face cache and llama.cpp install

## Prerequisites

- Python 3.9+ (developed on 3.12, not thoroughly tested on others)
- The following [llama.cpp](https://github.com/ggerganov/llama.cpp) tools in `PATH` or built under `GGUF_CLONE_LLAMA_CPP` (see below):
  - `llama-quantize`
  - `llama-gguf-split`

## Installation + Usage

```bash
# Recommended
uvx gguf-clone # assumes config.yml in current dir
uvx gguf-clone path/to/config.yml

# Or
pip install gguf-clone
gguf-clone
gguf-clone path/to/config.yml

# Verbose output
gguf-clone --verbose

# Non-interactive modes to skip prompts when outputs already exist
gguf-clone --use-existing
gguf-clone --overwrite
gguf-clone --cancel
```

Outputs are generated alongside the config file by default:

- `converted/*.gguf` - Target model converted into a GGUF for llama.cpp
- `params/*.json` - `llama-quantize` paramaters extracted from template GGUF(s)
- `quantized/*.gguf` - Final quantized outputs

## Configuration

This tool uses Hugging Face under-the-hood for convience. Use the `ORG/MODEL` labels to designate the template and target models. All matched original files will be downloaded to your local Hugging Face cache.

Minimal example (only required fields):

```yaml
template:
  repo: unsloth/Qwen3-0.6B-GGUF
  imatrix: "*imatrix*"
  ggufs: "*UD-IQ1_M*.gguf"
  
target:
  repo: Qwen/Qwen3-0.6B
```

All options:

```yaml
template:
  repo: unsloth/Qwen3-0.6B-GGUF
  imatrix: "*imatrix*"
  
  # List multiple patterns to create multiple quantizations
  ggufs:
    - "*UD-IQ1_M*.gguf"
    - "*UD-Q2_K_XL*.gguf"
  
  # Copy GGUF metadata (from the 1st shard if split)
  copy_metadata:
    - tokenizer.chat_template
    
  # Copy matched files
  copy_files:
    - "*mmproj*"

target:
  repo: unsloth/Qwen3-0.6B

output:
  # Output model is named PREFIX-ORG-MODEL
  prefix: not-unsloth
  
  # Add or edit GGUF metadata
  apply_metadata:
    general.quantized_by: "https://github.com/spicyneuron/gguf-clone"

  # Split output if greater than this size (unit can be M or G)
  split: 50G

  # Output directories (relative to config.yml location)
  converted_dir: converted
  params_dir: params
  quantized_dir: quantized
```

## Environment Variables

If you don't have llama.cpp tools on your `PATH`, point gguf-clone at a local llama.cpp repo:

```bash
GGUF_CLONE_LLAMA_CPP="path/to/llama.cpp/repo"
```

If `GGUF_CLONE_LLAMA_CPP` is set, gguf-clone will prefer that its scripts and tools over the `PATH` and vendored copies.

Hugging Face [environment variables](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables) can be used to change your download cache location.

## Technical Notes

- **Cloning accuracy**: The tool uses heuristics to determine quantization parameters from template GGUFs (most common tensor type as default, ignore lists for non-quantized tensors). For identical tensor names and shapes, results should be functionally equivalent.
- **Vendored dependencies**: `gguf-py/gguf` is vendored from [llama.cpp](https://github.com/ggml-org/llama.cpp) because the published version on PyPI is out of date with recent `llama.cpp` versions. `convert_hf_to_gguf.py` is also vendered so conversion behavior stays aligned.

## Gratitude

All credit goes to the [Unsloth](https://unsloth.ai/) and [llama.cpp](https://github.com/ggml-org/llama.cpp) teams for their hard work and responsiveness. This is merely a wrapper around their genius.

This project was originally inspired by [quant_clone](https://github.com/electroglyph/quant_clone). It didn't actually work, unfortunately, but pointed me in the right direction.
