> "I can't believe it's not Unsloth!"
> <br>— Grandma

# GGUF Clone

Create optimized GGUF quantizations by _cloning_ from any GGUF of the same architecture.

Whether you just fine-tuned your own model or stumbled upon a new hidden gem on Hugging Face, this tool allows you to quickly quantize it with the same settings as an existing, high-quality quant.

In theory, fine-tunes should benefit from the same imatrix data and optimizations. In practice... it's hard to tell for sure. But at least it's now dead simple to try.

## Key Features

- Simple YAML config
- Use a Hugging Face repo or local path as **template** (model to copy from)
- Use a Hugging Face repo or local path as **target** (model to quantize)
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

Outputs are generated under `output_dir` (default: alongside config file in `./output`):

- `converted/*.gguf` - Target model converted into a GGUF for llama.cpp
- `params/*.json` - `llama-quantize` parameters extracted from template GGUF(s)
- `template_files/*` - Optional staged files copied from the template source
- `quantized/gguf/*.gguf` - Final quantized outputs (or split shards under per-quant folders)

Output file names are keyed by source:

- Hugging Face sources use the repo id (for example `unsloth-Qwen3-0.6B`)
- Local `path` sources only the path leaf name (for example `/a/exp1/model` and `/b/exp2/model` both become `model`)

## Configuration (v2)

The config format is versioned and only `version: 2` is supported.

- `source.template`: template model source, either Hugging Face repo id (`org/model`) or local directory path
- `source.target`: target model source, either Hugging Face repo id or local directory path
- `output_dir`: output root directory (relative to config path by default)

Stage sections are optional. `run` executes declared stages in order.

- `extract_params`: build quantization parameter files from template GGUFs
- `quantize_gguf`: quantize target model using extracted params
- `quantize_mlx`: reserved for MLX stage (not implemented yet)

Minimal config:

```yaml
version: 2

source:
  template: unsloth/Qwen3-0.6B-GGUF
  target: Qwen/Qwen3-0.6B

extract_params:
  ggufs: "*UD-IQ1_M*.gguf"
```

Full v2 example:

```yaml
version: 2

source:
  template: unsloth/Qwen3-0.6B-GGUF
  target: Qwen/Qwen3-0.6B

output_dir: output

extract_params:
  ggufs:
    - "*UD-IQ1_M*.gguf"
    - "*UD-Q2_K_XL*.gguf"
  targets:
    - gguf
    - mlx
  mlx_arch: auto

quantize_gguf:
  # Use an existing converted GGUF instead of converting target source
  target_gguf: null
  # Convert target source with convert_hf_to_gguf.py when target_gguf is null
  target_convert: true
  # Set to null to disable imatrix usage
  imatrix: "*imatrix*"
  # Split output when exceeding this size
  output_max_size: 50G
  # Extract these string keys from template GGUF into params payload
  copy_metadata:
    - tokenizer.chat_template
  # Stage and copy these files into quantized output directory
  copy_files:
    - "*mmproj*"
  # Additional metadata to apply to each quantized output
  apply_metadata:
    general.quantized_by: "https://github.com/spicyneuron/gguf-clone"
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

This project was originally inspired by [quant_clone](https://github.com/electroglyph/quant_clone). It didn't fully work as expected, but pointed me in the right direction.
