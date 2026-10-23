# Local Bolt with Pi

Runs NVIDIA Nemotron 3.5 Bolt E4B locally on the GPU with llama.cpp and opens
the Pi coding harness against it. Pi's config and sessions stay in this
directory, and the model server stops when Pi exits.

## Requirements

- Linux x64 with an NVIDIA GPU (8 GB VRAM), driver, and CUDA libraries in
  `/opt/cuda/lib64`
- [Bun](https://bun.sh)
- [Pi](https://pi.dev) on `PATH`

## Installation

Download the model ([Unsloth GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF))
and the llama.cpp CUDA runtime ([b11840](https://github.com/ggml-org/llama.cpp/releases/tag/b11840)):

```bash
mkdir -p models runtime
curl -fL --retry 3 -C - \
  -o models/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf
curl -fL --retry 3 \
  -o runtime/llama-b11840-cuda13.4.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11840/llama-b11840-bin-ubuntu-cuda-13.4-x64.tar.gz
tar -xzf runtime/llama-b11840-cuda13.4.tar.gz -C runtime
```
