#!/bin/bash
# ==========================================================================
# Model Download Script
# ==========================================================================
# Downloads the recommended LLM and ASR models for the pipeline.
# Requires: pip install huggingface-hub
#
# GPU: NVIDIA RTX 6000 Ada (48 GB VRAM)
# Recommended: Qwen2.5-32B-Instruct Q8_0 (~34 GB VRAM)
# Alternative: Qwen2.5-72B-Instruct Q4_K_M (~42 GB VRAM)
# ==========================================================================

set -e

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

echo "==========================================="
echo " Paediatric Simulation Pipeline — Models"
echo "==========================================="
echo ""

# ------------------------------------------------------------------
# Option 1 (RECOMMENDED): Qwen2.5-32B-Instruct Q8_0
# - Near-lossless quantization
# - Excellent German + JSON output
# - ~34 GB VRAM, fast inference
# - File size: ~34 GB
# ------------------------------------------------------------------
echo "[1/2] Downloading Qwen2.5-32B-Instruct (Q8_0)..."
echo "      This is ~34 GB — will take a while."
echo ""

hf download \
    Qwen/Qwen2.5-32B-Instruct-GGUF \
    --include "qwen2.5-32b-instruct-q8_0*.gguf" \
    --local-dir "$MODEL_DIR"

# If the file is split, merge it:
SPLIT_FILE="$MODEL_DIR/qwen2.5-32b-instruct-q8_0-00001-of-00002.gguf"
MERGED_FILE="$MODEL_DIR/qwen2.5-32b-instruct-q8_0.gguf"
if [ -f "$SPLIT_FILE" ] && [ ! -f "$MERGED_FILE" ]; then
    echo "Merging split GGUF files..."
    echo "Run: llama-gguf-split --merge $SPLIT_FILE $MERGED_FILE"
    echo "(Requires llama.cpp built locally)"
fi

echo ""
echo "✓ LLM downloaded."
echo ""

# ------------------------------------------------------------------
# Whisper large-v3 (downloaded automatically by faster-whisper
# on first run, but you can pre-download it)
# ------------------------------------------------------------------
echo "[2/2] Whisper large-v3 will download automatically on first run."
echo "      (~3 GB, cached in ~/.cache/huggingface/)"
echo "      To pre-download, run in Python:"
echo ""
echo '      from faster_whisper import WhisperModel'
echo '      model = WhisperModel("large-v3", device="cuda", compute_type="float16")'
echo ""

echo "==========================================="
echo " Done!"
echo ""
echo " Update config/pipeline_config.yaml:"
echo "   model_path: \"models/qwen2.5-32b-instruct-q8_0.gguf\""
echo "==========================================="

# ==========================================================================
# ALTERNATIVE: Qwen2.5-72B-Instruct Q4_K_M (if you want max quality)
# Uncomment the block below and comment out the 32B block above.
# ==========================================================================
#
# echo "[1/2] Downloading Qwen2.5-72B-Instruct (Q4_K_M)..."
#
# huggingface-cli download \
#     Qwen/Qwen2.5-72B-Instruct-GGUF \
#     --include "qwen2.5-72b-instruct-q4_k_m*.gguf" \
#     --local-dir "$MODEL_DIR" \
#     --local-dir-use-symlinks False
#
# # Update config: model_path: "models/qwen2.5-72b-instruct-q4_k_m.gguf"
