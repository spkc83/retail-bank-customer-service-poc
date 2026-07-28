#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODEL_DIR LLAMA_CPP_DIR OUTPUT_DIR" >&2
  exit 2
fi

model_dir=$1
llama_cpp_dir=$2
output_dir=$3
converter="$llama_cpp_dir/convert_hf_to_gguf.py"
quantizer="$llama_cpp_dir/llama-quantize"

if [[ ! -f "$converter" ]]; then
  echo "Missing converter: $converter" >&2
  exit 2
fi
if [[ ! -x "$quantizer" ]]; then
  quantizer="$llama_cpp_dir/build/bin/llama-quantize"
fi
if [[ ! -x "$quantizer" ]]; then
  echo "Missing llama-quantize binary under: $llama_cpp_dir" >&2
  exit 2
fi
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "MODEL_DIR is not a downloaded Transformers checkpoint: $model_dir" >&2
  exit 2
fi

mkdir -p "$output_dir"
bf16_path="$output_dir/retail-bank-servicing-moe-9b-bf16.gguf"
q4_path="$output_dir/retail-bank-servicing-moe-9b-q4_k_m.gguf"

python "$converter" \
  "$model_dir" \
  --outfile "$bf16_path" \
  --outtype bf16

"$quantizer" "$bf16_path" "$q4_path" Q4_K_M

echo "BF16 GGUF: $bf16_path"
echo "Q4_K_M GGUF: $q4_path"
