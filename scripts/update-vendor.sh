#!/usr/bin/env bash
# Update vendored llama.cpp files (gguf-py + convert_hf_to_gguf.py)
# Usage: ./scripts/update-vendor.sh [commit-or-ref]
# Defaults to latest upstream master.

set -euo pipefail

REPO="https://github.com/ggml-org/llama.cpp.git"
REF="${1:-master}"
VENDOR_DIR="$(cd "$(dirname "$0")/../src/gguf_clone/_vendor" && pwd)"
VENDOR_INFO="$(cd "$(dirname "$0")/../src/gguf_clone" && pwd)/vendor_info.py"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Cloning llama.cpp at ref: $REF"
git clone --filter=blob:none --no-checkout "$REPO" "$TMPDIR/llama.cpp" 2>&1 | tail -1
cd "$TMPDIR/llama.cpp"
# Try known locations for convert_hf_to_gguf.py
CONVERT_PATHS=("convert_hf_to_gguf.py" "examples/convert_hf_to_gguf.py" "tools/convert_hf_to_gguf.py")
CONVERT_SRC=""
for p in "${CONVERT_PATHS[@]}"; do
  if git cat-file -e "$REF:$p" 2>/dev/null; then
    CONVERT_SRC="$p"
    break
  fi
done
[ -z "$CONVERT_SRC" ] && { echo "Could not find convert_hf_to_gguf.py in upstream"; exit 1; }

git checkout "$REF" -- gguf-py/gguf/ "$CONVERT_SRC" LICENSE

COMMIT="$(git rev-parse "$REF")"
echo "Resolved commit: $COMMIT"

# Sync gguf-py (excluding __pycache__ and .pyc)
rsync -a --delete \
  --exclude='__pycache__' --exclude='*.pyc' \
  "$TMPDIR/llama.cpp/gguf-py/gguf/" "$VENDOR_DIR/gguf/"

# Sync convert_hf_to_gguf.py
cp "$TMPDIR/llama.cpp/$CONVERT_SRC" "$VENDOR_DIR/llama_cpp/convert_hf_to_gguf.py"

# Sync license
cp "$TMPDIR/llama.cpp/LICENSE" "$VENDOR_DIR/LLAMA_CPP_LICENSE"

# Update commit pin
echo "LLAMA_CPP_COMMIT = \"$COMMIT\"" > "$VENDOR_INFO"

echo "Vendor updated to $COMMIT"
