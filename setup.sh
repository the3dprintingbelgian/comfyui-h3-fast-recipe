#!/usr/bin/env bash
# Set up the MiniMax H3 fast recipe on a fresh Linux machine (a Colab GPU runtime, a rented box, a new PC):
#   ComfyUI v0.45.0 + patches/comfyui-v0.45.0.patch
#   + Spark-H3 0.2.0 sparse attention (Blackwell / sm_120 build, from its GitHub release, checksum-verified)
#   + ComfyUI-Spectrum-MiniMax-H3 (pinned commit)
#   + the Comfy-Org/MiniMax-H3 weights (pinned Hugging Face revision).
#
# Usage:  bash setup.sh                      # installs into ./ComfyUI, downloads the reference-to-video weights
#         DEST=/content/ComfyUI MODELS=both bash setup.sh
# Env:    DEST    ComfyUI folder (default ./ComfyUI)
#         MODELS  ref2va (default) | fl2va | both | none
#         TORCH   auto (default: install torch 2.11.0+cu130 unless it is already there) | skip
#         SPARK   auto (default: only on a compute capability 12.x GPU) | yes | no
#         PYTHON  python to use (default: python3)
# Re-running is safe: finished steps are skipped.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${DEST:-$PWD/ComfyUI}
MODELS=${MODELS:-ref2va}
TORCH=${TORCH:-auto}
SPARK=${SPARK:-auto}
PYTHON=${PYTHON:-python3}

COMFY_REPO=https://github.com/Comfy-Org/ComfyUI.git
COMFY_COMMIT=989d93d0d968e704c589ce0146bb18c77f6b50cc          # v0.45.0
SPARK_TAG=comfyui-v0.2.0
SPARK_ZIP=ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm120-cu130.zip
SPARK_SHA256=b51a3ab20c6e677de67bfc3693727ff3789c9d2077680b4ff540d5766593858d
SPARK_WHEEL=spark_h3_kitchen-0.2.37+spark.h3.sm120.cu130.1-cp312-abi3-linux_x86_64.whl
SPECTRUM_REPO=https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git
SPECTRUM_COMMIT=7df883bb2361b7884ad789e97effd1dc37c8963a
HF_REPO=Comfy-Org/MiniMax-H3
HF_REVISION=4b53a35191437de05370da548e59aeabe2a5a09a

say() { printf '\n== %s\n' "$*"; }
pip_install() { "$PYTHON" -m pip install -q "$@"; }

say "GPU"
GPU=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
echo "${GPU:-no NVIDIA GPU found}"
CAP=$(echo "$GPU" | awk -F', ' '{print $2}')
DRIVER=$(echo "$GPU" | awk -F', ' '{print $3}' | cut -d. -f1)
if [ "$SPARK" = auto ]; then
  if [ "${CAP%%.*}" = 12 ]; then SPARK=yes; else SPARK=no; fi
fi
if [ "$SPARK" = yes ] && [ "${CAP%%.*}" != 12 ]; then
  echo "Spark-H3's build here is for compute capability 12.x (RTX 50xx / RTX PRO 6000 Blackwell); this GPU is ${CAP:-unknown}." >&2
  exit 1
fi
if [ -n "${DRIVER:-}" ] && [ "$DRIVER" -lt 580 ]; then
  echo "Warning: driver $DRIVER is older than 580; CUDA 13 builds (torch cu130, Spark) need 580 or newer." >&2
fi
echo "Spark-H3: $SPARK"

say "ComfyUI v0.45.0 at $DEST"
if [ ! -d "$DEST/.git" ]; then
  git clone -q "$COMFY_REPO" "$DEST"
fi
git -C "$DEST" checkout -q "$COMFY_COMMIT"
if git -C "$DEST" apply --reverse --check "$HERE/patches/comfyui-v0.45.0.patch" 2>/dev/null; then
  echo "patch already applied"
else
  git -C "$DEST" apply "$HERE/patches/comfyui-v0.45.0.patch"
  echo "patch applied"
fi

say "Python packages"
if [ "$TORCH" = auto ]; then
  if "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.11.0') and torch.version.cuda == '13.0' else 1)" 2>/dev/null; then
    echo "torch 2.11.0+cu130 already installed"
  else
    pip_install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
  fi
fi
pip_install -r "$DEST/requirements.txt"
pip_install "huggingface_hub>=0.25" hf_transfer
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok' if torch.cuda.is_available() else 'NO GPU')"

if [ "$SPARK" = yes ]; then
  say "Spark-H3 0.2.0 (sm_120)"
  NODE="$DEST/custom_nodes/ComfyUI-Spark-H3"
  if [ ! -f "$NODE/comfyui_nodes.py" ]; then
    TMP=$(mktemp -d)
    curl -fsSL -o "$TMP/$SPARK_ZIP" "https://github.com/zechengtang/Spark-H3/releases/download/$SPARK_TAG/$SPARK_ZIP"
    echo "$SPARK_SHA256  $TMP/$SPARK_ZIP" | sha256sum -c -
    (cd "$TMP" && "$PYTHON" -m zipfile -e "$SPARK_ZIP" .)
    pip_install --no-deps "$TMP/ComfyUI-Spark-H3/wheelhouse/$SPARK_WHEEL"
    # Like the reference install: the node files only. The release's own installer, kernel builder, patches,
    # wheelhouse and native/ folders are not used.
    for extra in install.py kernel_builder.py package_manager.py patches wheelhouse native scripts; do
      rm -rf -- "${TMP:?}/ComfyUI-Spark-H3/$extra"
    done
    mv "$TMP/ComfyUI-Spark-H3" "$NODE"
    rm -rf -- "${TMP:?}"
  fi
  "$PYTHON" -c "import spark_h3_kitchen; print('spark_h3_kitchen ok')"
fi

say "ComfyUI-Spectrum-MiniMax-H3"
SPEC="$DEST/custom_nodes/ComfyUI-Spectrum-MiniMax-H3"
[ -d "$SPEC/.git" ] || git clone -q "$SPECTRUM_REPO" "$SPEC"
git -C "$SPEC" checkout -q "$SPECTRUM_COMMIT"

say "Weights ($MODELS) from $HF_REPO@${HF_REVISION:0:7}"
FILES="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors"
case "$MODELS" in
  ref2va) FILES="$FILES diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" ;;
  fl2va)  FILES="$FILES diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ;;
  both)   FILES="$FILES diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ;;
  none)   FILES="" ;;
  *) echo "MODELS must be ref2va, fl2va, both or none" >&2; exit 1 ;;
esac
if [ -n "$FILES" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$PYTHON" - "$DEST/models" "$HF_REPO" "$HF_REVISION" $FILES <<'PY'
import sys, time
from huggingface_hub import hf_hub_download
models, repo, revision, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
start = time.time()
for name in files:
    hf_hub_download(repo, name, revision=revision, local_dir=models)
    print(f"  {name} ({time.time() - start:.0f} s)", flush=True)
PY
fi

say "Done"
cat <<EOF
Start ComfyUI (local only):   cd "$DEST" && $PYTHON main.py --listen 127.0.0.1 --port 8188
Run the example:              $PYTHON "$HERE/run_graph.py" "$HERE/graphs/h3-ref2va-fast-example.json" --image your-reference.png --comfy-dir "$DEST"
EOF
