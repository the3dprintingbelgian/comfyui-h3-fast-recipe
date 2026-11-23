# ComfyUI H3 fast recipe

A small, reproducible setup for running **MiniMax H3** (video + audio) in ComfyUI with the speed-ups we use on
Blackwell GPUs, on any fresh Linux machine: a Colab GPU runtime, a rented server or a new PC.

What it puts together:

| Piece | Version | Source |
|---|---|---|
| ComfyUI | v0.45.0 (`989d93d0`) + [`patches/comfyui-v0.45.0.patch`](patches/comfyui-v0.45.0.patch) | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) |
| Spark-H3 sparse attention | 0.2.0, Linux sm_120 / CUDA 13 build, sha256-checked | [zechengtang/Spark-H3](https://github.com/zechengtang/Spark-H3) release `comfyui-v0.2.0` |
| Spectrum for H3 | `7df883b` | [xmarre/ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) |
| torch | 2.11.0 + CUDA 13.0 | PyTorch cu130 wheels |
| Weights | pruned int8 diffusion model, NVFP4 text encoder, video + audio VAEs | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) at a pinned revision |

Everything except the patch file is downloaded from its original publisher by `setup.sh`; nothing is re-hosted here.

## Quick start

```bash
git clone https://github.com/the3dprintingbelgian/comfyui-h3-fast-recipe
bash comfyui-h3-fast-recipe/setup.sh          # ComfyUI in ./ComfyUI, ~42 GB of weights
cd ComfyUI && python3 main.py --listen 127.0.0.1 --port 8188 &
python3 ../comfyui-h3-fast-recipe/run_graph.py ../comfyui-h3-fast-recipe/graphs/h3-ref2va-fast-example.json \
    --image my-character.png --comfy-dir .
```

`setup.sh` options (environment variables): `DEST` (ComfyUI folder), `MODELS=ref2va|fl2va|both|none`,
`TORCH=auto|skip`, `SPARK=auto|yes|no`, `PYTHON`. It is safe to run again.

`run_graph.py` waits for the result and prints the run time and output files. It can replace the prompt, seed,
length and reference image, and drop the Spark or Spectrum node (`--no-spark`, `--no-spectrum`).

### GPUs

- **Spark-H3 needs compute capability 12.x** (RTX 50xx, RTX PRO 6000 Blackwell; Colab's "G4"). `setup.sh` installs it
  only on such a GPU. On anything else (A100, L4, H100) run the graph with `--no-spark`.
- CUDA 13 builds need NVIDIA driver 580 or newer.
- The NVFP4 text encoder is fastest on Blackwell. Other GPUs can use
  `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` from the same Hugging Face repo instead.

### Colab (google-colab-cli)

```bash
colab new --gpu G4                                   # RTX PRO 6000 Blackwell, 96 GB
colab exec -f colab_setup.py --timeout 600           # clones this repo into /content and runs setup.sh
colab stop                                           # stop the runtime when done, it bills while it runs
```

[`colab_setup.py`](colab_setup.py) streams the setup's output line by line. Still pass a long `--timeout` to
`colab exec`: by default it gives up after 30 s without output, while the cell keeps running on the server.

## The example graph

[`graphs/h3-ref2va-fast-example.json`](graphs/h3-ref2va-fast-example.json) is an API-format graph: reference-to-video,
one reference image, 1920×1088, 90 frames at 24 fps, 32 steps (res_multistep / simple, CFG 1), with:

- `MiniMaxH3SparkAttentionSM120`: fidelity mode, top-k ratio 0.2, warmup ratio 0.2
- `SpectrumApplyMiniMaxH3`: blend 0.5, degree 1, window 2.0, history in system RAM

On an RTX 5090 this recipe took about 4 minutes for a 3-second 1920×1088 shot, against about 12 minutes for the same
shot with stock nodes and the int8 text encoder.

## What the patch changes

`patches/comfyui-v0.45.0.patch` touches five files of ComfyUI v0.45.0:

1. **MiniMax H3 Fun ControlNet as a conditioning ControlNet** (`comfy/controlnet.py`, `comfy/ldm/minimax/model.py`,
   `comfy_extras/nodes_minimax_h3.py`): loads the Fun ControlNet-Union (VideoX-Fun naming or core naming, int8 too)
   and applies a control video, an inpainting mask with its source video, or both, through a regular
   `Apply MiniMax H3 Fun ControlNet` node with start/end percent. The upstream model-patch node stays available as
   `MiniMaxH3FunControlNetApplyV2`.
2. **Reference images** (`comfy_extras/nodes_minimax_h3.py`): in `max` mode a reference is also capped at 3584 px on
   its long edge (tall portrait references otherwise sent ~5.6 MP each into the vision tower and ran out of memory on
   32 GB cards with 8 references), and up to 15 reference images can be connected (stock: 9).
3. **Wan Animate 2 `log_scale`** (`comfy/ldm/wan/model_animate2.py`, `comfy_extras/nodes_wan.py`): an additive
   attention bias for the first generated latent frame, as used by distilled Animate-2 recipes (-1.3).

Differences from stock v0.45.0 to know about:

- `MiniMaxH3FunControlNetApply` becomes the conditioning version. A graph made for stock v0.45.0 that uses the
  model-patch version of that node must use `MiniMaxH3FunControlNetApplyV2` instead.
- References with a long edge above 3584 px are scaled a little further in `max` mode than stock does.
- Everything else only changes graphs that use the new inputs.

## Licences and credits

- The patch modifies ComfyUI and is released under the **GPL-3.0**, like ComfyUI ([LICENSE](LICENSE)).
- Spark-H3 is Apache-2.0 (zechengtang); ComfyUI-Spectrum-MiniMax-H3 is GPL-3.0 (xmarre). Both are downloaded from
  their own repositories, unchanged.
- The MiniMax H3 weights are published by MiniMax and repackaged by Comfy-Org under MiniMax's model licence; read it
  on the Hugging Face page before use. They are not redistributed here.
