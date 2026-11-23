#!/usr/bin/env python3
"""Queue an API-format ComfyUI graph on a running ComfyUI, wait for it and report the time and the output files.

    python run_graph.py graphs/h3-ref2va-fast-example.json --image ref.png --comfy-dir ComfyUI
    python run_graph.py graphs/h3-ref2va-fast-example.json --image ref.png --no-spark    # any GPU, slower

Options:
  --image PATH        copied into ComfyUI's input folder and used for every LoadImage node
  --prompt TEXT       replaces the prompt of the MiniMax H3 conditioning node
  --seed N            replaces the noise seed
  --frames N          replaces the length (24 fps; H3 lengths are 17k+5 for full latents, other values work too)
  --no-spark          drops the Spark-H3 attention node (for GPUs that are not compute capability 12.x)
  --no-spectrum       drops the Spectrum node
  --timings FILE      also writes the result as JSON (default: run-<time>.json next to this script)
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path


def call(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def drop_node(graph: dict, class_type: str) -> None:
    """Remove every node of this class and route its consumers to the node's own model input."""
    for node_id in [k for k, n in graph.items() if n["class_type"] == class_type]:
        upstream = graph[node_id]["inputs"]["model"]
        del graph[node_id]
        for node in graph.values():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                    node["inputs"][name] = upstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", default="ComfyUI")
    parser.add_argument("--image")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--no-spark", action="store_true")
    parser.add_argument("--no-spectrum", action="store_true")
    parser.add_argument("--timings")
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    if args.no_spark:
        drop_node(graph, "MiniMaxH3SparkAttentionSM120")
    if args.no_spectrum:
        drop_node(graph, "SpectrumApplyMiniMaxH3")
    for node in graph.values():
        kind, inputs = node["class_type"], node["inputs"]
        if kind == "LoadImage" and args.image:
            name = "h3_fast_" + Path(args.image).name
            shutil.copyfile(args.image, Path(args.comfy_dir) / "input" / name)
            inputs["image"] = name
        if kind.startswith("MiniMaxH3") and "prompt" in inputs and args.prompt:
            inputs["prompt"] = args.prompt
        if kind.startswith("MiniMaxH3") and "length" in inputs and args.frames:
            inputs["length"] = args.frames
        if kind == "RandomNoise" and args.seed is not None:
            inputs["noise_seed"] = args.seed

    started = time.time()
    prompt_id = call(f"{args.server}/prompt", {"prompt": graph})["prompt_id"]
    print(f"queued {prompt_id}", flush=True)
    while True:
        time.sleep(5)
        history = call(f"{args.server}/history/{prompt_id}")
        if prompt_id in history:
            break
    entry = history[prompt_id]
    status = entry.get("status", {})
    outputs = [f"{item.get('subfolder', '')}/{item.get('filename')}".lstrip("/")
               for node in entry.get("outputs", {}).values()
               for key in ("images", "videos", "gifs", "audio")
               for item in node.get(key, [])]
    result = {"prompt_id": prompt_id, "status": status.get("status_str"), "seconds": round(time.time() - started),
              "outputs": outputs, "messages": [m[0] for m in status.get("messages", [])]}
    print(json.dumps(result, indent=1), flush=True)
    timings = Path(args.timings) if args.timings else Path(__file__).resolve().parent / f"run-{int(started)}.json"
    timings.write_text(json.dumps(result, indent=1))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
