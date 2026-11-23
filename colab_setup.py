"""Colab cell: clone this repo into /content and run setup.sh, streaming its output into the cell.

    colab exec -f colab_setup.py --timeout 600

A plain subprocess call would write to the kernel's own stdout, which never reaches the cell, so the Colab CLI sees
silence and gives up. This reads the output and prints it line by line instead. Set the same environment variables
as for setup.sh (MODELS, SPARK, ...) in os.environ before running if you need other choices.
"""
import os
import subprocess
import sys

REPO = "https://github.com/the3dprintingbelgian/comfyui-h3-fast-recipe"
COMMAND = ("cd /content && (test -d comfyui-h3-fast-recipe || git clone -q {repo}) "
           "&& git -C comfyui-h3-fast-recipe pull -q --ff-only "
           "&& DEST=/content/ComfyUI PYTHON={python} bash comfyui-h3-fast-recipe/setup.sh").format(
               repo=REPO, python=sys.executable)

process = subprocess.Popen(COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           bufsize=1, env=os.environ.copy())
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait() != 0:
    raise SystemExit(f"setup.sh failed with exit code {process.returncode}")
