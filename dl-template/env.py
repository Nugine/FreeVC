import os
import json

import torch

import typer

"""
https://github.com/Nugine/dl-template
"""


# detect notebook
try:
    __IPYTHON__  # type: ignore
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False

# detect platform
PLATFORM = "local"
if os.getenv("COLAB_RELEASE_TAG"):
    PLATFORM = "colab"
elif os.path.exists("/kaggle"):
    PLATFORM = "kaggle"
elif os.path.exists("/root/autodl-tmp"):
    PLATFORM = "autodl"

# detect device
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

cli = typer.Typer()


@cli.command()
def mount_gdrive():
    assert PLATFORM == "colab"
    colab = __import__("google.colab").colab
    if not os.path.exists("/gdrive"):
        colab.drive.mount("/gdrive")


@cli.command()
def show_env():
    packages = {
        "torch": torch.__version__,
    }
    info = {
        "IN_NOTEBOOK": IN_NOTEBOOK,
        "PLATFORM": PLATFORM,
        "torch": {
            "DEVICE": str(DEVICE),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "packages": packages,
    }
    print(json.dumps(info, indent=4))
