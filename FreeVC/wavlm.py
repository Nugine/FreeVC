from .env import cli

import os

import huggingface_hub
import torch
from transformers import WavLMModel

REPO_ID = "microsoft/wavlm-large"


@cli.command()
def download_wavlm():
    huggingface_hub.snapshot_download(repo_id=REPO_ID, repo_type="model", resume_download=True)


def rename_state_key(state_dict, key, new_key):
    state_dict[new_key] = state_dict.pop(key)


@cli.command()
def load_wavlm():
    # https://github.com/huggingface/transformers/issues/30469
    bin_path = huggingface_hub.try_to_load_from_cache(repo_id=REPO_ID, filename="pytorch_model.bin")
    assert bin_path is not None

    # https://github.com/pytorch/pytorch/issues/102999
    # https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    state_dict = torch.load(bin_path)
    rename_state_key(
        state_dict,
        "encoder.pos_conv_embed.conv.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    )
    rename_state_key(
        state_dict,
        "encoder.pos_conv_embed.conv.weight_v",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    )

    model = WavLMModel.from_pretrained(os.path.dirname(bin_path), state_dict=state_dict)
    assert isinstance(model, WavLMModel)
    return model


@torch.no_grad()
def calc_ssl_features(model: WavLMModel, wav):
    return model(wav).last_hidden_state.transpose(1, 2)
