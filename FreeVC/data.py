from .env import cli
from .config import get_hard_config
from .utils import read_txt_lines
from .wavlm import load_wavlm, calc_ssl_features

import os
import random

import librosa
import numpy as np
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
import torch


def downsample(args):
    in_dir, wav_name, target = args

    speaker = wav_name[:4]
    wav_path = os.path.join(in_dir, speaker, wav_name)

    if not os.path.exists(wav_path):
        return

    # speaker 's5', 'p280', 'p315' are excluded,
    if "_mic2.flac" not in wav_path:
        return

    wav = None

    for out_dir, target_sr in target:
        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path = os.path.join(out_dir, speaker, save_name)
        if os.path.exists(save_path):
            continue

        if wav is None:
            wav, src_sr = librosa.load(wav_path)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = 0.98 * wav / peak

        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        target_wav = librosa.resample(wav, orig_sr=src_sr, target_sr=target_sr)
        wavfile.write(save_path, target_sr, (target_wav * np.iinfo(np.int16).max).astype(np.int16))


@cli.command()
def resample_vctk():
    config = get_hard_config()

    in_dir = config.dataset.vctk_dir
    target = [
        (config.dataset.vctk_16k_dir, 16000),
        (config.dataset.vctk_22k_dir, 22050),
    ]

    pool = Pool(processes=cpu_count() - 2)

    wav_names = []
    for speaker in os.listdir(in_dir):
        spk_dir = os.path.join(in_dir, speaker)
        if os.path.isdir(spk_dir):
            wav_names.extend(os.listdir(spk_dir))

    tasks = [(in_dir, wav_name, target) for wav_name in wav_names]

    with tqdm(total=len(tasks)) as pbar:
        for _ in pool.imap_unordered(downsample, tasks):
            pbar.update()

    print("Done!")


@cli.command()
def generate_split():
    config = get_hard_config()

    src_dir = config.dataset.vctk_16k_dir
    split_dir = config.dataset.split_dir

    train = []
    val = []
    test = []

    for speaker in os.listdir(src_dir):
        wav_names = os.listdir(os.path.join(src_dir, speaker))
        random.shuffle(wav_names)
        train.extend(wav_names[2:-10])
        val.extend(wav_names[:2])
        test.extend(wav_names[-10:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_list = os.path.join(split_dir, "train.txt")
    val_list = os.path.join(split_dir, "val.txt")
    test_list = os.path.join(split_dir, "test.txt")

    os.makedirs(split_dir, exist_ok=True)

    for list_path, wav_names in zip([train_list, val_list, test_list], [train, val, test]):
        with open(list_path, "w") as f:
            for wav_name in wav_names:
                speaker = wav_name[:4]
                f.write(f"{speaker}/{wav_name}" + "\n")

    print("Done!")


class VCTK:
    def __init__(self, *, use_sr_augment: bool) -> None:
        super().__init__()

        config = get_hard_config()
        self.vctk_16k_dir = config.dataset.vctk_16k_dir
        self.vctk_22k_dir = config.dataset.vctk_22k_dir

        self.use_sr_augment = use_sr_augment

        self.audio_paths = {}
        for split in ["train", "val", "test"]:
            split_path = os.path.join(config.dataset.split_dir, f"{split}.txt")
            self.audio_paths[split] = read_txt_lines(split_path)

        self.wavlm = load_wavlm()

    def load_wav_16k(self, path):
        wav, _ = torchaudio.load(os.path.join(self.vctk_16k_dir, path))
        return wav

    def load_wav_22k(self, path):
        wav, _ = torchaudio.load(os.path.join(self.vctk_22k_dir, path))
        return wav


class VCTKDataset(Dataset):
    def __init__(self, vctk: VCTK, split: str):
        super().__init__()
        self.vctk = vctk
        self.audio_paths = vctk.audio_paths[split]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        wav = self.vctk.load_wav_16k(self.audio_paths[idx])

        ssl = calc_ssl_features(self.vctk.wavlm, wav)

        return {"wav": wav, "ssl": ssl}


class VCTKCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        max_wav_length = max(sample["wav"].shape[-1] for sample in batch)
        wavs = []
        for sample in batch:
            wav = sample["wav"]
            if wav.shape[-1] < max_wav_length:
                wav = F.pad(wav, (0, max_wav_length - wav.shape[-1]))
            wavs.append(wav)
        return {"wav": torch.stack(wavs)}


# @cli.command()
# def show_vctk_dataset():
#     dataset = VCTKDataset("train")
#     print(len(dataset))
#     print(dataset[0])

#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=VCTKCollate())
#     batch = next(iter(dataloader))
#     print(batch["wav"].shape)


class VCTKDataModule(LightningDataModule):
    def __init__(self, *, batch_size: int, num_workers: int, use_sr_augment: bool):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.vctk = VCTK(use_sr_augment=False)
        self.train_set = VCTKDataset(self.vctk, "train")
        self.val_set = VCTKDataset(self.vctk, "val")
        self.test_set = VCTKDataset(self.vctk, "test")

    def _create_dataloader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=VCTKCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)
