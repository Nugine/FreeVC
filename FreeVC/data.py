from .env import cli
from .config import DataConfig
from .utils import read_txt_lines
from .wavlm import load_wavlm
from .hifigan import load_hifigan
from .mel_processing import mel_spectrogram_torch, spectrogram_torch
from . import vits

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
import torch
import torchvision.transforms.v2
import torchaudio.transforms


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
    config = DataConfig()

    in_dir = config.vctk_dir
    target = [(config.vctk_16k_dir, 16000), (config.vctk_22k_dir, 22050)]

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
    config = DataConfig()

    src_dir = config.vctk_16k_dir
    split_dir = config.split_dir

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


def mel_resize(mel, height):  # 68-92
    tgt = torchvision.transforms.v2.functional.resize(mel, [height, mel.size(-1)])
    if height >= mel.size(-2):
        return tgt[:, : mel.size(-2), :]
    else:
        silence = tgt[:, -1:, :].repeat(1, mel.size(-2) - height, 1)
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


class VCTK:
    def __init__(self, *, config: DataConfig) -> None:
        super().__init__()

        self.config = config

        self.audio_paths = {}
        for split in ["train", "val", "test"]:
            split_path = os.path.join(config.split_dir, f"{split}.txt")
            self.audio_paths[split] = read_txt_lines(split_path)

        self.wavlm = load_wavlm()
        self.wavlm = self.wavlm.cuda()  # type:ignore

        if self.config.use_sr_augment:
            self.hifigan, self.hifigan_config = load_hifigan()
            self.hifigan = self.hifigan.cuda()  # type:ignore
            self.resample_22kto16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
            self.resample_22kto16k = self.resample_22kto16k.cuda()  # type:ignore

    def load_sample(self, path):
        wav_16k, sr = torchaudio.load(os.path.join(self.config.vctk_16k_dir, path))
        assert sr == 16000
        wav_16k = wav_16k.cuda()

        wav_norm = wav_16k / self.config.max_wav_value

        spec = spectrogram_torch(
            wav_norm,
            n_fft=self.config.filter_length,
            sampling_rate=16000,
            hop_size=self.config.hop_length,
            win_size=self.config.win_length,
            center=False,
        ).squeeze_(0)

        if self.config.use_sr_augment:
            h = random.randint(68, 92)
            wav_22k, sr = torchaudio.load(os.path.join(self.config.vctk_22k_dir, path))
            assert sr == 22050
            wav_22k = wav_22k.cuda()
            wav_sr = self.sr_augment(wav_22k, h)
            ssl = self.calc_ssl_features(wav_sr)
        else:
            ssl = self.calc_ssl_features(wav_16k)

        spk = None  # TODO

        return ssl, spec, wav_norm, spk

    @torch.no_grad()
    def calc_ssl_features(self, wav):
        return self.wavlm(wav).last_hidden_state.transpose(1, 2)

    @torch.no_grad()
    def sr_augment(self, wav, h):
        mel = mel_spectrogram_torch(
            wav,
            n_fft=self.hifigan_config.n_fft,
            num_mels=self.hifigan_config.num_mels,
            sampling_rate=self.hifigan_config.sampling_rate,
            hop_size=self.hifigan_config.hop_size,
            win_size=self.hifigan_config.win_size,
            fmin=self.hifigan_config.fmin,
            fmax=self.hifigan_config.fmax,
        )

        mel_rs = mel_resize(mel, h)

        wav_rs = self.hifigan(mel_rs)[0]
        assert wav_rs.shape[0] == 1

        wav_rs = self.resample_22kto16k(wav_rs)
        return wav_rs


class VCTKDataset(Dataset):
    def __init__(self, vctk: VCTK, split: str):
        super().__init__()
        self.vctk = vctk
        self.audio_paths = vctk.audio_paths[split]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        return self.vctk.load_sample(self.audio_paths[idx])


@cli.command()
def iter_train_set():
    vctk = VCTK(config=DataConfig())
    train_set = VCTKDataset(vctk, "train")

    # for i in tqdm(range(len(train_set))):
    #     ssl, spec, wav_norm, spk = train_set[i]

    dm = VCTKDataModule(config=DataConfig())
    dm.setup("fit")
    for batch in tqdm(dm.train_dataloader()):
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)


class VCTKCollate:
    def __init__(self, config: DataConfig):
        self.config = config

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.config.use_pretrained_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        else:
            spks = None

        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, : c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            if self.config.use_pretrained_spk:
                spks[i] = row[3]  # type:ignore

        spec_seglen = (
            spec_lengths[-1] if spec_lengths[-1] < self.config.max_speclen + 1 else self.config.max_speclen + 1
        )
        wav_seglen = spec_seglen * self.config.hop_length

        spec_padded, ids_slice = vits.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)  # type:ignore
        wav_padded = vits.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)  # type:ignore
        c_padded = vits.slice_segments(c_padded, ids_slice, spec_seglen)[:, :, :-1]  # type:ignore

        spec_padded = spec_padded[:, :, :-1]
        wav_padded = wav_padded[:, :, : -self.config.hop_length]

        return c_padded, spec_padded, wav_padded, spks


class VCTKDataModule(LightningDataModule):
    def __init__(self, *, config: DataConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        self.vctk = VCTK(config=self.config)
        self.train_set = VCTKDataset(self.vctk, "train")
        self.val_set = VCTKDataset(self.vctk, "val")
        self.test_set = VCTKDataset(self.vctk, "test")

    def _create_dataloader(self, dataset, *, shuffle: bool):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=VCTKCollate(self.config),
            batch_size=self.config.batch_size,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)
