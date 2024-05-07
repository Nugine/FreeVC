from .config import Config, DataConfig
from .net import load_net
from .mel_processing import spec_to_mel_torch, mel_spectrogram_torch
from . import vits
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .data import VCTKDataModule, calc_ssl_features
from .env import cli
from .wavlm import load_wavlm

from speaker_encoder.voice_encoder import SpeakerEncoder

import dataclasses

from torch.nn.functional import l1_loss
from lightning import LightningModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
import torch
import librosa
from scipy.io import wavfile


class FreeVCModel(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.net_g, self.net_d = load_net(config)

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=self.config.train.learning_rate)
        optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.config.train.learning_rate)

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=self.config.train.lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=self.config.train.lr_decay)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def forward(self, ssl, spec, y, spk):
        if self.config.data.use_pretrained_spk:
            g = spk
        else:
            g = None

        mel = spec_to_mel_torch(
            spec,
            n_fft=self.config.data.filter_length,
            num_mels=self.config.data.n_mel_channels,
            sampling_rate=self.config.data.sampling_rate,
            fmin=self.config.data.mel_fmin,
            fmax=self.config.data.mel_fmax,
        )

        y_hat = self.net_g.infer(ssl, g=g, mel=mel)
        return y_hat

    def training_step(self, batch, batch_idx):
        ssl, spec, y, spk = batch

        if self.config.data.use_pretrained_spk:
            g = spk
        else:
            g = None

        mel = spec_to_mel_torch(
            spec,
            n_fft=self.config.data.filter_length,
            num_mels=self.config.data.n_mel_channels,
            sampling_rate=self.config.data.sampling_rate,
            fmin=self.config.data.mel_fmin,
            fmax=self.config.data.mel_fmax,
        )

        y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(ssl, spec, g=g, mel=mel)

        y_mel = vits.slice_segments(mel, ids_slice, self.config.net.segment_size // self.config.data.hop_length)

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            n_fft=self.config.data.filter_length,
            num_mels=self.config.data.n_mel_channels,
            sampling_rate=self.config.data.sampling_rate,
            hop_size=self.config.data.hop_length,
            win_size=self.config.data.win_length,
            fmin=self.config.data.mel_fmin,
            fmax=self.config.data.mel_fmax,
        )

        y = vits.slice_segments(y, ids_slice * self.config.data.hop_length, self.config.net.segment_size)

        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())

        opt_g, opt_d = self.optimizers()  # type: ignore
        lr_g = opt_g.param_groups[0]["lr"]
        lr_d = opt_d.param_groups[0]["lr"]

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)  # type:ignore
        grad_norm_d = vits.clip_grad_value_(self.net_d.parameters(), None)
        opt_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_mel = l1_loss(y_mel, y_hat_mel) * self.config.train.c_mel  # reduction???
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        grad_norm_g = vits.clip_grad_value_(self.net_g.parameters(), None)
        opt_g.step()

        self.log_dict(
            {
                "loss_disc_all": loss_disc_all,
                "loss_gen_all": loss_gen_all,
                "loss_gen": loss_gen,
                "loss_fm": loss_fm,
                "loss_mel": loss_mel,
                "loss_kl": loss_kl,
                "lr_g": lr_g,
                "lr_d": lr_d,
                "grad_norm_g": grad_norm_g,
                "grad_norm_d": grad_norm_d,
            }
        )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for field in dataclasses.fields(DataConfig):
            key = field.name
            parser.link_arguments(f"data.config.{key}", f"model.config.data.{key}")

        parser.set_defaults(
            {
                "trainer.enable_checkpointing": True,
                "trainer.max_epochs": 10,
                "trainer.log_every_n_steps": 5,
            }
        )


@cli.command()
@torch.no_grad()
def convert(ckpt_path: str, src_path: str, tgt_path: str, save_path: str):
    model = FreeVCModel.load_from_checkpoint(ckpt_path)
    model.net_g.load_state_dict(torch.load("./ckpt/freevc/freevc.pth")["model"])
    model = model.cuda()

    wavlm = load_wavlm().cuda()  # type:ignore

    if model.config.data.use_pretrained_spk:
        spk_encoder = SpeakerEncoder(model.config.data.pretrained_spk_ckpt_path)

    print("Model loaded")

    wav_src, _ = librosa.load(src_path, sr=model.config.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    ssl = calc_ssl_features(wavlm, wav_src)

    wav_tgt, _ = librosa.load(tgt_path, sr=model.config.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

    if model.config.data.use_pretrained_spk:
        g_tgt = spk_encoder.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
        audio = model.net_g.infer(ssl, g=g_tgt)
    else:
        wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
        mel_tgt = mel_spectrogram_torch(
            wav_tgt,
            n_fft=model.config.data.filter_length,
            num_mels=model.config.data.n_mel_channels,
            sampling_rate=model.config.data.sampling_rate,
            hop_size=model.config.data.hop_length,
            win_size=model.config.data.win_length,
            fmin=model.config.data.mel_fmin,
            fmax=model.config.data.mel_fmax,
        )
        audio = model.net_g.infer(ssl, mel=mel_tgt)

    if False:
        mel_src = mel_spectrogram_torch(
            wav_src,
            n_fft=model.config.data.filter_length,
            num_mels=model.config.data.n_mel_channels,
            sampling_rate=model.config.data.sampling_rate,
            hop_size=model.config.data.hop_length,
            win_size=model.config.data.win_length,
            fmin=model.config.data.mel_fmin,
            fmax=model.config.data.mel_fmax,
        )
        mel_audio = mel_spectrogram_torch(
            audio.squeeze(0),
            n_fft=model.config.data.filter_length,
            num_mels=model.config.data.n_mel_channels,
            sampling_rate=model.config.data.sampling_rate,
            hop_size=model.config.data.hop_length,
            win_size=model.config.data.win_length,
            fmin=model.config.data.mel_fmin,
            fmax=model.config.data.mel_fmax,
        )
        import matplotlib.pyplot as plt

        # plt mel spectrogram
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(mel_src.squeeze(0).cpu().numpy())
        plt.title("mel_src")
        plt.subplot(1, 2, 2)
        plt.imshow(mel_audio.squeeze(0).cpu().numpy())
        plt.title("mel_audio")
        plt.show()

    audio = audio.squeeze().cpu().float().numpy()
    print("audio:")
    print(audio.shape)
    print(audio)
    wavfile.write(save_path, rate=model.config.data.sampling_rate, data=audio)

    print("Done!")


if __name__ == "__main__":
    cli = MyLightningCLI(FreeVCModel, VCTKDataModule)
