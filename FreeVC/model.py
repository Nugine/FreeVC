from .config import Config
from .net import load_net
from .mel_processing import spec_to_mel_torch, mel_spectrogram_torch
from . import vits
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from torch.nn.functional import l1_loss

import lightning as L
import torch


class FreeVCModel(L.LightningModule):
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
        if self.config.net.use_pretrained_spk:
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

        y_hat = self.net_g.infer(ssl, g, mel)
        return y_hat

    def training_step(self, batch, batch_idx):
        ssl, spec, y, spk = batch

        if self.config.net.use_pretrained_spk:
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

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)  # type:ignore
        opt_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_mel = l1_loss(y_mel, y_hat_mel) * self.config.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        opt_g.step()

        self.log_dict(
            {
                "loss_disc_all": loss_disc_all,
                "loss_gen_all": loss_gen_all,
                "loss_gen": loss_gen,
                "loss_fm": loss_fm,
                "loss_mel": loss_mel,
                "loss_kl": loss_kl,
            }
        )
