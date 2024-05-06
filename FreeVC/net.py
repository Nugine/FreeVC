from . import vits
from .env import cli
from .config import Config


import torch
import torch.nn as nn


def load_net(config: Config):
    net_g = SynthesizerTrn(
        spec_channels=config.data.filter_length // 2 + 1,
        segment_size=config.net.segment_size // config.data.hop_length,
        inter_channels=config.net.inter_channels,
        hidden_channels=config.net.hidden_channels,
        filter_channels=config.net.filter_channels,
        n_heads=config.net.n_heads,
        n_layers=config.net.n_layers,
        kernel_size=config.net.kernel_size,
        p_dropout=config.net.p_dropout,
        resblock=config.net.resblock,
        resblock_kernel_sizes=config.net.resblock_kernel_sizes,
        resblock_dilation_sizes=config.net.resblock_dilation_sizes,
        upsample_rates=config.net.upsample_rates,
        upsample_initial_channel=config.net.upsample_initial_channel,
        upsample_kernel_sizes=config.net.upsample_kernel_sizes,
        gin_channels=config.net.gin_channels,
        ssl_dim=config.net.ssl_dim,
        use_pretrained_spk=config.data.use_pretrained_spk,
    )
    net_d = vits.MultiPeriodDiscriminator(
        use_spectral_norm=config.net.use_spectral_norm,
    )
    return (net_g, net_d)


@cli.command()
def show_net():
    config = Config()
    net_g, net_d = load_net(config)
    print(net_g)
    print(net_d)


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        mel_n_channels=80,
        model_num_layers=3,
        model_hidden_size=256,
        model_embedding_size=256,
    ):
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, dim=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


class Encoder(vits.PosteriorEncoder):
    pass


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
        use_pretrained_spk,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.use_pretrained_spk = use_pretrained_spk

        self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
        self.dec = vits.Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )
        self.flow = vits.ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=gin_channels,
        )
        if not self.use_pretrained_spk:
            self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)

    def forward(self, c, spec, g=None, mel=None, c_lengths=None, spec_lengths=None):
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

        if not self.use_pretrained_spk:
            g = self.enc_spk(mel.transpose(1, 2))  # type:ignore
        g = g.unsqueeze(-1)  # type:ignore

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        # print(f"DBG: c.shape={c.shape}, spec.shape={spec.shape}, z.shape={z.shape}, spec_lengths.shape={spec_lengths.shape}, segment_size={self.segment_size}")

        z_slice, ids_slice = vits.rand_slice_segments(z, spec_lengths, min(self.segment_size, z.size(-1)))
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, g=None, mel=None, c_lengths=None):
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if not self.use_pretrained_spk:
            g = self.enc_spk.embed_utterance(mel.transpose(1, 2))  # type:ignore
        g = g.unsqueeze(-1)  # type:ignore

        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g)

        return o
