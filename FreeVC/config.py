from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    vctk_dir: str = "./data/VCTK"

    vctk_16k_dir: str = "./data/vctk-16k"
    vctk_22k_dir: str = "./data/vctk-22k"

    split_dir: str = "./data/split"

    pretrained_spk_ckpt_path: str = "speaker_encoder/ckpt/pretrained_bak_5805000.pt"

    preprocess_spk_dir: str = "./data/spk"
    preprocess_ssl_dir: str = "./data/ssl"
    preprocess_sr_dir: str = "./data/sr"

    use_sr_augment: bool = True
    use_pretrained_spk: bool = True

    batch_size: int = 64

    max_wav_value: float = 32768.0
    filter_length: int = 1280
    hop_length: int = 320
    win_length: int = 1280
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    sampling_rate: int = 16000

    max_speclen: int = 128


@dataclass
class NetConfig:
    segment_size: int = 8960
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates: List[int] = field(default_factory=lambda: [10, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 256
    ssl_dim: int = 1024
    use_pretrained_spk: bool = DataConfig.use_pretrained_spk


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    lr_decay: float = 0.999875

    c_mel: int = 45
    c_kl: float = 1.0


@dataclass
class Config:
    data: DataConfig = DataConfig()
    net: NetConfig = NetConfig()
    train: TrainConfig = TrainConfig()
