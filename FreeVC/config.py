from dataclasses import dataclass


@dataclass
class DataConfig:
    vctk_dir: str = "./data/VCTK"

    vctk_16k_dir: str = "./data/vctk-16k"
    vctk_22k_dir: str = "./data/vctk-22k"

    split_dir: str = "./data/split"

    use_sr_augment: bool = True

    batch_size: int = 32
