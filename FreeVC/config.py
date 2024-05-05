from dataclasses import dataclass


@dataclass
class DatasetConfig:
    vctk_dir: str = "./data/VCTK"

    vctk_16k_dir: str = "./data/vctk-16k"
    vctk_22k_dir: str = "./data/vctk-22k"

    split_dir: str = "./data/split"


@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()


_config = Config()


def get_hard_config():
    return _config
