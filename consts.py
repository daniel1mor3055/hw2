from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelParams:
    im_channels: int = 1
    down_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    mid_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    time_emb_dim: int = 256
    down_sample: List[bool] = field(default_factory=lambda: [True, True, False])
    num_down_layers: int = 2
    num_mid_layers: int = 1
    num_up_layers: int = 2


@dataclass
class DiffusionParams:
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_timesteps_list: List[int] = field(default_factory=lambda: [5, 10, 50, 200])



@dataclass
class TrainParams:
    task_name: str = "default"
    ckpt_name: str = "ddpm_ckpt.pth"
    num_epochs: int = 10
    batch_size: int = 64
    lr: float = 0.001


@dataclass
class SamplingParams:
    sampling_algorithm: str = "vanilla"
    num_samples: int = 100
    sampling_batch_size: int = 10


@dataclass
class WandBParams:
    enable: bool = False
    project_name: str = "ddpm_experiment"


@dataclass
class Config:
    model_params: ModelParams = ModelParams()
    diffusion_params: DiffusionParams = DiffusionParams()
    train_params: TrainParams = TrainParams()
    sampling_params: SamplingParams = SamplingParams()
    wandb: WandBParams = WandBParams()


# Config instance with default values
config = Config()
