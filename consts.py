from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelParams:
    im_channels: int = 1
    im_size: int = 28  # Added based on ground truth
    down_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    mid_channels: List[int] = field(default_factory=lambda: [256, 256, 128])
    time_emb_dim: int = 128
    down_sample: List[bool] = field(default_factory=lambda: [True, True, False])
    num_down_layers: int = 2
    num_mid_layers: int = 2
    num_up_layers: int = 2
    num_heads: int = 4  # Added based on ground truth


@dataclass
class DiffusionParams:
    num_timesteps: int = 200
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_timesteps_list: List[int] = field(
        default_factory=lambda: [5, 10, 50, 200])  # Preserving the original list as a fallback


@dataclass
class TrainParams:
    task_name: str = "default"
    ckpt_name: str = "ddpm_ckpt.pth"
    num_epochs: int = 500
    batch_size: int = 64
    lr: float = 0.0001


@dataclass
class SamplingParams:
    sampling_algorithm: str = "vanilla"
    num_samples: int = 20
    sampling_batch_size: int = 100


@dataclass
class WandBParams:
    enable: bool = False
    project_name: str = "diffusion_model"


@dataclass
class Config:
    model_params: ModelParams = ModelParams()
    diffusion_params: DiffusionParams = DiffusionParams()
    train_params: TrainParams = TrainParams()
    sampling_params: SamplingParams = SamplingParams()
    wandb: WandBParams = WandBParams()


# Config instance with values from ground truth
config = Config()
