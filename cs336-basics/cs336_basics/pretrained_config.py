from dataclasses import dataclass
import torch

@dataclass
class PretrainedConfig():
    # project
    project_name: str
    # data parameter
    vocab_path: str
    merges_path: str
    special_tokens: list[str]
    train_path: str
    valid_path: str

    # model parameter (7.2 TinyStories)
    batch_size: int = 32 # 
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int =  1344
    rope_theta: float = 10000
    num_layers: int = 4  
    num_heads: int = 16
    use_compile: bool = True

    # training parameter (LLaMA: Open and Efficient Foundation Language Model)
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01 # 
    gradient_clipping: float = 1.0
    warmup_steps: int = 4000   # 10% of total_steps
    total_steps: int = 40000

    # logging and checkpoint
    log_freq: int = 100
    eval_freq: int = 1000
    eval_batch: int = 10
    checkpoint_freq: int = 5000
    checkpoint_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
