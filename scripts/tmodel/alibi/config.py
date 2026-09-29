from dataclasses import dataclass


@dataclass
class ALiBiConfig:
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    max_len: int = 512
    dropout: float = 0.1
    causal: bool = True
    expansion_factor: int = 1
