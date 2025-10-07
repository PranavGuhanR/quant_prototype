import torch
import torch.nn as nn
from typing import List

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1, 1)
        self.linear_1 = nn.Linear(1, 2)
        self.linear_2 = nn.Linear(2, 1, bias=False)
        self.lm_head = nn.Linear(1, 1, bias=False)


class SummaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1, 1)
        self.linear_1 = nn.Linear(10, 20)
        self.linear_2 = nn.Linear(2, 1, bias=False)
        self.lm_head = nn.Linear(1, 1, bias=False)


x = DummyModel()
y = SummaModel()


def replace_linear_with_target(module: nn.Module, target_class: , module_name_to_exclude: List[str]) -> None:
    module_name_to_exclude = set(module_name_to_exclude)
    for name, child in module.named_children():
        old_bias = child.bias
        new_module = target_class(
            child.in_features,
            child.out_features,
            old_bias is not None,
            child.weight.dtype,
        )
        setattr(module, name, new_module)
        if old_bias is not None:
            getattr(module, name).bias = old_bias


replace_linear_with_target(x, y, ["lm_head"])
print(x)
