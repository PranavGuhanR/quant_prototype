import torch.nn as nn
from typing import List


def replace_linear_with_target(module: nn.Module, target_class: nn.Module, module_name_to_exclude: List[str]) -> None:
    
    """
    Recursively replaces all child modules of type nn.Linear within a given module with instances
    of a specified target_class, except for the modules whose names are in the exclusion list.

    Args:
        module (nn.Module): The parent module containing submodules to be replaced.
        target_class (nn.Module): The class to replace nn.Linear modules with. It should have the same 
                                  constructor signature as nn.Linear:
                                  (in_features, out_features, bias, dtype).
        module_name_to_exclude (List[str]): List of module names to exclude from replacement.

    Returns:
        None: The function modifies the module in-place.

    Notes:
        - Only direct children of the given module are considered (does not recursively traverse nested children).
        - The bias from the original nn.Linear layer is preserved and reassigned on the new module if it exists.
        - Assumes replaced modules have attributes `in_features`, `out_features`, `bias`, and `weight`.
        - dtype is taken from the original module's weight tensor.
    """

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