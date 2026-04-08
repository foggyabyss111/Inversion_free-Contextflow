import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from modules.model import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention, WanAttentionBlock

def register_time(model, t):
    layers_to_modify = [0, 1, 2, 3, 4]
    
    for i, block in enumerate(model.blocks):
        if i in layers_to_modify and hasattr(block, 'self_attn') and isinstance(block.self_attn, WanSelfAttention):
            setattr(block.self_attn, "t", t)

def register_self_attention_pnp(model, injection_schedule):
    layers_to_modify = [0, 1, 2, 3, 4]
    
    for i, block in enumerate(model.blocks):
        if i in layers_to_modify and hasattr(block, 'self_attn') and isinstance(block.self_attn, WanSelfAttention):
            setattr(block.self_attn, "injection_schedule", injection_schedule)
