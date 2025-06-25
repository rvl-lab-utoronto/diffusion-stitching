from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn

from . import BaseNNDiffusion
from ..utils import GroupNorm1d
import matplotlib.pyplot as plt
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
    
class MultBiasLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bias_value = torch.randn((1))
        mult_value = torch.randn((1))
        self.bias_layer = torch.nn.Parameter(bias_value)
        self.mult_layer = torch.nn.Parameter(mult_value)
    
    def forward(self, x):
        return (x*self.mult_layer) + self.bias_layer
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 emb_dim: int,
                 kernel_size: int = 3,
                 linear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = False
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=padding,padding_mode='replicate')
        self.layer_norm_1 = LayerNorm(dim = out_channels) if not linear else nn.Identity()
        self.activation = nn.Mish() if not linear else nn.Identity()
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=padding,padding_mode='replicate')
        self.layer_norm_2 = LayerNorm(dim = out_channels) if not linear else MultBiasLayer()
        #self.last_activation = nn.Mish() if not linear else MultBiasLayer()
        self.last_activation = nn.Mish() if not linear else nn.Identity()
        self.emb_mlp = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, emb):
        out = self.conv1(x) 
        out = self.layer_norm_1(out)
        out = self.activation(out)
        out = out + self.emb_mlp(emb).unsqueeze(-1) # adds condition
        out = self.conv2(out)
        out = self.layer_norm_2(out)
        if self.in_channels == self.out_channels:
            return self.last_activation(out) + x
        else:
            return self.last_activation(out)

class CNN1dShiftEq(BaseNNDiffusion):
    def __init__(
            self,
            in_dim: int,
            emb_dim: int = 32,
            model_dim: int = 32,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None,
            use_timestep_emb: bool = True,
            padding_type = 'zeros',
            n_layers = 25,
            kernel_size = 3, # must be odd
            kernel_expansion_rate = 5,
            encode_position: bool = False # make it NOT positionally equivariant 
            
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.use_timestep_emb = use_timestep_emb
        self.conv_layers = nn.ModuleList([])
        channel_size = 64
        self.channel_size = channel_size
        expansion_counter = 0
        self.encode_position = encode_position
        if encode_position:
            self.conv_layers.append(ResidualBlock(in_channels=in_dim,
                                                out_channels=channel_size,
                                                kernel_size=kernel_size,
                                                emb_dim=emb_dim))
        else:
            self.conv_layers.append(ResidualBlock(in_channels=in_dim,
                                                out_channels=channel_size,
                                                kernel_size=kernel_size,
                                                emb_dim=emb_dim))
        for i in range(n_layers):
            expansion_counter += 1
            in_channels = channel_size #if not encode_position else channel_size + 1
            out_channels = channel_size 
            if expansion_counter >= kernel_expansion_rate:
                # below was channel expansion
                # out_channels = in_channels * 2
                # channel_size = channel_size * 2
                kernel_size += 2
                expansion_counter = 0
            #else:
            #    out_channels = in_channels
            
            self.conv_layers.append(ResidualBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,emb_dim=emb_dim))
        self.conv_layers.append(ResidualBlock(in_channels=out_channels,out_channels=in_dim,kernel_size=kernel_size,emb_dim=emb_dim,linear=True))

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, model_dim * 4), nn.Mish(),
            nn.Linear(model_dim * 4, model_dim))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        # check horizon dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        x = x.permute(0, 2, 1) # batch is now (b,in_dim, horizon)

        emb = self.map_noise(noise)
        if not self.use_timestep_emb: # zeroes out timestep embedding if wanted
            emb = emb * 0 
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb) # why does this exist? legitimately confused
        
        emb = self.map_emb(emb)

        for i, layer in enumerate(self.conv_layers):
            #print(x.shape)
            if self.encode_position and i == 1: # adds positional encoding to the thing
                horizon_length = x.shape[-1]
                #pos_enc = torch.linspace(0,1,horizon_length) # size (horizon)
                #pos_enc = torch.unsqueeze(torch.unsqueeze(pos_enc,0),0).repeat(x.shape[0],1,1) # size (b,1,horizon)
                pos_enc = self.get_sinusoidal_time_embeddings(size=horizon_length,
                                                              length=self.channel_size,
                                                              batch_size=x.shape[0])
                pos_enc = pos_enc.to('cpu') if x.get_device() == -1 else pos_enc.to('cuda') # puts on right device. note - doesn't support weird multi-device cuda shit
                #print(x.shape)
                #print(pos_enc.shape)
                #x = torch.cat((x,pos_enc),dim=1).to(torch.float32) # concatenates - should now be size (b,out_channels,horizon)
                x = x + pos_enc
            #print(x.type())
            x = layer(x,emb)


        x = x.permute(0, 2, 1)
        return x
    def get_sinusoidal_time_embeddings(self,size,length,batch_size,range_param=1000):
        position_enc = np.array([
            [pos / np.power(range_param, 2*i/size) for i in range(size)]
            if pos != 0 else np.zeros(size) for pos in range(length)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        # now position_enc is size (length,size). need to flip
        #position_enc = position_enc.T
        # repeat to right batch size
        position_enc = torch.unsqueeze(torch.tensor(position_enc),0)
        position_enc = position_enc.repeat(batch_size,1,1)
        return position_enc.to(torch.float32)
        #self.sin_time_emb = position_enc
    