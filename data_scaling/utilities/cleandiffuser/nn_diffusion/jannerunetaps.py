from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_diffusion import BaseNNDiffusion
from ..utils import GroupNorm1d



class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


    
### all the wacky APS shit starts after this 


class circular_pad(nn.Module):
    def __init__(self, padding = (1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding
        
    def forward(self, x):
            
        return F.pad(x, pad = self.pad_sizes , mode = 'circular')
def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding_mode = 'circular'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, padding_mode = 'circular', filter_size = 3, aps_criterion = 'l2'):
        super().__init__()

        self.maxpool = nn.Sequential(
            #get_pad_layer_1d(padding_mode)((0, 1, 0, 1)),
            get_pad_layer_1d(padding_mode)((0, 1)),
            nn.MaxPool1d(kernel_size = 2, stride = 1),
            ApsDown(channels = in_channels, filt_size = filter_size, stride = 2, 
                apspool_criterion = aps_criterion, pad_type = padding_mode)
            )

        
        self.double_conv = DoubleConv(in_channels, out_channels, padding_mode = padding_mode)

    def forward(self, x):
        down_out, polyphase_comp = self.maxpool(x)
        out = self.double_conv(down_out)
        
        return out, polyphase_comp


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, padding_mode = 'circular', filter_size = 3):
        super().__init__()
        
        self.padding_mode = padding_mode

        if bilinear:
            raise Exception('Implementation with bilinear mode currently not supported.')
            
            
        else:
            
            #replace conv transpose 2d with APS_up+circular_conv with kernel size 2
                
            self.up = nn.Sequential(ApsUp(channels = in_channels, filt_size = filter_size, stride = 2, pad_type = padding_mode),
                                    circular_pad((0, 1, 0, 1)),
                                    nn.Conv1d(in_channels , in_channels // 2, kernel_size=2, stride = 1))
                                    
            self.conv = DoubleConv(in_channels, out_channels, padding_mode = padding_mode)
            


    def forward(self, x1, x2, polyphase_indices):
        
        x1 = self.up({'inp': x1, 'polyphase_indices': polyphase_indices})
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ApsDown(nn.Module):
    def __init__(self, channels, pad_type='circular', filt_size=3, stride=2, apspool_criterion = 'l2'):
        super(ApsDown, self).__init__()
        self.filt_size = filt_size
        #self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        
        self.apspool_criterion = apspool_criterion
        
        a = construct_1d_array(self.filt_size)

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        #self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.register_buffer('filt', filt[None,:,:].repeat((self.channels,1,1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

        
    def forward(self, input_to_pool):
        
        if isinstance(input_to_pool, dict):
            inp, polyphase_indices = input_to_pool['inp'], input_to_pool['polyphase_indices']
    
        else:
#             this is the case when polyphase indices are not pre-defined
            inp = input_to_pool
            polyphase_indices = None

        down_func = aps_downsample_direct

        if(self.filt_size==1):
            return down_func(aps_pad(inp), self.stride, polyphase_indices, apspool_criterion = self.apspool_criterion)
            
        else:
            #print(self.filt.shape)
            blurred_inp = F.conv1d(self.pad(inp), self.filt, stride = 1, groups=inp.shape[1])
            return down_func(aps_pad(blurred_inp), self.stride, polyphase_indices, apspool_criterion = self.apspool_criterion)
        
        
        
class ApsUp(nn.Module):
    def __init__(self, channels, pad_type='circular', filt_size=3, stride=2, apspool_criterion = 'l2'):
        super(ApsUp, self).__init__()
        
        self.filt_size = filt_size
        #self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        
        self.apspool_criterion = apspool_criterion
        
        a = construct_1d_array(self.filt_size)

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        #self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        #print(filt.shape)
        self.register_buffer('filt', filt[None,:,:].repeat((self.channels,1,1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)
        
    def forward(self, inp_to_unpool):
        
        inp = inp_to_unpool['inp']
        polyphase_indices = inp_to_unpool['polyphase_indices']

        if inp.shape[2] == inp.shape[3]:
            up_func = aps_upsample

        else:
            up_func = aps_upsample_direct

        
        if(self.filt_size==1):
            return up_func(aps_pad(inp), self.stride, polyphase_indices)
        
        else:
            
            aps_up = up_func(aps_pad(inp), self.stride, polyphase_indices)
            return F.conv1d(self.pad(aps_up), self.filt, stride = 1, groups = inp.shape[1])
            
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)                    

        
        
        

def construct_1d_array(filt_size):
    
    if(filt_size==1):
        a = np.array([1.,])
    elif(filt_size==2):
        a = np.array([1., 1.])
    elif(filt_size==3):
        a = np.array([1., 2., 1.])
    elif(filt_size==4):    
        a = np.array([1., 3., 3., 1.])
    elif(filt_size==5):    
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size==6):    
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size==7):    
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
        
    return a


def aps_downsample_direct(x, stride, polyphase_indices = None, apspool_criterion = 'l2'):
    # NOTE - not sure if this adaptation is correct 
    if stride==1:
        return x

    elif stride>2:
        raise Exception('Stride>2 currently not supported in this implementation')

    else:

        #xpoly_0 = x[:, :, ::stride, ::stride]
        #xpoly_1 = x[:, :, 1::stride, ::stride]
        #xpoly_2 = x[:, :, ::stride, 1::stride]
        #xpoly_3 = x[:, :, 1::stride, 1::stride]
        xpoly_0 = x[:, :, ::stride]
        xpoly_1 = x[:, :, 1::stride]

        #xpoly_combined = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim = 1)
        xpoly_combined = torch.stack([xpoly_0, xpoly_1], dim = 1)

        if polyphase_indices is None:

            polyphase_indices = get_polyphase_indices_from_xpoly(xpoly_combined, apspool_criterion)

        B = xpoly_combined.shape[0]
        #output = xpoly_combined[torch.arange(B), polyphase_indices, :, :, :]
        output = xpoly_combined[torch.arange(B), polyphase_indices, :, :]
        
        return output, polyphase_indices


def get_polyphase_indices_from_xpoly(xpoly_combined, apspool_criterion):

    B = xpoly_combined.shape[0]

    if apspool_criterion == 'l2':
        norm_ind = 2

    elif apspool_criterion == 'l1':
        norm_ind = 1
    else:
        raise ValueError('Unknown criterion choice')


    #all_norms = torch.norm(xpoly_combined.view(B, 4, -1), dim = 2, p = norm_ind)
    all_norms = torch.norm(xpoly_combined.view(B, 3, -1), dim = 2, p = norm_ind) # idk

    return torch.argmax(all_norms, dim = 1)



    


def get_polyphase_indices_v2(x, apspool_criterion):
#     x has the form (B, 4, C, N_poly) where N_poly corresponds to the reduced version of the 2d feature maps

    if apspool_criterion == 'l2':
        norms = torch.norm(x, dim = (2, 3), p = 2)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l1':
        norms = torch.norm(x, dim = (2, 3), p = 1)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l_infty':
        B = x.shape[0]
        max_vals = torch.max(x.reshape(B, 4, -1).abs(), dim = 2).values
        polyphase_indices = torch.argmax(max_vals, dim = 1)
        
        
    elif apspool_criterion == 'l2_min':
        norms = torch.norm(x, dim = (2, 3), p = 2)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    elif apspool_criterion == 'l1_min':
        norms = torch.norm(x, dim = (2, 3), p = 1)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    else:
        raise Exception('Unknown APS criterion')
        
    return polyphase_indices


def permute_polyphase(N, stride = 2):
    
    base_even_ind = 2*torch.arange(int(N/2))[None, :]
    base_odd_ind = 1 + 2*torch.arange(int(N/2))[None, :]
    
    even_increment = 2*N*torch.arange(int(N/2))[:,None]
    odd_increment = N + 2*N*torch.arange(int(N/2))[:,None]
    
    p0_indices = (base_even_ind + even_increment).view(-1)
    p1_indices = (base_even_ind + odd_increment).view(-1)
    
    p2_indices = (base_odd_ind + even_increment).view(-1)
    p3_indices = (base_odd_ind + odd_increment).view(-1)
    
    permute_indices = torch.cat([p0_indices, p1_indices, p2_indices, p3_indices], dim = 0)
    
    return permute_indices





def aps_upsample(x, stride, polyphase_indices):
    #somewhat inefficient but working version. uses stride 2
    
    if stride ==1:
        return x
    
    elif stride>2:
        raise Exception('Currently only stride 2 supported')
        
    else:
    
        B, C, Nb2, _ = x.shape

        y = torch.zeros(B, 4, C, Nb2**2).to(x.dtype).cuda()
        y[torch.arange(B), polyphase_indices, :, :] = x.view(B, C, Nb2**2)

        y = y.permute(0, 2, 1, 3).reshape(B, C, 4*Nb2**2)

        permute_indices = permute_polyphase(2*Nb2)

        y0 = y.clone()
        y0[:, :, permute_indices] = y

        y0 = y0.view(B, C, 2*Nb2, 2*Nb2)

        return y0


def aps_upsample_direct(x, stride, polyphase_indices):
    #inefficient but working version. uses stride 2
    
    if stride ==1:
        return x
    
    elif stride>2:
        raise Exception('Currently only stride 2 supported')
        
    else:
    
        B, C, N1, N2 = x.shape

        y = torch.zeros(B, 4, C,  N1, N2).to(x.dtype).cuda()
        y1 = torch.zeros(B, C,  2*N1, 2*N2).to(x.dtype).cuda()
        
        y[torch.arange(B), polyphase_indices, :, :, :] = x

        y1[:, :, ::2, ::2] = y[:, 0, :, :, :]
        y1[:, :, 1::2, ::2] = y[:, 1, :, :, :]
        y1[:, :, ::2, 1::2] = y[:, 2, :, :, :]
        y1[:, :, 1::2, 1::2] = y[:, 3, :, :, :]

        return y1





def aps_pad(x):
    
    N1, N2 = x.shape[2:4]
    
    if N1%2==0 and N2%2==0:
        return x
    
    if N1%2!=0:
        x = F.pad(x, (0, 0, 0, 1), mode = 'circular')
    
    if N2%2!=0:
        x = F.pad(x, (0, 1, 0, 0), mode = 'circular')
    
    return x
        



class JannerUNet1dAps(BaseNNDiffusion):
    def __init__(
            self,
            in_dim: int,
            model_dim: int = 32,
            emb_dim: int = 32,
            dim_mult: List[int] = [1, 2, 2, 2],
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None,
            aps_criterion = 'l2',
            filter_size = 3,
            bilinear = False,
            padding_mode = 'zero'
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        factor = 2 # no fucking clue what this is but we need it 

        dims = [in_dim] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # initial down conv?

        # down convs
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if ind == 0:
                self.downs.append(DoubleConv(dim_in, dim_out, padding_mode = padding_mode))
            else:
                self.downs.append(
                    Down(dim_in, dim_out, padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion))

        # up convs
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                Up(dim_out, dim_in // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size))

        self.outc = OutConv(dims[1], in_dim) # might not have input dimension right - double chekc 
        

        # all this stuff is how they originally implemented it 
        """ self.inc = DoubleConv(in_dim, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down3 = Down(inner_channels_list[2], inner_channels_list[3] // factor, padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)

        self.up1 = Up(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up2 = Up(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up3 = Up(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.outc = OutConv(inner_channels_list[0], out_channels) """

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

        x = x.permute(0, 2, 1) # to be shape batch size,  # channels, conv. dimension (time)
        # might need to add support for conditions later... whatever

        residuals = []
        polys = []

        for i,layer in enumerate(self.downs):
            #print(i)
            print('X shape at Layer',i,':',x.shape)
            if i == 0:
                x = layer(x)
                residuals.append(x)
            else:
                x, poly = layer(x)
                residuals.append(x)
                polys.append(poly)
        residuals.pop()
        for i, layer in enumerate(self.ups):
            #print(i)
            x = layer(x,residuals.pop(),polys.pop())

        x = self.outc(x)

        x = x.permute(0, 2, 1)
        return x
