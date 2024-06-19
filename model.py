import math
import torch
import torch.nn.functional as F
from torch import nn


class WSConv2d(nn.Conv2d):
    '''
    Weight-Standardized Convolution
    https://arxiv.org/abs/1903.10520
    '''
    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, x):
        mean = self.weight.mean(dim=1, keepdim=True)
        var = self.weight.var(dim=1, correction=0, keepdim=True)
        norm_weight = (self.weight - mean) * torch.rsqrt(var + self.eps)
        out = F.conv2d(
            x, norm_weight,
            self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return out


class TimeResNetBlock(nn.Module):
    '''
    B: Batch size
    D: in_dim
    E: out_dim
    F: out_dim * 2
    G: E + F = out_dim * 3
    '''
    def __init__(self, in_dim, out_dim, t_dim):
        super().__init__()

        self.proj_t = nn.Linear(t_dim, out_dim*2)
        self.conv1 = nn.Sequential(
            WSConv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_dim)
        )
        self.conv2 = nn.Sequential(
            WSConv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_dim)
        )
        if in_dim != out_dim:
            self.rconv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.rconv = nn.Identity()

    def forward(self, x_BDHW, t_embd_BT):
        t_embd_BF = self.proj_t(t_embd_BT)
        scale_BE, shift_BE = t_embd_BF.reshape(*t_embd_BF.shape, 1, 1).chunk(2, dim=1)
        h_BEHW = self.conv1(x_BDHW)
        h_BEHW = F.silu(h_BEHW * (scale_BE + 1) + shift_BE)
        h_BEHW = self.conv2(h_BEHW)
        x_BEHW = h_BEHW + self.rconv(x_BDHW)
        return x_BEHW


class Attention(nn.Module):
    def __init__(self, d_embd, n_heads=4, d_head=32):
        super().__init__()
        d_hid = n_heads * d_head

        self.n_heads = n_heads
        self.d_head = d_head
        self.attn_proj = nn.Conv2d(d_embd, d_hid*3, kernel_size=1, bias=False)
        self.scale = d_head ** -0.5
        self.out_proj = nn.Conv2d(d_hid, d_embd, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.size()
        qkv_BCHW = self.attn_proj(x).chunk(3, dim=1)
        to_attn_head = lambda z: z.reshape(B, self.n_heads, self.d_head, H, W).flatten(-2).transpose(-2, -1).contiguous()
        q_BNED, k_BNED, v_BNED = map(to_attn_head, qkv_BCHW)

        # attn_BNEE = (q_BNED @ k_BNED.transpose(-2, -1)) * self.scale
        # score_BNEE = F.softmax(attn_BNEE, dim=-1)
        # y_BNED = score_BNEE @ v_BNED
        y_BNED = F.scaled_dot_product_attention(q_BNED, k_BNED, v_BNED, dropout_p=0.0, is_causal=False)

        y_BCHW = y_BNED.transpose(-2, -1).flatten(1, 2).reshape(B, -1, H, W)
        out_BCHW = self.out_proj(y_BCHW)

        return out_BCHW



class UNetDownsample(nn.Module):
    '''
    B: Batch size
    D: in_dim
    E: out_dim
    T: t_dim
    H, W: Last 2 dimensions of x
    Ho, Wo: (H, W) if is_last else (H // 2, W // 2)
    '''
    def __init__(self, in_dim, out_dim, t_dim, is_last=False):
        super().__init__()
        self.block1 = TimeResNetBlock(in_dim, in_dim, t_dim)
        self.block2 = TimeResNetBlock(in_dim, in_dim, t_dim)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_dim)
        self.attn = Attention(in_dim)

        if is_last:
            self.dsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.dsample = DownsampleOutProject(in_dim, out_dim)

    def forward(self, x_BDHW, t_embd_BT):
        fmap1_BDHW = self.block1(x_BDHW, t_embd_BT)
        fmap2_BDHW = self.block2(fmap1_BDHW, t_embd_BT)
        x_BDHW = self.attn(self.norm(fmap2_BDHW)) + fmap2_BDHW
        x_BDHoWo = self.dsample(x_BDHW)
        return x_BDHoWo, fmap1_BDHW, fmap2_BDHW

class DownsampleOutProject(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_proj = nn.Conv2d(4*in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.reshape(B, 4*C, H//2, W//2)
        x = self.out_proj(x)
        return x


class UNetUpsample(nn.Module):
    '''
    B: Batch size
    D: in_dim
    E: out_dim
    F: in_dim + out_dim
    T: t_dim
    H, W: Last 2 dimensions of x
    Ho, Wo: (H, W) if is_last else (H * 2, W * 2)
    '''
    def __init__(self, in_dim, out_dim, t_dim, is_last=False):
        super().__init__()
        self.block1 = TimeResNetBlock(in_dim+out_dim, out_dim, t_dim)
        self.block2 = TimeResNetBlock(in_dim+out_dim, out_dim, t_dim)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_dim)
        self.attn = Attention(out_dim)

        if is_last:
            self.usample = nn.Conv2d(out_dim, in_dim, kernel_size=3, padding=1)
        else:
            self.usample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_dim, in_dim, kernel_size=3, padding=1)
            )

    def forward(self, x_BEHW, fmap1_BDHW, fmap2_BDHW, t_embd_BT):
        x_BFHW = torch.cat([x_BEHW, fmap1_BDHW], dim=1)
        x_BEHW = self.block1(x_BFHW, t_embd_BT)

        x_BFHW = torch.cat([x_BEHW, fmap2_BDHW], dim=1)
        x_BEHW = self.block2(x_BFHW, t_embd_BT)

        x_BEHW = self.attn(self.norm(x_BEHW)) + x_BEHW
        x_BDHoWo = self.usample(x_BEHW)

        return x_BDHoWo


class UNetBlock(nn.Module):
    '''
    B: Batch size
    D: dim
    H, W: Last 2 dimensions of x
    T: t_dim
    '''
    def __init__(self, dim, t_dim):
        super().__init__()
        self.block1 = TimeResNetBlock(dim, dim, t_dim)
        self.block2 = TimeResNetBlock(dim, dim, t_dim)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.attn = Attention(dim)

    def forward(self, x_BDHW, t_embd_BT):
        x_BDHW = self.block1(x_BDHW, t_embd_BT)
        x_BDHW = self.block2(x_BDHW, t_embd_BT)
        x_BDHW = self.attn(self.norm(x_BDHW)) + x_BDHW
        return x_BDHW


class UNet(nn.Module):
    '''
    D: dim
    C: n_channels
    H, W: Last 2 dimensions of x
    T: t_dim
    F: dim // 2

    D0 = D
    H0 = H // 2
    W0 = W // 2
    
    D1 = D * 2
    H1 = H // 4
    W1 = W // 4
    
    D2 = D * 4
    H2 = H // 8
    W2 = W // 8
    
    D3 = D * 8

    G = D * 2
    '''
    def __init__(self, dim, n_channels):
        super().__init__()

        self.in_conv = nn.Conv2d(n_channels, dim, kernel_size=1, padding=0)

        t_dim = dim * 4
        amp = math.log(1e4) / (dim // 2 - 1)
        self.register_buffer(
            'freqs_F', torch.exp(torch.arange(dim//2) * -amp)
        )
        self.proj_t = nn.Sequential(
            nn.Linear(dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim)
        )

        dim0 = dim
        dim1 = dim * 2
        dim2 = dim * 4
        dim3 = dim * 8

        self.dsample0 = UNetDownsample(dim0, dim0, t_dim)
        self.dsample1 = UNetDownsample(dim0, dim1, t_dim)
        self.dsample2 = UNetDownsample(dim1, dim2, t_dim)
        self.dsample3 = UNetDownsample(dim2, dim3, t_dim, is_last=True)

        self.mblock = UNetBlock(dim3, t_dim)
        
        self.usample3 = UNetUpsample(dim2, dim3, t_dim)
        self.usample2 = UNetUpsample(dim1, dim2, t_dim)
        self.usample1 = UNetUpsample(dim0, dim1, t_dim)
        self.usample0 = UNetUpsample(dim0, dim0, t_dim, is_last=True)

        self.out_resblk = TimeResNetBlock(2*dim, dim, t_dim)
        self.out_conv = nn.Conv2d(dim, n_channels, kernel_size=1)

    def forward(self, x_BCHW, t_B):
        pos_embd_BF = t_B.unsqueeze(1) * self.freqs_F.unsqueeze(0)
        t_embd_BD = torch.cat([pos_embd_BF.sin(), pos_embd_BF.cos()], dim=-1)
        t_embd_BT = self.proj_t(t_embd_BD)
    
        x_BDHW = self.in_conv(x_BCHW)
        r_BDHW = x_BDHW.clone()

        x_BD0H0W0, fmap2_BD0HW  , fmap1_BD0HW   = self.dsample0(x_BDHW   , t_embd_BT)
        x_BD1H1W1, fmap2_BD0H0W0, fmap1_BD0H0W0 = self.dsample1(x_BD0H0W0, t_embd_BT)
        x_BD2H2W2, fmap2_BD1H1W1, fmap1_BD1H1W1 = self.dsample2(x_BD1H1W1, t_embd_BT)
        x_BD3H2W2, fmap2_BD2H2W2, fmap1_BD2H2W2 = self.dsample3(x_BD2H2W2, t_embd_BT)
        x_BD3H2W2 = self.mblock(x_BD3H2W2, t_embd_BT)
        x_BD2H1W1 = self.usample3(x_BD3H2W2, fmap1_BD2H2W2, fmap2_BD2H2W2, t_embd_BT)
        x_BD1H0W0 = self.usample2(x_BD2H1W1, fmap1_BD1H1W1, fmap2_BD1H1W1, t_embd_BT)
        x_BD0HW   = self.usample1(x_BD1H0W0, fmap1_BD0H0W0, fmap2_BD0H0W0, t_embd_BT)
        x_BDHW    = self.usample0(x_BD0HW  , fmap1_BD0HW  , fmap2_BD0HW  , t_embd_BT)

        x_BGHW = torch.cat([x_BDHW, r_BDHW], dim=1)
        x_BCHW = self.out_conv(self.out_resblk(x_BGHW, t_embd_BT))

        return x_BCHW


if __name__ == '__main__':
    B = 2
    C = 3
    H = 128
    W = 128
    D = 32

    torch.manual_seed(3985)
    x_BCHW = torch.rand([2, C, H, W], device='cuda')
    t_B = torch.randint(0, 1000, [B], device=x_BCHW.device)
    model = UNet(dim=D, n_channels=C).to(x_BCHW.device)

    shape = model(x_BCHW, t_B).size()
    shape_gt = (B, C, H, W)
    assert shape == shape_gt, f'Expected {shape_gt}, got {shape}'
