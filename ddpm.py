from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ffcv.transforms as T
from ffcv.fields import RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from model import UNet


class DDPM(nn.Module):
    def __init__(self, nT, beta_s, beta_e, img_dim, n_channels):
        super().__init__()
        self.img_dims = (n_channels, img_dim, img_dim)
        self.model = UNet(dim=img_dim, n_channels=n_channels)

        self.nT = nT
        beta = torch.linspace(beta_s, beta_e, nT)  # Linear schedule
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        var_schedule = {
            'sqrt_alpha_bar': alpha_bar.sqrt(),
            'sqrt_one_minus_alpha_bar': torch.sqrt(1.0 - alpha_bar),
            'rsqrt_alpha': alpha.rsqrt(),
            'beta_rsqrt_omab': beta * torch.rsqrt(1.0 - alpha_bar),
            'sigma': beta.sqrt()
        }
        for name, tensor in var_schedule.items():
            self.register_buffer(name, tensor.reshape(-1, 1, 1, 1))

    def forward(self, x0, eps, t):
        x_t = self.sqrt_alpha_bar[t, ...] * x0 + self.sqrt_one_minus_alpha_bar[t, ...] * eps
        eps_pred = self.model(x_t, t)
        return eps_pred

    @torch.inference_mode()
    def sample(self, n_sample, n_steps=None):
        if n_steps is None:
            n_steps = self.nT
        x_t = torch.randn([n_sample, *self.img_dims], device=self.sigma.device)

        for t in reversed(range(n_steps)):
            z = torch.randn_like(x_t) if t > 0 else 0.0
            eps = self.model(x_t, torch.full([n_sample], t, device=x_t.device))
            x_t = self.rsqrt_alpha[t, ...] * (x_t - self.beta_rsqrt_omab[t, ...] * eps) + self.sigma[t, ...] * z

        return x_t


@dataclass
class ModelConfig:
    nT: int = 1000
    beta_s: float = 1e-4
    beta_e: float = 2e-2
    img_dim: int = 32
    n_channels: int = 3


@dataclass
class TrainerConfig:
    device: str = 'cuda'
    bs: int = 3072
    nw: int = 16
    lr: float = 2e-4
    n_epochs: int = 500
    ckpt_name: str = 'cifar10_fp16_ffcv'


def main():
    cfg_m = ModelConfig()
    cfg_t = TrainerConfig()

    ddpm = torch.compile(DDPM(**asdict(cfg_m)).to(cfg_t.device))
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=cfg_t.lr)
    scaler = torch.cuda.amp.GradScaler()

    if not Path('./cifar10.beton').exists():
        ds = CIFAR10('./cifar10', train=True, download=True)
        writer = DatasetWriter('./cifar10.beton', {'image': RGBImageField(max_resolution=32)})
        writer.from_indexed_dataset(ds)
    img_tsfms = [
        SimpleRGBImageDecoder(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.ToDevice(torch.device(cfg_t.device)),
        T.ToTorchImage(),
        T.NormalizeImage(  # [0, 255] -> [-1, 1]
            mean=np.array([127.5, 127.5, 127.5]),
            std=np.array([127.5, 127.5, 127.5]),
            type=np.float32
        )
    ]
    dataloader = Loader(
        './cifar10.beton', batch_size=cfg_t.bs, num_workers=cfg_t.nw, drop_last=True, os_cache=True,
        order=OrderOption.RANDOM, pipelines={'image': img_tsfms}
    )

    for epoch in range(cfg_t.n_epochs):
        ddpm.train()
        for x0, in (pbar := tqdm(dataloader)):
            optimizer.zero_grad()

            eps = torch.randn_like(x0)
            t = torch.randint(0, cfg_m.nT, [cfg_t.bs], device=cfg_t.device)

            with torch.cuda.amp.autocast():
                eps_pred = ddpm(x0, eps, t)
                loss = F.smooth_l1_loss(eps, eps_pred)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f'loss={loss.item():.4f}')

        if epoch == cfg_t.n_epochs - 1 or (epoch + 1) % (cfg_t.n_epochs // 10) == 0:
            ddpm.eval()
            xs = ddpm.sample(16)
            grid = make_grid(xs, nrow=4, normalize=True)
            save_image(grid, f'ddpm_samples/ddpm_sample_epoch{epoch}.png')

    torch.save(ddpm.state_dict(), f'./ddpm_{cfg_t.ckpt_name}.pth')


if __name__ == '__main__':
    torch.manual_seed(3985)
    main()
