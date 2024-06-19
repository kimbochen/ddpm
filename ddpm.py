from pathlib import Path
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, FashionMNIST
from tqdm import tqdm

from model import UNet

import numpy as np
from ffcv.fields import RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage, ToDevice, NormalizeImage, RandomHorizontalFlip, Convert
from ffcv.writer import DatasetWriter


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
    ckpt_name: str = 'cifar10'


def train(
    nT, beta_s, beta_e, img_dim, n_channels,
    ds, bs, nw, device, lr, n_epochs, ckpt_name
):
    dataloader = DataLoader(ds, batch_size=bs, num_workers=nw, drop_last=True)
    ddpm = torch.compile(DDPM(nT, beta_s, beta_e, img_dim, n_channels).to(device))
    optimizer = AdamW(ddpm.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    ddpm.train()

    for epoch in range(n_epochs):
        for x0, _ in (pbar := tqdm(dataloader)):
            optimizer.zero_grad()

            x0 = x0.to(device)
            eps = torch.randn_like(x0)
            t = torch.randint(0, nT, [bs], device=device)
            # eps_pred = ddpm(x0, eps, t)
        
            # loss = F.smooth_l1_loss(eps, eps_pred)
            # loss.backward()
            # optimizer.step()

            # pbar.set_description(f'{loss=:.4f}')

            with torch.cuda.amp.autocast():
                eps_pred = ddpm(x0, eps, t)
                loss = F.smooth_l1_loss(eps, eps_pred)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            pbar.set_description(f'loss={loss.item():.4f}')

    torch.save(ddpm.state_dict(), f'./ddpm_{ckpt_name}.pth')


def train_cifar10_ffcv_main():
    if not Path('./cifar10.beton').exists():
	    ds = CIFAR10('./cifar10', train=True, download=True)
	    writer = DatasetWriter('./cifar10.beton', {'image': RGBImageField(max_resolution=32)})
	    writer.from_indexed_dataset(ds)
    cfg_m = ModelConfig()
    cfg_t = TrainerConfig(ckpt_name='cifar10_fp16_ffcv')
    train_cifar10_ffcv(**asdict(cfg_m), **asdict(cfg_t))

def train_cifar10_ffcv(
    nT, beta_s, beta_e, img_dim, n_channels,
    bs, nw, device, lr, n_epochs, ckpt_name
):
    img_tsfms = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(device)),
        ToTorchImage(),
        NormalizeImage(
            mean=np.array([127.5, 127.5, 127.5]), std=np.array([127.5, 127.5, 127.5]),  # [0, 255] -> [-1, 1]
            type=np.float32
        )
    ]
    dataloader = Loader(
        './cifar10.beton', batch_size=bs, num_workers=nw, drop_last=True, os_cache=True,
        order=OrderOption.RANDOM, pipelines={'image': img_tsfms}
    )
    ddpm = torch.compile(DDPM(nT, beta_s, beta_e, img_dim, n_channels).to(device))
    optimizer = AdamW(ddpm.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    ddpm.train()

    for epoch in range(n_epochs):
        for x0, in (pbar := tqdm(dataloader)):
            optimizer.zero_grad()

            eps = torch.randn_like(x0)
            t = torch.randint(0, nT, [bs], device=device)

            with torch.cuda.amp.autocast():
                eps_pred = ddpm(x0, eps, t)
                loss = F.smooth_l1_loss(eps, eps_pred)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            pbar.set_description(f'loss={loss.item():.4f}')

    torch.save(ddpm.state_dict(), f'./ddpm_{ckpt_name}.pth')



def train_cifar10():
    img2tensor = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)  # [0, 1] -> [-1, 1]
    ])
    ds = CIFAR10('./cifar10', train=True, transform=img2tensor, download=True)
    cfg_m = ModelConfig()
    cfg_t = TrainerConfig(ckpt_name='cifar10_fp16')
    train(ds=ds, **asdict(cfg_m), **asdict(cfg_t))


def train_fashion_mnist():
    img2tensor = T.Compose([
        T.Resize(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)
    ])
    ds = FashionMNIST('./mnist', train=True, transform=img2tensor, download=True)
    cfg_m = ModelConfig(n_channels=1)
    cfg_t = TrainerConfig(n_epochs=30, ckpt_name='fashion_mnist_fp16')
    train(ds=ds, **asdict(cfg_m), **asdict(cfg_t))


if __name__ == '__main__':
    torch.manual_seed(3985)
    # train_cifar10_ffcv_main()
    train_fashion_mnist()
