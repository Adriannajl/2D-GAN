# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, efficientnet_b0, EfficientNet_B0_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image, ImageFile
from tqdm import tqdm
import cv2
import datetime
import time
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from thop import profile

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = {
    "batch_size": 4,
    "image_size": 256,
    "channels": 1,
    "epochs": 200,
    "lr": 2e-5,
    "beta1": 0.9,
    "lambda_mse": 0.1,
    "lambda_vgg": 0.3,
    "lambda_eff": 0.2,
    "lambda_adv": 0.5,
    "lambda_freq": 0.2,
    "lambda_contrast": 0.1,
    "lambda_ssim": 0.2,
    "save_interval": 50,
    "model_dir": "../work_dir/save_models",
    "data_path": "../data/ImageNet",
    "num_workers": 4,
    "perceptual_layers": [0, 3, 6, 9],
    "grad_clip": 0.05,
    "prefetch_factor": None
}

os.makedirs(config['model_dir'], exist_ok=True)


class CrossLevelAttention(nn.Module):
    def __init__(self, in_high, in_low, out_channels):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_high, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_low, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_high, feat_low):
        query = self.query(feat_high)
        key = self.key(feat_low)
        attn = self.sigmoid(query + key)
        weighted = feat_high * attn
        return weighted + feat_low


class EnhancedImageDataset(Dataset):
    def __init__(self, root_dir, subset='train', train_ratio=0.7, val_ratio=0.2):
        self.root_dir = root_dir
        self.subset = subset
        self.image_files = [os.path.join(dp, f) for dp, _, fn in os.walk(root_dir) for f in fn if f.lower().endswith(('.jpg', '.jpeg', '.tif'))]
        total = len(self.image_files)
        indices = torch.randperm(total).tolist()
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        if subset == 'train':
            self.image_files = [self.image_files[i] for i in indices[:train_end]]
        elif subset == 'val':
            self.image_files = [self.image_files[i] for i in indices[train_end:val_end]]
        elif subset == 'test':
            self.image_files = [self.image_files[i] for i in indices[val_end:]]
        else:
            raise ValueError("subset must be 'train', 'val', or 'test'")

        self.transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.RandomCrop(config['image_size']),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.tensor(exposure.equalize_adapthist(x.numpy(), clip_limit=0.03))),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            path = self.image_files[idx]
            with Image.open(path) as img:
                img = img.convert('L')
                tensor = self.transform(img)
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    tensor = self.repair_image(tensor)
                if torch.rand(1) > 0.5:
                    tensor += torch.randn_like(tensor) * 0.05
                return tensor.clamp(-1, 1), tensor.clamp(-1, 1)
        except:
            dummy = torch.zeros(1, config['image_size'], config['image_size'])
            return dummy, dummy

    def repair_image(self, tensor):
        arr = tensor.numpy()
        arr[np.isnan(arr) | np.isinf(arr)] = 0
        arr = (arr * 255).astype(np.uint8)
        repaired = cv2.inpaint(arr, (arr == 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        return torch.tensor(repaired / 255.0).float()


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:16]
        self.model = nn.Sequential(*list(vgg.children()))
        for p in self.model.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        return self.criterion(self.model(x), self.model(y))


class EfficientNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList([nn.Sequential(*list(base.children())[:i+1]) for i in config['perceptual_layers']])
        for p in self.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()
        self.weights = [0.1, 0.3, 0.5, 1.0]

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        loss = 0.0
        for i, layer in enumerate(self.layers):
            loss += self.criterion(layer(x), layer(y)) * self.weights[i]
        return loss / len(self.layers)


class EnhancedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Generator structure: encoder-decoder network with multiple modules including cross-level attention
        pass

    def forward(self, x):
        # Forward pass of generator network
        pass


class EnhancedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Discriminator structure: patch-based Markov discriminator with multiple convolutional layers
        pass

    def forward(self, x):
        # Forward pass of discriminator network
        pass


def compute_gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = EnhancedGenerator().to(device)
    discriminator = EnhancedDiscriminator().to(device)
    vgg_loss = VGGLoss().to(device)
    eff_loss = EfficientNetLoss().to(device)
    optimizer_G = optim.AdamW(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=config['lr'] * 0.5, betas=(config['beta1'], 0.999))
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=5)
    scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=5)

    train_loader = DataLoader(EnhancedImageDataset(config['data_path'], 'train'),
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True,
                              prefetch_factor=config['prefetch_factor'])

    for epoch in range(config['epochs']):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []
        for real_imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            real_imgs = real_imgs.to(device)
            optimizer_D.zero_grad()
            fake_imgs = generator(real_imgs)
            real_pred = discriminator(real_imgs)
            fake_pred = discriminator(fake_imgs.detach())
            d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + 10 * compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            gen_imgs = generator(real_imgs)
            adv_loss = -torch.mean(discriminator(gen_imgs))
            mse = nn.MSELoss()(gen_imgs, real_imgs)
            vgg = vgg_loss(gen_imgs, real_imgs)
            eff = eff_loss(gen_imgs, real_imgs)
            g_total = (config['lambda_adv'] * adv_loss + config['lambda_mse'] * mse +
                       config['lambda_vgg'] * vgg + config['lambda_eff'] * eff)
            g_total.backward()
            optimizer_G.step()

            g_losses.append(g_total.item())
            d_losses.append(d_loss.item())

        scheduler_G.step(np.mean(g_losses))
        scheduler_D.step(np.mean(d_losses))

        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(generator.state_dict(), os.path.join(config['model_dir'], f'gen_epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    train_model()
