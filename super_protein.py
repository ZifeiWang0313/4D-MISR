import glob
import random
import shutil
import math
import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch import optim
from torch.nn import PixelShuffle
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import Resize
from denoise_model import R2AttU_Net, U_Net, AttU_Net, R2U_Net

learning_rate = 0.5e-4
weight_decay = 1e-4
batch_size = 32
batch_count_per_feedback = 1
epoch_count = 512
weight_folder = "./super_resolution_weight_1024"
weight_initial = None
current_version = "1"
seed = 1919810
sr_count = 1024
hr_size = [171, 171]

os.makedirs(weight_folder, exist_ok=True)
os.makedirs("./result", exist_ok=True)

device = torch.device("cuda:0")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
poisson_gen = torch.Generator(device=device)
poisson_gen.manual_seed(seed)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device)

def ssim(img1, img2, window_size=6, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)

    mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, pred, label):
        if pred.dim() == 3: pred = pred.unsqueeze(1)
        if label.dim() == 3: label = label.unsqueeze(1)
        return 1 - ssim(pred, label, window_size=self.window_size, size_average=self.size_average)

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, pred, label):
        mse = nn.functional.mse_loss(pred, label)
        if mse == 0:
            return torch.tensor(0.0, device=pred.device)
        psnr = 20 * torch.log10(torch.tensor(self.max_val)) - 10 * torch.log10(mse)
        return -psnr

class DiffractionDenoiseResidueTransform(torch.nn.Module):
    def __init__(self, dose_range, sigma_range, bias_range):
        super().__init__()
        self.dose_range = dose_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range

    def forward(self, image_batch):
        dose = torch.randint(size=(image_batch.shape[0], 1, 1, 1), low=int(self.dose_range[0]), high=int(self.dose_range[1]), device=device)
        sigma = torch.rand(size=(image_batch.shape[0], 1, 1, 1), device=device) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        bias = torch.rand(size=(image_batch.shape[0], 1, 1, 1), device=device) * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]
        image_batch = torch.poisson(torch.mul(image_batch, dose), poisson_gen)
        noise_batch = torch.add(torch.mul(torch.randn(image_batch.size(), device=device), sigma), bias)
        return image_batch + noise_batch

transform_size = v2.Compose([
    v2.Resize(size=hr_size, interpolation=InterpolationMode.BILINEAR)
])

transform_dose = v2.Compose([
    DiffractionDenoiseResidueTransform(dose_range=(1e2, 1e5), sigma_range=(0, 0), bias_range=(0, 0)),
])

class SuperResolutionModel(nn.Module):
    def __init__(self, input_channel_count=200, super_resolution_scale=3):
        super().__init__()
        self.layer_model = AttU_Net(img_ch=input_channel_count, output_ch=super_resolution_scale ** 2)
        self.layer_relu = nn.PReLU()
        self.layer_shuffle = PixelShuffle(super_resolution_scale)

    def forward(self, x):
        x = self.layer_model(x)
        x = self.layer_relu(x)
        x = self.layer_shuffle(x)
        return x

class SuperResolutionDataset(Dataset):
    def __init__(self, device, base_dir, lr_data_prefix="lr_data", hr_data_prefix="ptycho", data_bias=0, data_count=999):
        super().__init__()
        dir_list = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        self.lr_data_list = []
        self.hr_data_list = []
        self.file_prefix_list = []
        self.device = device
        current_data_index = 0
        for dir in dir_list:
            cur_lr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{lr_data_prefix}*.npy")))
            cur_hr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{hr_data_prefix}*.npy")))
            if not cur_lr_data_path_list or not cur_hr_data_path_list:
                continue
            cur_hr_data_path = cur_hr_data_path_list[0]
            current_data_index += 1
            if current_data_index < data_bias: continue
            if current_data_index > data_count + data_bias: break
            for cur_lr_data_path in cur_lr_data_path_list:
                lr_data = torch.Tensor(np.load(cur_lr_data_path)).view(1, -1, *np.load(cur_lr_data_path).shape[-2:]).squeeze(0)[0:sr_count, :, :]
                hr_data = torch.Tensor(np.load(cur_hr_data_path) / np.max(np.load(cur_hr_data_path)))[0:hr_size[0], 0:hr_size[1]]
                self.lr_data_list.append(lr_data.to(self.device))
                self.hr_data_list.append(hr_data.to(self.device))
                self.file_prefix_list.append(os.path.splitext(os.path.basename(cur_lr_data_path))[0])

    def __len__(self):
        return len(self.lr_data_list)

    def __getitem__(self, index):
        return self.lr_data_list[index], self.hr_data_list[index], self.file_prefix_list[index]

train_dataset = SuperResolutionDataset(device, "./dataset", data_bias=0, data_count=400)
test_dataset = SuperResolutionDataset(device, "./dataset", data_bias=400, data_count=50)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SuperResolutionModel(input_channel_count=sr_count, super_resolution_scale=3).to(device)
ssim_loss = SSIMLoss()
psnr_loss = PSNRLoss(max_val=1.0)
psnr_weight = 0.001
optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch_index in range(epoch_count):
    model.train()
    optimizer.zero_grad()
    loss_sum = 0
    for batch, (image, label, _) in enumerate(train_loader):
        image = transform_dose(image)
        pred = transform_size(model(image)).squeeze()
        label = label.squeeze()
        loss = ssim_loss(pred, label) + psnr_weight * psnr_loss(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        current = batch * batch_size + len(image)
        loss_sum += loss.detach()
        if current % (batch_size * batch_count_per_feedback) == 0:
            print(f"TRAIN>> Epoch {epoch_index} loss: {loss_sum / batch_count_per_feedback:.7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")
            loss_sum = 0

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (image, label, file_prefix) in enumerate(test_loader):
            image = transform_dose(image)
            pred = transform_size(model(image)).squeeze()
            label = label.squeeze()
            test_loss += ssim_loss(pred, label) + psnr_weight * psnr_loss(pred, label)

            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label[0].cpu().numpy(), cmap='gray')
            ax[0].set_title("Label")
            ax[1].imshow(pred[0].cpu().detach().numpy(), cmap='gray')
            ax[1].set_title("Prediction")
            plt.tight_layout()
            plt.savefig(f"./result/{epoch_index}_{file_prefix}_image.png", bbox_inches='tight', pad_inches=0)
            plt.close()

    test_loss /= len(test_loader)
    print(f"TEST>> Epoch {epoch_index} loss: {test_loss:.7f}")
    print(f"TEST>> Epoch {epoch_index} ssim:{ssim_loss(pred, label):.7f} psnr:{psnr_loss(pred, label):.7f}")
    torch.save(model.state_dict(), os.path.join(weight_folder, f"weight_v{current_version}_AttU_Net_1e2_1e5_for_protein_read.pth"))
