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
from torch.utils.data import random_split
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import Resize
from torchvision.models import vgg19
from pytorch_msssim import MS_SSIM
from models import R2AttU_Net, U_Net, AttU_Net, R2U_Net

learning_rate = 0.5e-4
weight_decay = 1e-4
batch_size = 32
batch_count_per_feedback = 1
epoch_count = 768
weight_folder = "./super_resolution_weight_1024"
weight_initial = None
current_version = "3"
seed = 1919810
sr_count = 1024
hr_size = [171, 171]

os.makedirs(weight_folder, exist_ok=True)

device = torch.device("cuda:0")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
poisson_gen = torch.Generator(device=device)
poisson_gen.manual_seed(seed)
    
class DiffractionDenoiseResidueTransform(torch.nn.Module):
    def __init__(self, dose_range, sigma_range, bias_range):
        super(DiffractionDenoiseResidueTransform, self).__init__()
        self.dose_range = dose_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range

    def forward(self, image_batch):
        dose = torch.randint(size=(image_batch.shape[0], 1, 1, 1),
                             low=int(self.dose_range[0]),
                             high=int(self.dose_range[1]),
                             device=device)
        sigma = torch.rand(size=(image_batch.shape[0], 1, 1, 1),
                           device=device) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        bias = torch.rand(size=(image_batch.shape[0], 1, 1, 1),
                          device=device) * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]
        image_batch = torch.poisson(image_batch * dose, generator=poisson_gen)
        noise_batch = torch.randn(image_batch.size(), device=device) * sigma + bias
        image_batch = image_batch + noise_batch
        image_batch = image_batch.clamp(min=0.0)
        max_val = image_batch.view(image_batch.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        image_batch = image_batch / (max_val + 1e-6)
        return image_batch

transform_size = v2.Compose([
    v2.Resize(size=hr_size, interpolation=InterpolationMode.BILINEAR,antialias=True)
])

transform_dose = v2.Compose([
    DiffractionDenoiseResidueTransform(dose_range=(1e2, 1e5), sigma_range=(0, 0), bias_range=(0, 0)),
])

class SuperResolutionModel(nn.Module):
    def __init__(self, input_channel_count=200, super_resolution_scale=3):
        super(SuperResolutionModel, self).__init__()
        self.layer_model = AttU_Net(img_ch=input_channel_count, output_ch=super_resolution_scale ** 2)
        self.layer_relu = nn.PReLU()
        self.layer_shuffle = PixelShuffle(super_resolution_scale)
        self.conv_smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer_model(x)
        x = self.layer_relu(x)
        x = self.layer_shuffle(x)
        x = self.conv_smooth(x)
        return x

class SuperResolutionDataset(Dataset):
    def __init__(self, device, base_dir, lr_data_prefix="lr_data", hr_data_prefix="ptycho", data_bias=0, data_count=999):
        super().__init__()
        dir_list = os.listdir(base_dir)
        dir_list = [os.path.join(base_dir, d) for d in dir_list if os.path.isdir(os.path.join(base_dir, d))]
        dir_list = sorted(dir_list)
        self.lr_data_list = []
        self.hr_data_list = []
        self.device = device
        current_data_index = 0
        for dir in dir_list:
            print(f">> Preindexing {dir}")
            cur_lr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{lr_data_prefix}*.npy")))
            cur_hr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{hr_data_prefix}*.npy")))
            if len(cur_lr_data_path_list) == 0 or len(cur_hr_data_path_list) == 0:
                continue
            cur_hr_data_path = cur_hr_data_path_list[0]
            current_data_index += 1
            if current_data_index < data_bias:
                continue
            if current_data_index > data_count + data_bias:
                break
            for cur_lr_data_path in cur_lr_data_path_list:
                lr_data = np.load(cur_lr_data_path)
                lr_data = torch.Tensor(lr_data)
                _, _, rx, ry = lr_data.shape
                lr_data = lr_data.view(1, -1, rx, ry).squeeze(0)[0:sr_count, :, :]
                hr_data = np.load(cur_hr_data_path)
                hr_data = hr_data / np.max(hr_data)
                hr_data = torch.Tensor(hr_data)[0:hr_size[0], 0:hr_size[1]]
                self.lr_data_list.append(lr_data.to(self.device))
                self.hr_data_list.append(hr_data.to(self.device))

    def __len__(self):
        return len(self.lr_data_list)

    def __getitem__(self, index):
        lr_data = self.lr_data_list[index]
        hr_data = self.hr_data_list[index]
        return lr_data, hr_data

train_dataset = SuperResolutionDataset(device, "./dataset", data_bias=0, data_count=400)
test_dataset = SuperResolutionDataset(device, "./dataset", data_bias=400, data_count=50)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = SuperResolutionModel(input_channel_count=sr_count, super_resolution_scale=3)
model.to(device)

class PerceptualSSIMLoss(torch.nn.Module):
    def __init__(self, device):
        super(PerceptualSSIMLoss, self).__init__()

        vgg_features = vgg19(pretrained=True).features
        self.vgg_layers = torch.nn.Sequential()
        for i, layer in enumerate(vgg_features):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.vgg_layers.add_module(f"maxpool_{i}", torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            else:
                self.vgg_layers.add_module(f"layer_{i}", layer)
            if i >= 19:
                break

        self.vgg_layers.to(device).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.pixel_loss_fn = torch.nn.L1Loss()
        self.perceptual_loss_fn = torch.nn.MSELoss()
        self.ssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, label):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if label.dim() == 3:
            label = label.unsqueeze(1)

        pixel_loss = self.pixel_loss_fn(pred, label)
        
        pred_vgg = pred.repeat(1, 3, 1, 1)
        label_vgg = label.repeat(1, 3, 1, 1)

        percep_loss = self.perceptual_loss_fn(self.vgg_layers(pred_vgg), self.vgg_layers(label_vgg))

        ssim_val = self.ssim_loss_fn(pred, label)
        ssim_loss = 1 - ssim_val

        return pixel_loss + 0.006 * percep_loss + ssim_loss

loss_fn = PerceptualSSIMLoss(device)
optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch_index in range(epoch_count):
    model.train()
    optimizer.zero_grad()
    size = len(train_loader.dataset)
    loss_sum = 0
    for batch, (image, label) in enumerate(train_loader):
        image = transform_dose(image)
        pred = model(image)
        pred = transform_size(pred).squeeze()
        label = label.squeeze()
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(label[0, :, :].squeeze().cpu().numpy(), cmap='gray')
        ax[1].imshow(pred[0, :, :].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.show()

        current = batch * batch_size + len(image)
        loss_sum += loss.detach()
        if current % (batch_size * batch_count_per_feedback) == 0:
            print(f"TRAIN>> Epoch {epoch_index} loss: {loss_sum / batch_count_per_feedback:.7f} [{current:>5d}/{size:>5d}]")
            loss_sum = 0

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (image, label) in enumerate(test_loader):
            image = transform_dose(image)
            pred = model(image)
            pred = transform_size(pred).squeeze()
            label = label.squeeze()
            test_loss += loss_fn(pred, label)
            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label[0, :, :].squeeze().cpu().numpy(), cmap='gray')
            ax[1].imshow(pred[0, :, :].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.show()

    test_loss = test_loss / len(test_loader)
    print(f"TEST>> Epoch {epoch_index} loss: {test_loss:.7f}")
    torch.save(model.state_dict(), os.path.join(weight_folder, f"weight_v{current_version}_MISR_1e2_1e5.pth"))
