import glob
import os
import math
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import Resize

from models import R2AttU_Net, U_Net, AttU_Net, R2U_Net

weight_folder = "./super_resolution_weight_1024"
dataset_dir = "./dataset"

dose = 2e2 # 1e2 0.5e3 1e3 1e4

file_dir = "./val_files/"
file_name = "8rqb_32_32.npy"
seed = 1919810
sr_count = 1024
hr_size = [171, 171]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous().to(device)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)

    mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, pred, label):
        if pred.dim()==3: pred=pred.unsqueeze(1)
        if label.dim()==3: label=label.unsqueeze(1)
        return 1 - ssim(pred, label, self.window_size, self.size_average)

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred, label):
        mse = nn.functional.mse_loss(pred, label)
        if mse==0: return torch.tensor(0.0, device=pred.device)
        psnr = 20*torch.log10(torch.tensor(self.max_val)) - 10*torch.log10(mse)
        return -psnr

class DiffractionDenoiseResidueTransform(torch.nn.Module):
    def __init__(self, dose_range, sigma_range, bias_range):
        super(DiffractionDenoiseResidueTransform, self).__init__()
        self.dose_range = dose_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range

    def forward(self, image_batch):
        dose = torch.randint(
            size=(image_batch.shape[0], 1, 1, 1),
            low=int(self.dose_range[0]),
            high=int(self.dose_range[1]),
            device=image_batch.device
        )
        sigma = torch.rand(
            size=(image_batch.shape[0], 1, 1, 1),
            device=image_batch.device
        ) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        bias = torch.rand(
            size=(image_batch.shape[0], 1, 1, 1),
            device=image_batch.device
        ) * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]

        image_batch = torch.poisson(torch.mul(image_batch, dose))
        noise_batch = torch.add(torch.mul(torch.randn(image_batch.size(), device=image_batch.device), sigma), bias)
        image_batch = image_batch + noise_batch
        return image_batch

transform_size = v2.Compose([
    Resize(size=hr_size, interpolation=InterpolationMode.BILINEAR)
])

transform_dose = v2.Compose([
    DiffractionDenoiseResidueTransform(dose_range=(dose, dose+1), sigma_range=(0, 0), bias_range=(0, 0)),
])


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, device, base_dir, lr_data_prefix="lr_data", hr_data_prefix="ptycho", data_bias=350, data_count=1):
        super().__init__()

        dir_list = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        dir_list = sorted(dir_list)
        self.lr_data_list = []
        self.hr_data_list = []
        self.device = device
        current_data_index = 0
        for dir in dir_list:
            cur_lr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{lr_data_prefix}*.npy")))
            cur_hr_data_path_list = sorted(glob.glob(os.path.join(dir, f"{hr_data_prefix}*.npy")))
            if len(cur_lr_data_path_list) == 0 or len(cur_hr_data_path_list) == 0:
                continue
            cur_hr_data_path = cur_hr_data_path_list[0]
            current_data_index += 1
            if current_data_index < data_bias:
                continue
            if current_data_index > data_bias + data_count:
                break
            for cur_lr_data_path in cur_lr_data_path_list:
                lr_data = np.load(cur_lr_data_path)
                lr_data = torch.Tensor(lr_data)

                _, _, rx, ry = lr_data.shape
                lr_data = lr_data.view(1, -1, rx, ry).squeeze(0)

                lr_data = lr_data[0:sr_count, :, :]
                hr_data = np.load(cur_hr_data_path)
                hr_data = hr_data / np.max(hr_data)
                hr_data = torch.Tensor(hr_data)
                hr_data = hr_data[0:hr_size[0], 0:hr_size[1]]
                self.lr_data_list.append(lr_data.to(self.device))
                self.hr_data_list.append(hr_data.to(self.device))
        
    def __len__(self):
        return len(self.lr_data_list)
    
    def __getitem__(self, index):
        return self.lr_data_list[index], self.hr_data_list[index]
    
class SuperResolutionDataset_LI(torch.utils.data.Dataset):
    def __init__(self, device, lr_path, hr_path):

        super().__init__()
        self.device = device
        self.lr = []
        self.hr = []

        lr_data = np.load(lr_path)           
        lr_data = torch.Tensor(lr_data)
        _, _, rx, ry = lr_data.shape
        lr_data = lr_data.view(1, -1, rx, ry).squeeze(0)

        lr_data = lr_data[0:sr_count, :, :].to(self.device)

        hr_data = np.load(hr_path)

        hr_data = hr_data / hr_data.max()
        hr_data = torch.Tensor(hr_data)
        hr_data = hr_data[0:hr_size[0], 0:hr_size[1]].to(self.device)
        self.lr = lr_data
        self.hr = hr_data

    def __len__(self):
        return 1

    def __getitem__(self):
        return self.lr, self.hr
lr_path = os.path.join(file_dir,file_name)
# test_dataset = SuperResolutionDataset(device, dataset_dir, data_bias=300, data_count=1)
test_dataset = SuperResolutionDataset_LI(device,lr_path=lr_path, hr_path="./val_files/8rqb_172.npy")
if len(test_dataset) == 0:
    print("Failed to load test data. Please check the data path or file name.")
    exit()



lr_sample, hr_sample = test_dataset.lr, test_dataset.hr
lr_sample = lr_sample.unsqueeze(0) 

class SuperResolutionModel(nn.Module):
    def __init__(self, input_channel_count=200, super_resolution_scale=3):
        super(SuperResolutionModel, self).__init__()
        self.layer_model = AttU_Net(img_ch=input_channel_count, output_ch=super_resolution_scale ** 2)
        self.layer_relu = nn.PReLU()
        self.layer_shuffle = nn.PixelShuffle(super_resolution_scale)

    def forward(self, x):
        x = self.layer_model(x)
        x = self.layer_relu(x) 
        x = self.layer_shuffle(x)
        return x

model = SuperResolutionModel(input_channel_count=sr_count, super_resolution_scale=3)
model.to(device)

weight_path = "./super_resolution_weight_1024/weight_v1_MISR_1e2_1e5.pth"
if weight_path:
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Successfully loaded weight:{weight_path}")
else:
    print("The weight file was not found. Please train and save the model first.")
    exit()

model.eval()

with torch.no_grad():
    lr_sample_transformed = transform_dose(lr_sample)
    output = model(lr_sample_transformed)
    # output = transform_size(output)
    output_image = output.squeeze().cpu().numpy()
    hr_sample = hr_sample.unsqueeze(0).unsqueeze(0)  


plt.figure(figsize=(6, 6))
plt.imshow(output_image, cmap='gray')
plt.axis('off')

output_image_file = f"{file_name}_{dose}_32.png"
plt.savefig(output_image_file, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"The test image has been saved as {output_image_file}")

arr_fft = np.abs(np.fft.fftshift(np.fft.fft2(output_image)))
np.save(f"{file_name}_{dose}.npy",output_image)
np.save(f"{file_name}_{dose}_fft.npy", arr_fft)