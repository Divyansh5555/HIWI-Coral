import os
import json
import cv2
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from statistics import mean, stdev

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF

# ------------------  Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  U-Net++ Architecture ------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = lambda x: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)

        self.conv0_0 = ConvBlock(in_ch, features[0])
        self.conv1_0 = ConvBlock(features[0], features[1])
        self.conv2_0 = ConvBlock(features[1], features[2])
        self.conv3_0 = ConvBlock(features[2], features[3])

        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])

        self.conv0_2 = ConvBlock(features[0]*2 + features[1], features[0])
        self.conv1_2 = ConvBlock(features[1]*2 + features[2], features[1])

        self.conv0_3 = ConvBlock(features[0]*3 + features[1], features[0])

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], 1))

        return torch.sigmoid(self.final(x0_3))

# ------------------  Dataset ------------------
class CSVSegmentationDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.data = self.df.groupby('filename')['region_shape_attributes'].apply(list).reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.loc[idx, 'filename']
        annots = self.data.loc[idx, 'region_shape_attributes']
        img_path = os.path.join(self.image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for annot in annots:
            shape = json.loads(annot)
            if 'all_points_x' in shape and 'all_points_y' in shape:
                pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        return img, mask.unsqueeze(0).float() / 255.0

# ------------------  Loss ------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return 1 - dice + self.bce(inputs, targets)

# ------------------  Training ------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# ------------------  Main ------------------
def main():
    image_dir = "/home/ddixit/Cutouts/coral_dataset/train/images"
    csv_file = "/home/ddixit/Cutouts/coral_dataset/train/annotations.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on: {device}")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    total_seeds = 10
    all_losses = []

    for seed in range(total_seeds):
        print(f"\n Seed {seed}")
        set_seed(seed)

        dataset = CSVSegmentationDataset(csv_file, image_dir, transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = UNetPlusPlus().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = DiceBCELoss()

        for epoch in range(30):
            loss = train(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/30, Loss: {loss:.4f}")

        all_losses.append(loss)

        if seed == total_seeds - 1:
            torch.save(model.state_dict(), "unetpp_coral_seed.pth")
            print(" Final model saved to 'unetpp_coral_seed.pth'")

    print("\n Mean ± Std Loss over 5 Seeds:")
    print("Loss: {:.4f} ± {:.4f}".format(mean(all_losses), stdev(all_losses)))

if __name__ == "__main__":
    main()
