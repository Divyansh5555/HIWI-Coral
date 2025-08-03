import os
import json
import cv2
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from statistics import mean, stdev

# ------------------  Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  U-Net Architecture ------------------
class UNetOfficial(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def crop_and_concat(self, upsampled, bypass):
        _, _, h, w = upsampled.size()
        bypass_cropped = TF.center_crop(bypass, [h, w])
        return torch.cat((bypass_cropped, upsampled), dim=1)

    def forward(self, x):
        x1 = self.down1(x); p1 = self.pool1(x1)
        x2 = self.down2(p1); p2 = self.pool2(x2)
        x3 = self.down3(p2); p3 = self.pool3(x3)
        x4 = self.down4(p3); p4 = self.pool4(x4)
        bn = self.bottleneck(p4)
        up4 = self.upconv4(bn); up4 = self.conv4(self.crop_and_concat(up4, x4))
        up3 = self.upconv3(up4); up3 = self.conv3(self.crop_and_concat(up3, x3))
        up2 = self.upconv2(up3); up2 = self.conv2(self.crop_and_concat(up2, x2))
        up1 = self.upconv1(up2); up1 = self.conv1(self.crop_and_concat(up1, x1))
        return torch.sigmoid(self.final(up1))

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
            if 'all_points_x' in shape:
                pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask.unsqueeze(0).float() / 255.0

# ------------------  Loss ------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + 1)/(inputs.sum() + targets.sum() + 1)
        return 1 - dice + self.bce(inputs, targets)

# ------------------  Training ------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        _, _, h, w = preds.shape
        diff_y = masks.size(2) - h
        diff_x = masks.size(3) - w
        masks = masks[:, :, diff_y // 2 : diff_y // 2 + h, diff_x // 2 : diff_x // 2 + w]
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

        model = UNetOfficial(in_ch=3, out_ch=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = DiceBCELoss()

        for epoch in range(30):
            loss = train(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/30, Loss: {loss:.4f}")

        all_losses.append(loss)

        if seed == total_seeds - 1:
            torch.save(model.state_dict(), "unet_coral_seed.pth8")
            print(" Final model weights saved as 'unet_coral_seed.pth8'")

    print("\n Mean ± Std Loss over 5 Seeds:")
    print("Loss: {:.4f} ± {:.4f}".format(mean(all_losses), stdev(all_losses)))

if __name__ == "__main__":
    main()
