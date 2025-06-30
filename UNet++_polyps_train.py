import os
import json
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------  Set Seeding ------------------
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
        self.conv0_1 = ConvBlock(features[0]+features[1], features[0])
        self.conv1_1 = ConvBlock(features[1]+features[2], features[1])
        self.conv2_1 = ConvBlock(features[2]+features[3], features[2])
        self.conv0_2 = ConvBlock(features[0]*2+features[1], features[0])
        self.conv1_2 = ConvBlock(features[1]*2+features[2], features[1])
        self.conv0_3 = ConvBlock(features[0]*3+features[1], features[0])
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
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, coco_json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.image_sizes = {img['file_name']: (img['height'], img['width']) for img in coco_data['images']}

        self.annotations_per_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_per_image:
                self.annotations_per_image[img_id] = []
            self.annotations_per_image[img_id].append(ann)

        self.image_ids = list(self.annotations_per_image.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        filename = self.image_id_to_filename[img_id]
        height, width = self.image_sizes[filename]

        img_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.annotations_per_image[img_id]:
            if ann.get("segmentation"):
                for seg in ann["segmentation"]:
                    if len(seg) < 6:
                        continue
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float() / 255.0

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
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ------------------  Main ------------------
def main():
    image_dir = "/home/ddixit/Cutouts/Polyps_dataset/train"
    coco_json_path = "/home/ddixit/Cutouts/Polyps_dataset/train_annotations.coco.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    final_losses = []

    for seed in range(10):
        print(f"\n Seed {seed}")
        set_seed(seed)

        transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

        dataset = COCOSegmentationDataset(image_dir, coco_json_path, transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

        model = UNetPlusPlus().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = DiceBCELoss()

        for epoch in range(30):
            loss = train(model, loader, optimizer, criterion, device)
            print(f" Epoch [{epoch+1}/30] - Loss: {loss:.4f}")

        final_losses.append(loss)

        if seed == 9:
            torch.save(model.state_dict(), "unetpp_polypsnew.pth8")
            print(" Final model saved: unetpp_polypsnew.pth8")

    # ------------------  Summary ------------------
    mean_loss = np.mean(final_losses)
    std_loss = np.std(final_losses)
    print(f"\n Final Loss Summary (5 Seeds): {mean_loss:.4f} ± {std_loss:.4f}")

if __name__ == "__main__":
    main()
