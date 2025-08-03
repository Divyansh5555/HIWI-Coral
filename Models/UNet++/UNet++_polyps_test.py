import os
import json
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------  Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  UNet++ ------------------
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
class CocoSegmentationDataset(Dataset):
    def __init__(self, image_dir, coco_json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(coco_json_path) as f:
            coco = json.load(f)

        self.image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
        self.image_to_annotations = {}

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        self.image_ids = list(self.image_to_annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.image_to_annotations[image_id]:
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float(), filename

# ------------------  Visualization ------------------
def visualize(img, mask, pred, fname, save_dir):
    img = img.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fn = (mask == 1) & (pred == 0)
    fp = (mask == 0) & (pred == 1)
    diff = np.zeros((*mask.shape, 3), dtype=np.uint8)
    diff[fn] = [255, 0, 0]
    diff[fp] = [0, 255, 0]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow((img - img.min()) / (img.max() - img.min()))
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title(" Unet++ Prediction")
    ax[3].imshow(diff)
    ax[3].set_title("Error (FN=Red, FP=Green)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname.replace(".jpg", "_compare.png")))
    plt.close()

# ------------------  Testing ------------------
def test_model(model, dataloader, device, save_dir="unetpp_polypsnew8_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_dice, total_iou, total_time, total_samples = 0, 0, 0, 0

    with torch.no_grad():
        for img, mask, fname in tqdm(dataloader):
            img, mask = img.to(device), mask.to(device)
            start = time.time()
            pred = model(img)
            end = time.time()

            pred_bin = (pred > 0.5).float()

            for i in range(img.size(0)):
                pred_flat = pred_bin[i].view(-1)
                mask_flat = mask[i].view(-1)

                intersection = (pred_flat * mask_flat).sum().item()
                union = pred_flat.sum().item() + mask_flat.sum().item() - intersection

                dice = (2. * intersection) / (pred_flat.sum().item() + mask_flat.sum().item() + 1e-8)
                iou = intersection / (union + 1e-8)

                total_dice += dice
                total_iou += iou
                total_time += (end - start)
                total_samples += 1

                visualize(img[i], mask[i], pred_bin[i], fname[i], save_dir)

    if total_samples == 0:
        print(" No valid samples were processed.")
        return

    print(" Testing Complete")
    print(f" Saved to: {save_dir}")
    print(f" Avg Dice: {total_dice / total_samples:.4f}")
    print(f" Avg IoU : {total_iou / total_samples:.4f}")
    print(f" Avg Time: {total_time / total_samples:.4f}s")

# ------------------  Main ------------------
def main():
    set_seed()

    image_dir = "/home/ddixit/Cutouts/Polyps_dataset/test"
    coco_json_path = "/home/ddixit/Cutouts/Polyps_dataset/test/_annotations.coco.json"

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CocoSegmentationDataset(image_dir, coco_json_path, transform)
    if len(dataset) == 0:
        print("⚠ No test samples found.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = UNetPlusPlus().to(device)
    model.load_state_dict(torch.load("unetpp_polypsnew.pth8", map_location=device))

    test_model(model, loader, device)

if __name__ == "__main__":
    main()
