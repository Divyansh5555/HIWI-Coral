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

# ------------------ UNet (Original) ------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        # Up path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.double_conv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return torch.sigmoid(self.final_conv(x))

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

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
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = (img * 255).astype(np.uint8).copy()

    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fn = (mask == 1) & (pred == 0)
    fp = (mask == 0) & (pred == 1)

    overlay = img_rgb.copy()
    overlay[fn] = [255, 0, 0]   # Red for False Negatives
    overlay[fp] = [0, 255, 0]   # Green for False Positives

    fig, ax = plt.subplots(1, 4, figsize=(30, 10))  # Wider for spacing
    ax[0].imshow(img)
    ax[0].set_title("Original Image", fontsize=16)
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth", fontsize=16)
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("UNet Prediction", fontsize=16)
    ax[3].imshow(overlay)
    ax[3].set_title("(FN=Red, FP=Green)", fontsize=16)

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname.replace(".jpg", "_compare.png"))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High resolution & no cropping
    plt.close()
# ------------------  Testing ------------------


def test_model(model, dataloader, device, save_dir="unet_polypsnew1_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_dice, total_iou, total_time = 0, 0, 0
    all_preds, all_targets = [], []

    total_samples = 0

    with torch.no_grad():
        for img, mask, fname in tqdm(dataloader):
            img, mask = img.to(device), mask.to(device)
            start = time.time()
            pred = model(img)
            end = time.time()

            pred_bin = (pred > 0.5).float()

            for i in range(img.size(0)):
                pred_flat = pred_bin[i].view(-1).cpu().numpy()
                mask_flat = mask[i].view(-1).cpu().numpy()

                intersection = np.sum(pred_flat * mask_flat)
                union = np.sum(pred_flat) + np.sum(mask_flat) - intersection

                dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(mask_flat) + 1e-8)
                iou = intersection / (union + 1e-8)

                total_dice += dice
                total_iou += iou
                total_time += (end - start)
                total_samples += 1

                # Save for metric computation
                all_preds.extend(pred_flat)
                all_targets.extend(mask_flat)

                visualize(img[i], mask[i], pred_bin[i], fname[i], save_dir)

    if total_samples == 0:
        print("⚠ No valid samples were processed.")
        return

    # Compute classification metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)

    print(" Testing Complete")
    print(f" Saved to: {save_dir}")
    print(f" Avg Dice: {total_dice / total_samples:.4f}")
    print(f" Avg IoU : {total_iou / total_samples:.4f}")
    print(f" Avg Time: {total_time / total_samples:.4f}s")

# ------------------  Main ------------------
def main():
    set_seed()

    image_dir = "/home/ddixit/Cutouts/Polyps_dataset/test"
    coco_json_path = "/home/ddixit/Cutouts/Polyps_dataset/test_annotations.coco.json"

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CocoSegmentationDataset(image_dir, coco_json_path, transform)
    if len(dataset) == 0:
        print(" No test samples found.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_polyps_seed.pth8", map_location=device))

    test_model(model, loader, device)

if __name__ == "__main__":
    main()
