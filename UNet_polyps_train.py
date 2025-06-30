import os
import json
import cv2
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------ Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  U-Net Architecture ------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

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

# ------------------  Run One Seed ------------------
def run_experiment(seed):
    set_seed(seed)
    image_dir = "/home/ddixit/Cutouts/Polyps_dataset/train"
    coco_json_path = "/home/ddixit/Cutouts/Polyps_dataset/train_annotations.coco.json"

    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = COCOSegmentationDataset(image_dir, coco_json_path, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    for epoch in range(30):
        loss = train(model, loader, optimizer, criterion, device)
        print(f" Epoch [{epoch+1}/30] - Loss: {loss:.4f}")

    return loss, model.state_dict()

# ------------------  Run All Seeds ------------------
if __name__ == "__main__":
    all_losses = []
    best_loss = float("inf")
    best_model_state = None
    best_seed = None

    for seed in range(10):  # Limit to 10 seeds
        print(f"\n Running experiment for seed {seed}")
        final_loss, model_state = run_experiment(seed)
        all_losses.append(final_loss)
        print(f" Final Loss (Seed {seed}): {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_model_state = model_state
            best_seed = seed

    torch.save(best_model_state, "unet_polyps_seed.pth8")
    print(f"\n Best model (seed {best_seed}) saved to 'unet_polyps_seed.pth8'")

    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    print(f"\n 10-Seed Summary: Mean Final Loss = {mean_loss:.4f} ± {std_loss:.4f}")
