import os
import json
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from deeplabv3pp_segmentation import DeepLabV3PlusCustom

# ------------------  Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  Config ------------------
CSV_FILE = '/home/ddixit/Cutouts/coral_dataset/train/annotations.csv'
IMAGE_DIR = '/home/ddixit/Cutouts/coral_dataset/train/images'
IMG_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            try:
                shape = json.loads(annot)
                if shape.get("name") == "polyline" and 'all_points_x' in shape:
                    pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                elif shape.get("name") == "circle" and all(k in shape for k in ['cx', 'cy', 'r']):
                    cv2.circle(mask, (int(shape['cx']), int(shape['cy'])), int(shape['r']), 255, -1)
            except:
                continue

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask.unsqueeze(0).float() / 255.0

# ------------------  Loss Function ------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        eps = 1e-7
        dice = (2. * intersection + eps) / (inputs.sum() + targets.sum() + eps)
        return 1 - dice + self.bce(inputs, targets)

# ------------------  Training Loop for 5 Seeds ------------------
all_ious = []
all_dices = []
best_dice_final_seed = -1

for seed in range(10):
    print(f"\n Seed {seed}")
    set_seed(seed)

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CSVSegmentationDataset(CSV_FILE, IMAGE_DIR, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = DeepLabV3PlusCustom().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    best_dice = -1
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] - Seed {seed}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images))
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                outputs = torch.sigmoid(model(images)).cpu().numpy()
                outputs = (outputs > 0.3).astype(np.uint8)
                all_preds.append(outputs.flatten())
                all_gts.append(masks.numpy().flatten())

        iou = jaccard_score(np.concatenate(all_gts), np.concatenate(all_preds), zero_division=1)
        dice = f1_score(np.concatenate(all_gts), np.concatenate(all_preds), zero_division=1)
        print(f" Validation IoU: {iou:.4f}, Dice: {dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            if seed == 9:
                best_model_state = model.state_dict()

    print(f" Best Dice for seed {seed}: {best_dice:.4f}")
    all_ious.append(iou)
    all_dices.append(dice)

    if seed == 9:
        torch.save(best_model_state, "deeplabv3pp_coral_seed8.pth")
        best_dice_final_seed = best_dice
        print(" Final model saved for seed 9!")

# ------------------  Summary ------------------
mean_iou = np.mean(all_ious)
std_iou = np.std(all_ious)
mean_dice = np.mean(all_dices)
std_dice = np.std(all_dices)

print("\n Summary Over 10 Seeds")
print(f"Avg IoU  : {mean_iou:.4f} ± {std_iou:.4f}")
print(f"Avg Dice : {mean_dice:.4f} ± {std_dice:.4f}")
print(f" Final saved model Dice Score (Seed 4): {best_dice_final_seed:.4f}")
