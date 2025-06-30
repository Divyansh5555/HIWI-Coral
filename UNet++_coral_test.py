import os
import json
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# ------------------  Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------  U-Net++ ------------------
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
class CSVSegmentationDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.data = self.df.groupby("filename")["region_shape_attributes"].apply(list).reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.loc[idx, "filename"]
        shape_strs = self.data.loc[idx, "region_shape_attributes"]
        img_path = os.path.join(self.image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for shape_str in shape_strs:
            try:
                shape = json.loads(shape_str)
                if "all_points_x" in shape and "all_points_y" in shape:
                    pts = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            except Exception as e:
                print(f" Error parsing annotation: {e}")

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return img, mask.unsqueeze(0).float(), filename

# ------------------  Metrics ------------------
def dice_score(pred, target, smooth=1e-4):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-4):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# ------------------  Visualization ------------------
def save_individual_visualization(image, gt_mask, pred_mask, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Normalize image for visualization
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-4)

    # Convert masks to NumPy and binarize
    gt = (gt_mask > 0.5).float().cpu().numpy()[0]
    pred = (pred_mask > 0.5).float().cpu().numpy()[0]

    # Prepare GT and prediction masks for grayscale display
    gt_mask_disp = gt * 255
    pred_mask_disp = pred * 255

    # Create overlay image for FN and FP
    overlay = (image * 255).astype(np.uint8).copy()
    fn = (gt > 0.5) & (pred <= 0.5)  # False Negatives (missed prediction)
    fp = (gt <= 0.5) & (pred > 0.5)  # False Positives (wrong prediction)
    overlay[fn] = [255, 0, 0]        # FN = Red
    overlay[fp] = [0, 255, 0]        # FP = Green

    # Plot 4-panel visualization
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))  # Wider figure for better spacing
    axs[0].imshow(image)
    axs[0].set_title("Original Image", fontsize=16)
    axs[1].imshow(gt_mask_disp, cmap='gray')
    axs[1].set_title("Ground Truth", fontsize=16)
    axs[2].imshow(pred_mask_disp, cmap='gray')
    axs[2].set_title("UNet++ Prediction", fontsize=16)
    axs[3].imshow(overlay)
    axs[3].set_title("FP = Green | FN = Red", fontsize=16)

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"vis_{filename}.png"),
        dpi=300,
        bbox_inches="tight"  # Prevent cropping of titles
    )
    plt.close()

# ------------------  Main Testing Function ------------------
def test_model(seed, model_path, csv_path, image_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Testing with seed {seed} | Device: {device}")

    model = UNetPlusPlus().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CSVSegmentationDataset(csv_path, image_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_scores, iou_scores, results = [], [], []
    output_dir = f"unetpp_coral_seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img, mask, fname in tqdm(loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_bin = (pred > 0.5).float()

            dice = dice_score(pred_bin, mask).item()
            iou = iou_score(pred_bin, mask).item()

            results.append({
                "filename": fname[0],
                "dice": round(dice, 4),
                "iou": round(iou, 4)
            })

            dice_scores.append(dice)
            iou_scores.append(iou)
            save_individual_visualization(img[0], mask[0], pred[0], fname[0].replace(".jpg", ""), output_dir)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"\nSeed {seed} - Avg Dice: {np.mean(dice_scores):.4f}, IoU: {np.mean(iou_scores):.4f}")
    print(f" Visualizations + metrics saved to: {output_dir}/")

# ------------------  Entry Point ------------------
if __name__ == "__main__":
    for seed in range(1):  # Run for 10 seeds
        test_model(
            seed=seed,
            model_path="/home/ddixit/Cutouts/unetpp_coral_seed8.pth",
            csv_path="/home/ddixit/Cutouts/coral_dataset/test/annotations.csv",
            image_dir="/home/ddixit/Cutouts/coral_dataset/test/images"
        )
