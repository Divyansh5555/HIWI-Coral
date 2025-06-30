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

# ------------------  U-Net Architecture ------------------
class UNetOfficial(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
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

    def forward(self, x):
        x1 = self.down1(x); p1 = self.pool1(x1)
        x2 = self.down2(p1); p2 = self.pool2(x2)
        x3 = self.down3(p2); p3 = self.pool3(x3)
        x4 = self.down4(p3); p4 = self.pool4(x4)
        bn = self.bottleneck(p4)
        up4 = self.upconv4(bn); up4 = self.conv4(torch.cat([x4, up4], dim=1))
        up3 = self.upconv3(up4); up3 = self.conv3(torch.cat([x3, up3], dim=1))
        up2 = self.upconv2(up3); up2 = self.conv2(torch.cat([x2, up2], dim=1))
        up1 = self.upconv1(up2); up1 = self.conv1(torch.cat([x1, up1], dim=1))
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
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for annot in annots:
            shape = json.loads(annot)
            if 'all_points_x' in shape:
                pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        return img, mask.unsqueeze(0).float(), filename

# ------------------  Metrics ------------------
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
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
    axs[2].set_title("U-Net Prediction", fontsize=16)
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

    model = UNetOfficial().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CSVSegmentationDataset(csv_path, image_dir, transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    dice_scores, iou_scores, results = [], [], []
    output_dir = f"unet_coral_seed_{seed}"
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
    print(f"\n Seed {seed} - Avg Dice: {np.mean(dice_scores):.4f}, IoU: {np.mean(iou_scores):.4f}")
    print(f" Visualizations + metrics saved to: {output_dir}/")

# ------------------  Entry Point ------------------
if __name__ == "__main__":
    for seed in range(1):  # Run for 10 seeds
        test_model(
            seed=seed,
            model_path="/home/ddixit/Cutouts/unet_coral_seed.pth8",
            csv_path="/home/ddixit/Cutouts/coral_dataset/test/annotations.csv",
            image_dir="/home/ddixit/Cutouts/coral_dataset/test/images"
        )
