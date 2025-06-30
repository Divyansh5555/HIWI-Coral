import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, f1_score
from torchvision.transforms import functional as TF
from deeplabv3pp_segmentation import DeepLabV3PlusCustom

# -----------------------------
# Dataset for Testing
# -----------------------------
class CSVTestSegmentationDataset(Dataset):
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

        if not os.path.exists(img_path):
            print(f" Skipping missing image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.imread(img_path)
        if img is None:
            print(f" OpenCV failed to read image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

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
                    cx, cy, r = int(shape["cx"]), int(shape["cy"]), int(shape["r"])
                    cv2.circle(mask, (cx, cy), r, 255, -1)
            except Exception as e:
                print(f" Annotation error in {filename}: {e}")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask.unsqueeze(0).float() / 255.0, filename

# -----------------------------
# Metrics
# -----------------------------
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# -----------------------------
# Visualization
# -----------------------------
def save_visualization(img_tensor, gt_mask, pred_mask, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    gt_mask = gt_mask.squeeze().cpu().numpy() * 255
    pred_mask = pred_mask.squeeze().cpu().numpy() * 255

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[1].imshow(gt_mask, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title(" Deeplabv3++ Prediction")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_{filename.replace('/', '_')}.png"))
    plt.close()

# -----------------------------
# Test Function
# -----------------------------
def test_deeplabv3pp(csv_file, image_dir, model_path, output_dir, img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusCustom().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CSVTestSegmentationDataset(csv_file, image_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_scores, iou_scores, filenames = [], [], []

    with torch.no_grad():
        for img, mask, fname in tqdm(loader):
            img, mask = img.to(device), mask.to(device)

            pred = torch.sigmoid(model(img))
            _, _, H, W = pred.shape
            mask = TF.center_crop(mask, [H, W])
            pred = TF.center_crop(pred, [H, W])

            pred_bin = (pred > 0.5).float()
            dice = dice_score(pred_bin, mask).item()
            iou = iou_score(pred_bin, mask).item()

            dice_scores.append(dice)
            iou_scores.append(iou)
            filenames.append(fname[0])

            save_visualization(img[0].cpu(), mask[0].cpu(), pred_bin[0].cpu(), fname[0], output_dir)

    df = pd.DataFrame({
        'filename': filenames,
        'dice_score': dice_scores,
        'iou_score': iou_scores
    })
    df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    print("\n Avg Dice:", round(np.mean(dice_scores), 4))
    print(" Avg IoU :", round(np.mean(iou_scores), 4))
    print(" Results saved to:", output_dir)

# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    test_deeplabv3pp(
        csv_file='/home/ddixit/Cutouts/coral_dataset/train/annotations.csv',
        image_dir='/home/ddixit/Cutouts/coral_dataset/train/images',
        model_path='/home/ddixit/Cutouts/Deeplabv3++/deeplabv3pp_coral_seed8.pth',
        output_dir='deeplabv3pp_coral_test_results'
    )
