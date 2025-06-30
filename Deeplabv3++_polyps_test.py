import os
import cv2
import json
import torch
import random
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from deeplabv3pp_segmentation import DeepLabV3PlusCustom
import matplotlib.pyplot as plt

# ------------------ Seeding ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------ Dataset ------------------
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, coco_json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(coco_json_path) as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            if ann['category_id'] != 1:
                continue
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        img_path = os.path.join(self.image_dir, filename)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.annotations[img_id]:
            for polygon in ann['segmentation']:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0).float() / 255.0, filename

# ------------------ Visualization ------------------
def visualize(image, gt_mask, pred_mask, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    gt = gt_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().numpy()

    fn = (gt > 0.5) & (pred <= 0.5)
    fp = (gt <= 0.5) & (pred > 0.5)

    overlay = (image * 255).astype(np.uint8).copy()
    overlay[fn] = [255, 0, 0]  # Red = FN
    overlay[fp] = [0, 255, 0]  # Green = FP

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Deeplabv3++ Prediction")
    axs[3].imshow(overlay)
    axs[3].set_title("FP=Green | FN=Red")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_output.png"))
    plt.close()

# ------------------ Evaluation ------------------
def evaluate(model, dataloader, device, output_dir="deeplabv3pp_polyps_outputs"):
    model.eval()
    total_dice, total_iou, total_time = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for image, mask, filename in tqdm(dataloader):
            image, mask = image.to(device), mask.to(device)

            start_time = time.time()
            output = model(image)
            end_time = time.time()

            output = torch.sigmoid(output)
            output = TF.center_crop(output, mask.shape[2:])
            pred = (output > 0.5).float()

            pred_flat = pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten()

            intersection = np.sum(pred_flat * mask_flat)
            union = np.sum(pred_flat) + np.sum(mask_flat) - intersection

            dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(mask_flat) + 1e-8)
            iou = intersection / (union + 1e-8)

            total_dice += dice
            total_iou += iou
            total_time += (end_time - start_time)
            total_samples += 1

            visualize(image[0].cpu(), mask[0].cpu(), pred[0].cpu(), filename[0].split('.')[0], output_dir)

    print("\n--- Evaluation Summary ---")
    print(f"Avg Dice : {total_dice / total_samples:.4f}")
    print(f"Avg IoU  : {total_iou / total_samples:.4f}")
    print(f"Avg Time : {total_time / total_samples:.4f} seconds")

# ------------------ Main ------------------
def main():
    set_seed()
    IMAGE_DIR = "/home/ddixit/Cutouts/Polyps_dataset/test"
    COCO_JSON = "/home/ddixit/Cutouts/Polyps_dataset/test_annotations.coco.json"
    MODEL_PATH = "deeplabv3pp_polypsnew_seed.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = COCOSegmentationDataset(IMAGE_DIR, COCO_JSON, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DeepLabV3PlusCustom().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
