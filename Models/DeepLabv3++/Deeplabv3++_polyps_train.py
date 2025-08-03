import os
import cv2
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from deeplabv3pp_segmentation import DeepLabV3PlusCustom

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

# ------------------ Evaluation ------------------
def evaluate(model, dataloader, device):
    model.eval()
    total_dice, total_iou, total_time = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for image, mask, filename in tqdm(dataloader):
            image, mask = image.to(device), mask.to(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(image)
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end) / 1000.0 if device.type == 'cuda' else 0.0

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
            total_time += inference_time
            total_samples += 1

    print("\n--- Evaluation Summary ---")
    print(f"Avg Dice : {total_dice / total_samples:.4f}")
    print(f"Avg IoU  : {total_iou / total_samples:.4f}")
    print(f"Avg Time : {total_time / total_samples:.4f} seconds")

# ------------------ Main ------------------
def main():
    set_seed()
    IMAGE_DIR = "/home/ddixit/Cutouts/Polyps_dataset/test"
    COCO_JSON = "/home/ddixit/Cutouts/Polyps_dataset/test_annotations.coco.json"
    BATCH_SIZE = 2
    IMG_SIZE = 256
    MODEL_PATH = "deeplabv3pp_polypsnew_seed.pth"
    device = torch.device("cpu")  # Force CPU to avoid NCCL/CUDA errors

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = COCOSegmentationDataset(IMAGE_DIR, COCO_JSON, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = DeepLabV3PlusCustom().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
