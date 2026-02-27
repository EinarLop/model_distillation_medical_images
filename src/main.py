from teacher_setup import get_frozen_teacher
from dataset import CheXpertZipDataset
import torch
from torch.utils.data import DataLoader


def main():
    teacher, img_processor, device = get_frozen_teacher("codewithdark/vit-chest-xray", "./models/teacher_weights")
    print(f"Number of target classes: {teacher.config.num_labels}")

    train_dataset = CheXpertZipDataset(
        zip_path="data/raw/chexpert.zip",
        csv_internal_path= "train.csv",
        processor= img_processor
    )

    # Wraps the dataset and handles batching and parallel loading.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0 # Increase on cluster
    )

    # Test Batch
    batch_pixels, batch_labels = next(iter(train_loader))
    # NCHW (Number (Batch Size), Channels, Height, Width).
    print(f"Test Batch Pixels Tensor Shape: {batch_pixels.shape}")
    print(f"Test Batch Labels Tensor Shape: {batch_labels.shape}")

    # Test Forward Pass
    batch_pixels = batch_pixels.to(device)
    # Do not calculate gradient
    with torch.no_grad(): 
        teacher_outputs = teacher(batch_pixels)
        teacher_logits = teacher_outputs.logits
        
    print("Test Forward Pass")
    print(f"Teacher Output Logits Shape: {teacher_logits.shape}")

if __name__ == "__main__":
    main()