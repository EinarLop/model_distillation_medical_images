from teacher_setup import get_frozen_teacher
from student_setup import get_student_model
from trainer import train_one_epoch, validate_one_epoch
from distiller import Distiller
from dataset import CheXpertZipDataset, CheXpertDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os


def main():

    if torch.cuda.is_available():
        # For your Linux server
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # For your Mac M1 testing
        device = torch.device("mps")
    else:
        # Fallback
        device = torch.device("cpu")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    dataset_dir = os.environ.get("DATASET_DIR")

    print("Num Workers", num_workers, dataset_dir)

    teacher, img_processor = get_frozen_teacher("codewithdark/vit-chest-xray",
                                                        "./models/teacher_weights", device)
    
    print(f"Number of target classes: {teacher.config.num_labels}")

   # train_dataset_zip = CheXpertZipDataset(
   #     zip_path="/scratch/project_2018357/data/chexpert.zip",
   #     csv_internal_path= "train.csv",
   #     processor= img_processor
   # )


    train_dataset = CheXpertDataset(
        data_root= dataset_dir,
        csv_path=dataset_dir + "/train.csv",
        processor=img_processor
       )

    # Wraps the dataset and handles batching and parallel loading.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=num_workers, # Increase on cluster
        persistent_workers=True, # Keeps workers alive
        prefetch_factor=2
    )

   # val_dataset_zip = CheXpertZipDataset(
   #     zip_path="/scratch/project_2018357/data/chexpert.zip",
   #     csv_internal_path="valid.csv",
   #     processor=img_processor
   #)

    val_dataset = CheXpertDataset(
        data_root= dataset_dir,
        csv_path=dataset_dir + "/valid.csv",
        processor=img_processor
     )

    # Shuffle is False for validation, ensuring consistent evaluation order
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers,persistent_workers=True, # Keeps workers alive
    prefetch_factor=2)

    # Test Batch
    #batch_pixels, batch_labels = next(iter(train_loader))
    # NCHW (Number (Batch Size), Channels, Height, Width).
    #print(f"Test Batch Dataset Pixels Tensor Shape: {batch_pixels.shape}")
    #print(f"Test Batch Dataset Labels Tensor Shape: {batch_labels.shape}")

    # Test Forward Pass
    #batch_pixels = batch_pixels.to(device)
    # Do not calculate gradient
    #with torch.no_grad(): 
    #    teacher_outputs = teacher(batch_pixels)
    #    teacher_logits = teacher_outputs.logits

    #print("Teacher Test Forward Pass")
    #print(f"Teacher Output Logits Shape: {teacher_logits.shape}")

    student = get_student_model("WinKawaks/vit-small-patch16-224", 
                      "./models/student_weights",
                        device)
    
    # Verify the student dimensions
    #dummy_input = torch.randn(8, 3, 224, 224).to(device)
    #outputs = student(dummy_input)

    #print(f"Student Output Logits Shape: {outputs.logits.shape}")

    distiller = Distiller(teacher, student)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)

    os.makedirs("../models/student_checkpoints/", exist_ok=True)

    epochs = 2
    best_auroc = 0.0

    print(f"\nTesting training for {epochs} epochs:")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        avg_loss = train_one_epoch(distiller, train_loader, optimizer, device)
        print(f"Average Training Loss: {avg_loss:.4f}")

        val_auroc = validate_one_epoch(student, val_loader, device)
        print(f"Validation AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            print(f"New best AUROC! Saving model...")
            best_auroc = val_auroc
            
            # Save strictly the weights, not the entire model architecture
            save_path = f"../models/student_checkpoints/best_student_vit.pth"
            torch.save(student.state_dict(), save_path)

if __name__ == "__main__":
    main()
