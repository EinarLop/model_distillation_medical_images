from teacher_setup import get_frozen_teacher
from student_setup import get_student_model
from trainer import train_one_epoch, validate_one_epoch
from distiller import Distiller
from dataset import CheXpertDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4), 8)
    dataset_dir = os.environ.get("DATASET_DIR", "/Users/einar/Documents/EDISS/UIB/TFM/model_distillation_medical_images/data")
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    root_path =  os.environ.get("ROOT_PATH", ".")

    print("Num Workers", num_workers, dataset_dir)

    teacher, img_processor = get_frozen_teacher("codewithdark/vit-chest-xray",
                                                root_path + "/models/teacher_weights",
                                                device)

    train_dataset = CheXpertDataset(
        data_root= dataset_dir,
        csv_path=dataset_dir + "/train.csv",
        processor=img_processor
       )

    # Wraps the dataset and handles batching and parallel loading.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, # Increase on cluster
        persistent_workers=True, # Keeps workers alive
        prefetch_factor=2
    )

    val_dataset = CheXpertDataset(
        data_root= dataset_dir,
        csv_path=dataset_dir + "/valid.csv",
        processor=img_processor
     )

    # Shuffle is False for validation, ensuring consistent evaluation order
    val_loader = DataLoader(val_dataset,
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            persistent_workers=True, # Keeps workers alive
            prefetch_factor=2)


    # NCHW (Number (Batch Size), Channels, Height, Width).
    

    student = get_student_model("WinKawaks/vit-small-patch16-224", 
                                root_path +"/models/student_weights",
                                device)


    distiller = Distiller(teacher, student)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)

    epochs = 10 

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,                  # The peak learning rate
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3                 # Spends the first 30% of training warming up
    )
   
    best_auroc = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"distillation_{timestamp}"
    log_dir = root_path + f"/runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        avg_loss = train_one_epoch(distiller, train_loader, optimizer, device, writer, global_step, scheduler)
        print(f"Average Training Loss: {avg_loss:.4f}")
        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)

        val_macro_auroc, val_class_auroc = validate_one_epoch(student, val_loader, device)
        print(f"Validation AUROC: {val_macro_auroc:.4f}")
        writer.add_scalar('Validation/Epoch_AUROC', val_macro_auroc, epoch)

        disease_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']
        for i, disease in enumerate(disease_names):
            writer.add_scalar(f'Validation/Epoch_Disease_AUROC-{disease}', val_class_auroc[i], epoch)

        if val_macro_auroc > best_auroc:
            print(f"New best AUROC! Saving model...")
            best_auroc = val_macro_auroc
            
            # Save weights
            save_path = f"models/student_checkpoints/best_student_vit.pth"
            torch.save(student.state_dict(), save_path)

if __name__ == "__main__":
    main()
