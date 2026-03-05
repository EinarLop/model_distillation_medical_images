import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def train_one_epoch(distiller, dataloader, optimizer, device):
    """
    Iterates through the dataset once, calculating loss and updating student weights.
    Args: 
        distiller: distiller class instance
        dataloader: pytorch dataloader
        optimizer: training optimizer
        device: device
    """
    
    distiller.student.train()
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss, _ = distiller.compute_loss(images, labels)
        loss.backward()
        optimizer.step()
        
        # .item() extracts the float value from the tensor. 
        # This is critical for preventing memory leaks
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_one_epoch(student_model, dataloader, device):
    """
    Evaluates the student model on the validation set and calculates the macro AUROC.
     Args: 
        student_model: model
        dataloader: pytorch dataloader
        device: device
    """
    # Set to Evaluation Mode
    # Disables Dropout and locks BatchNorm layers so predictions are deterministic.
    student_model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Disable Gradient Tracking
    # Saves memory and speeds up inference since we are not updating weights here.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get the raw student logits
            logits = student_model(images).logits
            
            # Apply Sigmoid to convert raw logits into probabilities (0.0 to 1.0)
            probs = torch.sigmoid(logits)
            
            # Move data back to CPU and convert to numpy for scikit-learn
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    # Concatenate all batches into single arrays
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate AUROC
    # We use 'macro' average to calculate the metric independently for each of the 5 classes
    # and then find their unweighted mean.
    try:
        auroc_score = roc_auc_score(all_targets, all_predictions, average='macro')
    except ValueError:
        # This prevents a crash if your test set is so small that a class has only 0s or only 1s
        print("Warning: AUROC calculation failed (likely due to missing positive labels in test batch).")
        auroc_score = 0.0
        
    return auroc_score