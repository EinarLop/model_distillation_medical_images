import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

def get_student_model(model_name, cache_folder, device):
    """
    Loads a pre-trained ViT-Small from Hugging Face, discards its original 1000-class head, 
    and randomly initializes a 5-class head for the CheXpert targets.
    Args: 
        model_name: from huggingface 
        cache_folder: custom path to save model
    """
    
    # By passing num_labels=5 and ignore_mismatched_sizes=True, 
    # the library automatically drops the old ImageNet classifier layer 
    # and attaches a new nn.Linear(384, 5) layer
    student_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=5,
        ignore_mismatched_sizes=True,
        cache_dir= cache_folder
    )
    
    # Ensure gradients are calculated for the student (non-frozen weights) 
    for param in student_model.parameters():
        param.requires_grad = True
        
    # Move to GPU
    student_model.to(device)

    print(f"\nStudent model successfully loaded on {device}.")

    return student_model

