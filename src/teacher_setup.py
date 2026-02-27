import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def get_frozen_teacher(model_name, cache_folder):
    """
    Loads a Hugging Face ViT model, freezes its weights, and sets it to eval mode
    Args: 
        model_name: from huggingface 
        cache_folder: custom path to save model
    """
    # Handles converting raw PIL images into the tensor format ViT expects
    processor = AutoImageProcessor.from_pretrained(model_name,
                                                   cache_dir=cache_folder)
    
    # Load the Model with Classification Head
    teacher_model = AutoModelForImageClassification.from_pretrained(model_name,
                                                                    cache_dir=cache_folder)
    
    # Freeze the Weights
    # Teacher does not get updated during the distillation loop
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # Set to Evaluation Mode
    # Disables training-specific layers like Dropout
    # Ensures deterministic outputs for the same image
    teacher_model.eval()
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    
    print(f"\nTeacher model successfully loaded and frozen on {device}.")
    
    return teacher_model, processor, device
    
    