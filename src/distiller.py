import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller:
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5):
        """
        Initializes the distillation manager.
        Args:
            teacher_model: The frozen, pre-trained ViT-Base.
            student_model: The ViT-Small being trained.
            temperature: T scales the logits. Higher T -> softer probabilities.
            alpha: Balances the weight between the true labels (hard) and teacher (soft).
        """

        self.teacher = teacher_model
        self.student = student_model
        self.T = temperature
        self.alpha = alpha
        
        # Repeated but kept for safety
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, images, labels):
        """
        Executes the forward pass for both models and computes the combined BCE loss.
        """
        # Teacher Forward Pass (No gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(images).logits
            
        # Student Forward Pass (Gradients)
        student_logits = self.student(images).logits
        
        # Calculate the Hard Loss (Student vs Ground Truth)
        # Cannot use Softmax in either soft or hard loss. 
        # It forces all probabilities to sum to 1 -> patient only has one disease. 
        # In CheXpert, a patient can have both Pneumonia and Cardiomegaly at the same time.
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
        
        # Calculate the Soft Loss (Student vs Teacher)
        # We soften both logits by dividing by the Temperature.
        # We must explicitly apply Sigmoid to the teacher's scaled logits to turn them 
        # into target probabilities between 0 and 1 before calculating the BCE loss
        soft_student_logits = student_logits / self.T
        soft_teacher_probs = torch.sigmoid(teacher_logits / self.T)
        
        soft_loss = F.binary_cross_entropy_with_logits(soft_student_logits, soft_teacher_probs)
        
        # Blend the losses using Alpha
        # Multiply soft_loss by T^2 to scale the gradients back up, as dividing by T shrinks them.
        total_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss * (self.T ** 2))
        
        return total_loss, student_logits