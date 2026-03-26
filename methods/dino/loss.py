import torch
import torch.nn.functional as F


def dino_loss(student_logits, teacher_logits, student_temp=0.1, teacher_temp=0.07, epsilon=1e-8):
    """
    DINO loss - knowledge distillation with centering

    Args:
        student_logits: Student network output
        teacher_logits: Teacher network output
        student_temp: Student temperature
        teacher_temp: Teacher temperature
        epsilon: Small value for numerical stability

    Returns:
        Loss value
    """
    # Get probabilities
    student_probs = F.softmax(student_logits / student_temp, dim=1)
    teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=1)

    # Symmetric KL divergence
    loss = F.kl_div(F.log_softmax(student_logits / student_temp, dim=1), teacher_probs, reduction='batchmean')
    loss += F.kl_div(F.log_softmax(teacher_logits / teacher_temp, dim=1), student_probs, reduction='batchmean')

    return loss / 2
