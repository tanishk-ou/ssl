import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import copy


class DINOModel(nn.Module):
    """DINO - Vision Transformer with self-distillation"""

    def __init__(self, embed_dim=768, num_prototypes=65536, student_temp=0.1,
                 teacher_temp_start=0.04, teacher_temp_end=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.student_temp = student_temp
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_end = teacher_temp_end

        # Student network
        vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_hidden_layers=12,
            hidden_size=embed_dim,
            num_attention_heads=12,
            intermediate_size=3072,
            drop_path_rate=0.1
        )
        self.student_encoder = ViTModel(vit_config)
        self.student_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_prototypes)
        )

        # Teacher network (EMA of student)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_head = copy.deepcopy(self.student_head)

        # Freeze teacher
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        self.momentum = 0.999

    def update_teacher(self):
        """Update teacher network with EMA"""
        with torch.no_grad():
            for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
                param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data
            for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data

    def forward(self, x_student, x_teacher=None):
        """Forward pass for student (and optionally teacher)"""
        if x_teacher is None:
            x_teacher = x_student

        # Student forward pass
        student_out = self.student_encoder(pixel_values=x_student, interpolate_pos_encoding=True)
        student_cls = student_out['last_hidden_state'][:, 0, :]
        student_logits = self.student_head(student_cls)

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_out = self.teacher_encoder(pixel_values=x_teacher, interpolate_pos_encoding=True)
            teacher_cls = teacher_out['last_hidden_state'][:, 0, :]
            teacher_logits = self.teacher_head(teacher_cls)

        return student_logits, teacher_logits

    def loss(self, student_logits, teacher_logits, epoch=0, total_epochs=1000):
        """DINO loss"""
        # Temperature schedule for teacher
        teacher_temp = self.teacher_temp_start + \
                      (self.teacher_temp_end - self.teacher_temp_start) * (epoch / total_epochs)

        # Softmax with temperature
        student_probs = F.softmax(student_logits / self.student_temp, dim=1)
        teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=1)

        # Cross-entropy loss
        loss = -(teacher_probs * torch.log(student_probs + 1e-8)).sum(dim=1).mean()
        return loss
