import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAnswering(nn.Module):
    def __init__(self, hid_dim, num_class, answering_layer_num=1):
        super().__init__()
        self.answering_layer_num = answering_layer_num
        self.num_class = num_class

        self.answering = nn.ModuleList()
        self.bns_answer = nn.ModuleList()

        for i in range(answering_layer_num - 1):
            self.bns_answer.append(nn.LayerNorm(hid_dim))
            self.answering.append(nn.Linear(hid_dim, hid_dim))

        self.bn_hid_answer = nn.LayerNorm(hid_dim)
        self.final_answer = nn.Linear(hid_dim, num_class)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        for i, lin in enumerate(self.answering):
            x = self.bns_answer[i](x)
            x = torch.relu(lin(x))

        x = self.bn_hid_answer(x)
        x = self.final_answer(x)
        return x


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.proj = nn.Linear(in_dim, in_dim)
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        logits = self.scale * torch.matmul(x_norm, w_norm.t())
        return logits


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


__all__ = ["MLPAnswering", "CosineClassifier", "LinearClassifier"]
