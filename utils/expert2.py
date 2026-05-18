import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vmamba import SS2D


# class SegmentationExpert(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(SegmentationExpert, self).__init__()
#         self.convs = nn.Conv2d(input_channels, 64, 1)
#         self.conv1 = nn.Conv2d(64, output_channels, 1)
#
#     def forward(self, x):
#         x1 = self.convs(x)
#         x2 = self.conv1(x1)
#         return x2

def choquet_integral(features, weights):
    """
    计算Choquet积分
    """
    sorted_indices = torch.argsort(features, dim=1, descending=True)
    cum_weights = torch.cumsum(weights[sorted_indices], dim=1)
    integral = torch.sum(features * cum_weights, dim=1)
    return integral


# class MixtureOfExperts(nn.Module):
#     def __init__(self, num_experts, input_channels, output_channels, uncertainty_threshold=0.5):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList([SegmentationExpert(input_channels, output_channels) for _ in range(num_experts)])
#         self.uncertainty_threshold = uncertainty_threshold
#
#     def forward(self, x):
#         expert_outputs = [expert(x) for expert in self.experts]
#         expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_channels, H, W)
#
#         # 计算每个专家输出的概率分布并计算不确定性
#         probabilities = F.softmax(expert_outputs, dim=2)
#         num_classes = probabilities.size(2)
#         features = probabilities.view(-1, num_classes)
#         weights = torch.linspace(0, 1, num_classes, device=probabilities.device)
#         integral = choquet_integral(features, weights)
#
#         # 归一化积分
#         integral = (integral - integral.min()) / (integral.max() - integral.min())
#
#         # 将积分重塑为 (batch_size, num_experts, H, W)
#         uncertainty_map = integral.view(x.size(0), self.num_experts, x.size(2), x.size(3))
#
#         # 根据不确定性阈值生成掩码
#         uncertainty_mask = (uncertainty_map > self.uncertainty_threshold).float()
#
#         # 使用路由机制进行加权混合
#         weights = F.softmax(uncertainty_mask, dim=1)  # 在专家维度进行 softmax
#         mixed_output = torch.sum(expert_outputs * weights.unsqueeze(2), dim=1)
#
#         return mixed_output,expert_outputs

class SegmentationExpert(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegmentationExpert, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, input_channels, output_channels, uncertainty_threshold=0.5):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([SegmentationExpert(input_channels, output_channels) for _ in range(num_experts)])
        self.uncertainty_threshold = uncertainty_threshold

        # Initialize mixture weights
        self.mixture_weights = nn.Parameter(torch.rand(num_experts))
        self.softmax = nn.Softmax(dim=0)

        # Gate mechanism parameters
        self.gate_fc = nn.Linear(input_channels*224*224, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        input_features = x.view(batch_size, -1)

        # Gate mechanism
        gate_logits = self.gate_fc(input_features)
        gate_values = self.sigmoid(gate_logits)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_channels, H, W)

        # Apply gate values to expert outputs
        expert_outputs = expert_outputs * gate_values.view(batch_size, self.num_experts, 1, 1, 1)

        # Compute mixture weights
        mixture_weights = self.softmax(self.mixture_weights)

        # Compute final output
        final_output = torch.sum(mixture_weights.view(1, self.num_experts, 1, 1, 1) * expert_outputs, dim=1)

        return final_output, expert_outputs



# # 示例：创建混合专家模型并进行前向传播
# num_experts = 3
# input_channels = 3
# output_channels = 1
# model = MixtureOfExperts(num_experts, input_channels, output_channels, uncertainty_threshold=0.5)
# input_data = torch.randn(1, input_channels, 64, 64)  # 示例输入数据
# print(input_data.shape)
# output = model(input_data)
# print(output.shape)


