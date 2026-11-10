"""C2 and C3 definitions from the study notebook."""

import torch
from torch import nn
from torchvision.models import resnet18


class DirectionalDepthwiseBranch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        (kernel_height, kernel_width) = kernel_size
        padding = (kernel_height // 2, kernel_width // 2)
        self.branch = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.GELU(), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        return self.branch(x)


class MultiScaleDirectionalResidualBlock(nn.Module):

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError('channels must be divisible by four')
        branch_channels = channels // 4
        self.branch_1x7 = DirectionalDepthwiseBranch(in_channels=channels, out_channels=branch_channels, kernel_size=(1, 7))
        self.branch_1x11 = DirectionalDepthwiseBranch(in_channels=channels, out_channels=branch_channels, kernel_size=(1, 11))
        self.branch_3x7 = DirectionalDepthwiseBranch(in_channels=channels, out_channels=branch_channels, kernel_size=(3, 7))
        self.branch_5x9 = DirectionalDepthwiseBranch(in_channels=channels, out_channels=branch_channels, kernel_size=(5, 9))
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.GELU(), nn.Dropout2d(p=dropout))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.output_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_features = torch.cat([self.branch_1x7(x), self.branch_1x11(x), self.branch_3x7(x), self.branch_5x9(x)], dim=1)
        directional_feature = self.fusion(branch_features)
        output = x + self.residual_scale * directional_feature
        return self.output_activation(output)


class C2DirectionalResNet18Top96Fusion(nn.Module):

    def __init__(self, attribute_dim=96, image_dropout=0.3, attribute_dropout=0.2, fusion_dropout=0.35, direction_dropout=0.1):
        super().__init__()
        self.backbone = resnet18(weights=None)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.direction_block1 = MultiScaleDirectionalResidualBlock(channels=64, dropout=direction_dropout)
        self.direction_block2 = MultiScaleDirectionalResidualBlock(channels=128, dropout=direction_dropout)
        self.image_projection = nn.Sequential(nn.Linear(backbone_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(image_dropout))
        self.attribute_encoder = nn.Sequential(nn.Linear(attribute_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(attribute_dropout), nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU())
        self.gate_network = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(256 * 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(fusion_dropout), nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(fusion_dropout), nn.Linear(64, 1))

    def extract_image_feature(self, image):
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.direction_block1(x)
        x = self.backbone.layer2(x)
        x = self.direction_block2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, image, attributes, return_gate=False, return_embedding=False):
        raw_image_feature = self.extract_image_feature(image)
        image_embedding = self.image_projection(raw_image_feature)
        attribute_embedding = self.attribute_encoder(attributes)
        joint_embedding = torch.cat([image_embedding, attribute_embedding], dim=1)
        gate = self.gate_network(joint_embedding)
        gated_embedding = gate * image_embedding + (1.0 - gate) * attribute_embedding
        final_embedding = torch.cat([image_embedding, attribute_embedding, gated_embedding], dim=1)
        logits = self.classifier(final_embedding).squeeze(1)
        if return_gate and return_embedding:
            return (logits, gate, final_embedding)
        if return_gate:
            return (logits, gate)
        if return_embedding:
            return (logits, final_embedding)
        return logits


class TemporalResidualBlock(nn.Module):

    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.GroupNorm(num_groups=8, num_channels=channels), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.GroupNorm(num_groups=8, num_channels=channels), nn.GELU(), nn.Dropout(dropout))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.output_activation = nn.GELU()

    def forward(self, x):
        residual = self.block(x)
        return self.output_activation(x + self.residual_scale * residual)


class C3DepthContextTCN(nn.Module):

    def __init__(self, input_dim=771, embedding_dim=768, channels=256, dilations=(1, 2, 4), dropout=0.2, center_position=7):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.center_position = center_position
        self.input_normalization = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(nn.Conv1d(input_dim, channels, kernel_size=1, bias=False), nn.GroupNorm(num_groups=8, num_channels=channels), nn.GELU())
        self.temporal_blocks = nn.ModuleList([TemporalResidualBlock(channels=channels, dilation=dilation, dropout=dropout) for dilation in dilations])
        self.center_projection = nn.Sequential(nn.Linear(embedding_dim, channels), nn.LayerNorm(channels), nn.GELU())
        self.context_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.LayerNorm(channels), nn.GELU(), nn.Linear(channels, channels), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(channels * 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, sequence, return_gate=False):
        normalized_sequence = self.input_normalization(sequence)
        temporal_input = normalized_sequence.transpose(1, 2)
        temporal_feature = self.input_projection(temporal_input)
        for block in self.temporal_blocks:
            temporal_feature = block(temporal_feature)
        context_embedding = temporal_feature[:, :, self.center_position]
        center_raw_embedding = sequence[:, self.center_position, :self.embedding_dim]
        center_embedding = self.center_projection(center_raw_embedding)
        gate_input = torch.cat([center_embedding, context_embedding], dim=1)
        context_gate = self.context_gate(gate_input)
        gated_embedding = context_gate * context_embedding + (1.0 - context_gate) * center_embedding
        final_embedding = torch.cat([center_embedding, context_embedding, gated_embedding], dim=1)
        logits = self.classifier(final_embedding).squeeze(1)
        if return_gate:
            return (logits, context_gate)
        return logits
