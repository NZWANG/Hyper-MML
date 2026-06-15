# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

class RecorderMeter:
    pass



# =========================
# 固定路径
# =========================
INPUT_DIR = "/root/autodl-tmp/dataset/EAV/Input_images/Vision"
OUTPUT_DIR = "/root/autodl-tmp/dataset/EAV/features/Vision_manet_new"
CHECKPOINT_FILE = "/root/autodl-tmp/dataset/EAV/features/Vision_manet/[02-08]-[21-19]-model_best-acc88.33.pth"

START_SUBJECT = 1
END_SUBJECT = 42
NUM_CLASSES = 7


# =========================
# CBAM
# =========================
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False
    ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == "avg":
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(pool)
            elif pool_type == "max":
                pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(pool)
            else:
                continue

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x


# =========================
# MANet
# =========================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MulScaleBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)
        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)

        self.conv2_2_1 = conv3x3(scale_width, scale_width)
        self.bn2_2_1 = norm_layer(scale_width)
        self.conv2_2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2_2 = norm_layer(scale_width)
        self.conv2_2_3 = conv3x3(scale_width, scale_width)
        self.bn2_2_3 = norm_layer(scale_width)
        self.conv2_2_4 = conv3x3(scale_width, scale_width)
        self.bn2_2_4 = norm_layer(scale_width)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        out_1_1 = self.relu(self.bn1_2_1(self.conv1_2_1(sp_x[0])))
        out_1_2 = self.relu(self.bn1_2_2(self.conv1_2_2(out_1_1 + sp_x[1])))
        out_1_3 = self.relu(self.bn1_2_3(self.conv1_2_3(out_1_2 + sp_x[2])))
        out_1_4 = self.bn1_2_4(self.conv1_2_4(out_1_3 + sp_x[3]))

        output_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)

        out_2_1 = self.relu(self.bn2_2_1(self.conv2_2_1(sp_x[0])))
        out_2_2 = self.relu(self.bn2_2_2(self.conv2_2_2(out_2_1 + sp_x[1])))
        out_2_3 = self.relu(self.bn2_2_3(self.conv2_2_3(out_2_2 + sp_x[2])))
        out_2_4 = self.bn2_2_4(self.conv2_2_4(out_2_3 + sp_x[3]))

        output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)

        out = output_1 + output_2

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MANet(nn.Module):
    def __init__(self, block_b, block_m, block_a, layers, num_classes=7):
        super(MANet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)

        self.layer3_1_p1 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p1 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p2 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p2 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p3 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p3 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p4 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p4 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_2 = self._make_layer(block_m, 128, 256, layers[2], stride=2)
        self.layer4_2 = self._make_layer(block_m, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512, num_classes)
        self.fc_2 = nn.Linear(512, num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))

        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        patch_11 = x[:, :, 0:14, 0:14]
        patch_12 = x[:, :, 0:14, 14:28]
        patch_21 = x[:, :, 14:28, 0:14]
        patch_22 = x[:, :, 14:28, 14:28]

        branch_1_p1_out = self.layer4_1_p1(self.layer3_1_p1(patch_11))
        branch_1_p2_out = self.layer4_1_p2(self.layer3_1_p2(patch_12))
        branch_1_p3_out = self.layer4_1_p3(self.layer3_1_p3(patch_21))
        branch_1_p4_out = self.layer4_1_p4(self.layer3_1_p4(patch_22))

        branch_1_out_1 = torch.cat([branch_1_p1_out, branch_1_p2_out], dim=3)
        branch_1_out_2 = torch.cat([branch_1_p3_out, branch_1_p4_out], dim=3)
        branch_1_out = torch.cat([branch_1_out_1, branch_1_out_2], dim=2)

        branch_1_out = self.avgpool(branch_1_out)
        branch_1_embedding = torch.flatten(branch_1_out, 1)
        branch_1_cls = self.fc_1(branch_1_embedding)

        branch_2_out = self.layer4_2(self.layer3_2(x))
        branch_2_out = self.avgpool(branch_2_out)
        branch_2_embedding = torch.flatten(branch_2_out, 1)
        branch_2_cls = self.fc_2(branch_2_embedding)

        if return_embedding:
            return torch.cat([branch_1_embedding, branch_2_embedding], dim=1)

        return branch_1_cls, branch_2_cls


def manet(**kwargs):
    return MANet(
        block_b=BasicBlock,
        block_m=MulScaleBlock,
        block_a=AttentionBlock,
        layers=[2, 2, 2, 2],
        **kwargs
    )


# =========================
# 工具函数
# =========================
def load_model(device):
    model = manet(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    pre_trained_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(pre_trained_dict)
    model.eval()
    return model


def preprocess_frames(frames, transform):
    processed = []
    for i in range(frames.shape[0]):
        frame = frames[i].astype(np.uint8)
        frame = transform(frame)
        processed.append(frame)
    return torch.stack(processed, dim=0)


def extract_features(model, vis_data, transform, device):
    all_features = []

    with torch.no_grad():
        for i in tqdm(range(vis_data.shape[0]), leave=False):
            frames = vis_data[i]  # (25, 56, 56, 3)
            frames_tensor = preprocess_frames(frames, transform).to(device)  # (25, 3, 224, 224)
            embeddings = model(frames_tensor, return_embedding=True)          # (25, 1024)
            feature = torch.mean(embeddings, dim=0).cpu().numpy()            # (1024,)
            all_features.append(feature)

    return np.array(all_features)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = load_model(device)

    for subject in range(START_SUBJECT, END_SUBJECT + 1):
        input_file = os.path.join(INPUT_DIR, f"subject_{subject:02d}_vis.pkl")
        output_file = os.path.join(OUTPUT_DIR, f"subject_{subject:02d}_vis.pkl")

        if not os.path.exists(input_file):
            print(f"[跳过] 文件不存在: {input_file}")
            continue

        with open(input_file, "rb") as f:
            vis_list = pickle.load(f)

        tr_vis, _, te_vis, _ = vis_list

        print(f"subject_{subject:02d}")
        print("train:", tr_vis.shape)
        print("test :", te_vis.shape)

        tr_feature = extract_features(model, tr_vis, transform, device)
        te_feature = extract_features(model, te_vis, transform, device)

        scaler = StandardScaler()
        tr_features_standardized = scaler.fit_transform(tr_feature)
        te_features_standardized = scaler.transform(te_feature)

        vis_feature = [tr_features_standardized, te_features_standardized]

        with open(output_file, "wb") as f:
            pickle.dump(vis_feature, f)

        print(f"vis features of subject_{subject:02d} has been saved!")


if __name__ == "__main__":
    main()
