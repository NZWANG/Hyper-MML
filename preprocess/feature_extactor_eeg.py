import os
import math
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================
# 1. Config
# =========================================
class Config:
    # 原始 EEG 数据目录
    data_dir = "/root/autodl-tmp/dataset/EAV/Input_images/EEG"   
    # 已训练好的 best model
    model_path = "./EEG_abema/abema_best.pt"
    # 导出 embedding 的保存目录
    output_dir = "./EEG_abema"

    # data info
    num_subjects = 42
    num_channels = 30
    seq_len = 500
    fs = 100
    num_classes = 5

    # model info: 必须和训练时一致
    d_k = 128
    embed_dim = 1024
    hidden_dim = 128
    dropout = 0.2

    # extraction
    batch_size = 64
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# 2. Dataset
# =========================================
class EEGOnlyDataset(Dataset):
    def __init__(self, samples, subject_id, normalize_per_sample=True):
        self.samples = samples.astype(np.float32)
        self.subject_id = int(subject_id)
        self.normalize_per_sample = normalize_per_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]   # [C, L]

        if self.normalize_per_sample:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)
        s = torch.tensor(self.subject_id, dtype=torch.long)
        return x, s


# =========================================
# 3. Model Definition (must match training)
# =========================================
class SubjectSpecificLayer(nn.Module):
    def __init__(self, num_subjects=42, num_channels=30, init_noise=0.01):
        super(SubjectSpecificLayer, self).__init__()
        weight = torch.eye(num_channels).unsqueeze(0).repeat(num_subjects, 1, 1)
        weight = weight + init_noise * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x, subject_ids):
        W = self.weight[subject_ids]  # [B, C, C]
        out = torch.einsum("bij,bjt->bit", W, x)
        return out


class TemporalSpatialEncoder(nn.Module):
    def __init__(self, num_channels=30, seq_len=500, hidden_dim=128, dropout=0.2):
        super(TemporalSpatialEncoder, self).__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.channel_mixing = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_channels, kernel_size=1),
        )

        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x):
        x = x + self.temporal_conv(x)
        x = x + self.channel_mixing(x)
        x = self.norm(x)
        return x


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, fs=100, seq_len=500, eps=1e-6):
        super(FrequencyFeatureExtractor, self).__init__()
        self.fs = fs
        self.seq_len = seq_len
        self.eps = eps

        freqs = np.fft.rfftfreq(seq_len, d=1.0 / fs)
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

        self.band_defs = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 50.0),
        }

    def band_mask(self, low, high):
        if high >= self.fs / 2:
            return (self.freqs >= low) & (self.freqs <= high)
        else:
            return (self.freqs >= low) & (self.freqs < high)

    def forward(self, x):
        Xf = torch.rfft(x, signal_ndim=1, normalized=False, onesided=True)  # [B,C,F,2]
        power = Xf[..., 0].pow(2) + Xf[..., 1].pow(2)  # [B,C,F]

        band_features = {}
        for band_name, (low, high) in self.band_defs.items():
            mask = self.band_mask(low, high)
            band_power = power[:, :, mask]
            psd = band_power.mean(dim=-1)
            var = band_power.mean(dim=-1) + self.eps
            de = 0.5 * torch.log(2 * math.pi * math.e * var)

            band_features[band_name] = {
                "de": de,
                "psd": psd
            }

        return band_features


class IntraBandFusion(nn.Module):
    def __init__(self, num_channels=30, d_k=128, dropout=0.2):
        super(IntraBandFusion, self).__init__()

        self.de_proj = nn.Sequential(
            nn.Linear(num_channels, d_k),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.psd_proj = nn.Sequential(
            nn.Linear(num_channels, d_k),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_k * 2, d_k),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_k)

    def forward(self, de, psd):
        h_de = self.de_proj(de)
        h_psd = self.psd_proj(psd)
        g = self.gate(torch.cat([h_de, h_psd], dim=-1))
        out = g * h_de + (1.0 - g) * h_psd
        out = self.norm(out)
        return out


class InterBandAttention(nn.Module):
    def __init__(self, d_k=128, num_heads=4, dropout=0.2):
        super(InterBandAttention, self).__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=d_k,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_k)

        self.ffn = nn.Sequential(
            nn.Linear(d_k, d_k * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_k * 2, d_k),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(d_k)

    def forward(self, x):
        x_t = x.transpose(0, 1)  # [5, B, d_k]
        attn_out, _ = self.mha(x_t, x_t, x_t)
        x_t = x_t + attn_out
        x = x_t.transpose(0, 1)  # [B, 5, d_k]
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AdaptiveBandFusion(nn.Module):
    def __init__(self, d_k=128, embed_dim=1024, dropout=0.2):
        super(AdaptiveBandFusion, self).__init__()

        self.band_score = nn.Linear(d_k, 1)

        self.projector = nn.Sequential(
            nn.Linear(d_k, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        scores = self.band_score(x).squeeze(-1)   # [B, 5]
        alpha = torch.softmax(scores, dim=1)      # [B, 5]
        fused = torch.sum(alpha.unsqueeze(-1) * x, dim=1)  # [B, d_k]
        emb = self.projector(fused)               # [B, 1024]
        return emb, alpha


class ABEMAEncoder(nn.Module):
    def __init__(
        self,
        num_subjects=42,
        num_channels=30,
        seq_len=500,
        fs=100,
        d_k=128,
        embed_dim=1024,
        hidden_dim=128,
        dropout=0.2
    ):
        super(ABEMAEncoder, self).__init__()

        self.subject_layer = SubjectSpecificLayer(
            num_subjects=num_subjects,
            num_channels=num_channels,
            init_noise=0.01
        )

        self.encoder = TemporalSpatialEncoder(
            num_channels=num_channels,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.freq_extractor = FrequencyFeatureExtractor(
            fs=fs,
            seq_len=seq_len
        )

        self.band_names = ["delta", "theta", "alpha", "beta", "gamma"]

        self.intra_fusion = nn.ModuleDict({
            band: IntraBandFusion(num_channels=num_channels, d_k=d_k, dropout=dropout)
            for band in self.band_names
        })

        self.inter_band = InterBandAttention(
            d_k=d_k,
            num_heads=4,
            dropout=dropout
        )

        self.adaptive_fusion = AdaptiveBandFusion(
            d_k=d_k,
            embed_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, x, subject_ids):
        x = self.subject_layer(x, subject_ids)
        x = self.encoder(x)
        band_features = self.freq_extractor(x)

        fused_bands = []
        for band in self.band_names:
            de = band_features[band]["de"]
            psd = band_features[band]["psd"]
            f_b = self.intra_fusion[band](de, psd)
            fused_bands.append(f_b)

        F = torch.stack(fused_bands, dim=1)
        F = self.inter_band(F)
        emb, alpha = self.adaptive_fusion(F)
        return emb, alpha


class ABEMAClassifier(nn.Module):
    def __init__(self, num_classes=5, **encoder_kwargs):
        super(ABEMAClassifier, self).__init__()
        self.encoder = ABEMAEncoder(**encoder_kwargs)
        embed_dim = encoder_kwargs.get("embed_dim", 1024)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, subject_ids):
        emb, alpha = self.encoder(x, subject_ids)
        logits = self.classifier(emb)
        return logits, emb, alpha


# =========================================
# 4. Extraction
# =========================================
def build_model(cfg):
    model = ABEMAClassifier(
        num_classes=cfg.num_classes,
        num_subjects=cfg.num_subjects,
        num_channels=cfg.num_channels,
        seq_len=cfg.seq_len,
        fs=cfg.fs,
        d_k=cfg.d_k,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout
    )
    return model


def extract_embeddings_only(model, loader, device):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for x, s in loader:
            x = x.to(device)
            s = s.to(device)

            emb, alpha = model.encoder(x, s)
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def load_subject_data(file_path):
    with open(file_path, "rb") as f:
        eeg_list = pickle.load(f)

    if len(eeg_list) != 4:
        raise ValueError("%s should contain [train_x, train_y, test_x, test_y]" % file_path)

    train_x, train_y, test_x, test_y = eeg_list
    return train_x, test_x


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Torch version:", torch.__version__)
    print("Device:", cfg.device)
    print("Loading model...")

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    print("Model loaded from:", cfg.model_path)

    for subject in range(1, cfg.num_subjects + 1):
        subject_file = os.path.join(cfg.data_dir, "subject_%02d_eeg.pkl" % subject)
        if not os.path.exists(subject_file):
            print("Skip missing file:", subject_file)
            continue

        print("Processing subject %02d ..." % subject)

        train_x, test_x = load_subject_data(subject_file)
        subject_id = subject - 1  # 0..41

        train_dataset = EEGOnlyDataset(
            train_x, subject_id=subject_id, normalize_per_sample=True
        )
        test_dataset = EEGOnlyDataset(
            test_x, subject_id=subject_id, normalize_per_sample=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False
        )

        train_embeddings = extract_embeddings_only(model, train_loader, cfg.device)
        test_embeddings = extract_embeddings_only(model, test_loader, cfg.device)

        save_list = [train_embeddings, test_embeddings]

        out_path = os.path.join(cfg.output_dir, "subject_%02d_eeg.pkl" % subject)
        with open(out_path, "wb") as f:
            pickle.dump(save_list, f)

        print(
            "Saved %s | train_emb: %s | test_emb: %s"
            % (out_path, str(train_embeddings.shape), str(test_embeddings.shape))
        )

    print("All subjects finished.")


if __name__ == "__main__":
    main()
