import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler



INPUT_AUDIO_DIR = Path("/root/autodl-tmp/dataset/EAV/Input_images/Audio")
OUTPUT_FEATURE_DIR = Path("/root/autodl-tmp/dataset/EAV/features/Audio_openSMILE")

OPENSMILE_BIN = Path(
    "/root/autodl-tmp/dataset/EAV/features/opensmile-3.0.2-linux-x86_64/bin/SMILExtract"
)
OPENSMILE_CONFIG = Path(
    "/root/autodl-tmp/dataset/EAV/features/opensmile-3.0.2-linux-x86_64/config/is09-13/IS10_paraling.conf"
)

SUBJECT_START = 1
SUBJECT_END = 42

SAMPLE_RATE = 16000
TRAIN_COUNT = 280
EXPECTED_TOTAL = 400

FEATURE_START_COL = 1
FEATURE_END_COL = 1583


# =========================
# 工具函数
# =========================
def check_path_exists(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")


def load_audio_pkl(pkl_path: Path):
    """
    原始音频 pkl 格式默认是:
        [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
    或至少满足:
        第0项 = train audio
        第2项 = test audio
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)) or len(data) < 3:
        raise ValueError(f"音频 pkl 格式异常: {pkl_path}")

    tr_x_aud = data[0]
    te_x_aud = data[2]
    return tr_x_aud, te_x_aud


def save_wav(audio_array, wav_path: Path, sample_rate=16000):
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    wavfile.write(str(wav_path), sample_rate, audio_int16)


def run_opensmile(wav_path: Path, csv_path: Path):
    cmd = [
        str(OPENSMILE_BIN),
        "-C", str(OPENSMILE_CONFIG),
        "-I", str(wav_path),
        "-O", str(csv_path),
        "--logLevel", "2",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(
            f"openSMILE 提取失败:\n"
            f"WAV: {wav_path}\n"
            f"STDOUT:\n{result.stdout.decode(errors='ignore')}\n"
            f"STDERR:\n{result.stderr.decode(errors='ignore')}"
        )


def read_feature_from_csv(csv_path: Path):
    """
    兼容 openSMILE 输出的带头部信息文件。
    常见格式是：
        @relation ...
        @attribute ...
        ...
        @data
        unknown,0.1,0.2,...,xxx

    原 notebook 的目标是取：
        data.iloc[0, 1:1583]
    即跳过第一列，取后面 1582 维特征。
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 找 @data
    data_idx = None
    for i, line in enumerate(lines):
        if line.lower() == "@data":
            data_idx = i
            break

    if data_idx is None:
        raise ValueError(f"文件中未找到 @data: {csv_path}")

    # 找 @data 后第一行真正数据
    feature_line = None
    for i in range(data_idx + 1, len(lines)):
        line = lines[i]
        if not line.startswith("@"):
            feature_line = line
            break

    if feature_line is None:
        raise ValueError(f"@data 后未找到特征数据行: {csv_path}")

    parts = [x.strip() for x in feature_line.split(",")]

    expected_dim = FEATURE_END_COL - FEATURE_START_COL
    if len(parts) < FEATURE_END_COL:
        raise ValueError(
            f"特征列数不足: {csv_path}, got {len(parts)}, expected at least {FEATURE_END_COL}"
        )

    feat = np.asarray(parts[FEATURE_START_COL:FEATURE_END_COL], dtype=np.float32)

    if feat.shape[0] != expected_dim:
        raise ValueError(
            f"特征维度异常: {csv_path}, got {feat.shape[0]}, expected {expected_dim}"
        )

    return feat



def extract_single_feature(audio_array, temp_dir: Path, file_id: int):
    """
    单条音频:
    ndarray -> 临时wav -> openSMILE csv -> 读取特征 -> 返回feature
    """
    wav_path = temp_dir / f"{file_id:03d}.wav"
    csv_path = temp_dir / f"{file_id:03d}_features.csv"

    save_wav(audio_array, wav_path, SAMPLE_RATE)
    run_opensmile(wav_path, csv_path)
    feat = read_feature_from_csv(csv_path)

    if feat.shape[0] != (FEATURE_END_COL - FEATURE_START_COL):
        raise ValueError(
            f"特征维度异常: {csv_path}, got {feat.shape[0]}, "
            f"expected {FEATURE_END_COL - FEATURE_START_COL}"
        )

    return feat


def process_subject(subject_id: int):
    subject_name = f"subject_{subject_id:02d}_aud.pkl"
    input_pkl = INPUT_AUDIO_DIR / subject_name
    output_pkl = OUTPUT_FEATURE_DIR / subject_name

    print(f"\n========== 开始处理 subject_{subject_id:02d} ==========")
    print(f"读取: {input_pkl}")

    tr_x_aud, te_x_aud = load_audio_pkl(input_pkl)

    print(f"训练音频数量: {len(tr_x_aud)}")
    print(f"测试音频数量: {len(te_x_aud)}")

    if len(tr_x_aud) != TRAIN_COUNT:
        print(f"[警告] 训练样本数为 {len(tr_x_aud)}，预期为 {TRAIN_COUNT}")

    if len(tr_x_aud) + len(te_x_aud) != EXPECTED_TOTAL:
        print(f"[警告] 总样本数为 {len(tr_x_aud) + len(te_x_aud)}，预期为 {EXPECTED_TOTAL}")

    all_features = []

    with tempfile.TemporaryDirectory(prefix=f"subject_{subject_id:02d}_") as temp_folder:
        temp_dir = Path(temp_folder)

        # 训练集
        for idx, audio_array in enumerate(tr_x_aud):
            feat = extract_single_feature(audio_array, temp_dir, idx)
            all_features.append(feat)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(tr_x_aud):
                print(f"  训练集已处理: {idx + 1}/{len(tr_x_aud)}")

        # 测试集
        for idx, audio_array in enumerate(te_x_aud):
            file_id = idx + len(tr_x_aud)
            feat = extract_single_feature(audio_array, temp_dir, file_id)
            all_features.append(feat)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(te_x_aud):
                print(f"  测试集已处理: {idx + 1}/{len(te_x_aud)}")

    all_features = np.asarray(all_features, dtype=np.float32)

    tr_features = all_features[:len(tr_x_aud)]
    te_features = all_features[len(tr_x_aud):]

    print(f"提取完成，训练特征形状: {tr_features.shape}")
    print(f"提取完成，测试特征形状: {te_features.shape}")

    scaler = StandardScaler()
    tr_features_std = scaler.fit_transform(tr_features)
    te_features_std = scaler.transform(te_features)

    final_data = [tr_features_std, te_features_std]

    OUTPUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(final_data, f)

    print(f"最终特征已保存: {output_pkl}")


def main():
    check_path_exists(INPUT_AUDIO_DIR, "输入音频目录")
    check_path_exists(OPENSMILE_BIN, "openSMILE 可执行文件")
    check_path_exists(OPENSMILE_CONFIG, "openSMILE 配置文件")

    OUTPUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    for subject_id in range(SUBJECT_START, SUBJECT_END + 1):
        input_pkl = INPUT_AUDIO_DIR / f"subject_{subject_id:02d}_aud.pkl"
        if not input_pkl.exists():
            print(f"[跳过] 文件不存在: {input_pkl}")
            continue

        try:
            process_subject(subject_id)
        except Exception as e:
            print(f"[错误] subject_{subject_id:02d} 处理失败: {e}")

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
