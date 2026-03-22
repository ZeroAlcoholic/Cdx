"""PPO 訓練流程骨架（MVP placeholder）。"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    """時間切分設定。"""

    train_start: str
    valid_start: str
    test_start: str


def time_split(df: pd.DataFrame, cfg: SplitConfig):
    """依時間切 train/valid/test。"""
    dates = pd.to_datetime(df["date"])
    train = df[(dates >= pd.to_datetime(cfg.train_start)) & (dates < pd.to_datetime(cfg.valid_start))]
    valid = df[(dates >= pd.to_datetime(cfg.valid_start)) & (dates < pd.to_datetime(cfg.test_start))]
    test = df[dates >= pd.to_datetime(cfg.test_start)]
    return train, valid, test


def train_ppo_placeholder(feature_daily: pd.DataFrame, checkpoint_dir: str, seed: int = 42) -> str:
    """建立可重現訓練輸出（占位實作）；可替換成真實 PPO。"""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    artifact = {
        "algo": "PPO-placeholder",
        "seed": seed,
        "n_samples": int(len(feature_daily)),
        "status": "trained",
    }
    ckpt_path = os.path.join(checkpoint_dir, "ppo_baseline_checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return ckpt_path
