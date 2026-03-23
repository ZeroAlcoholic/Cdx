"""PPO 訓練流程骨架（MVP placeholder）。"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from validate_env import run_all_checks, summarize_results


@dataclass
class SplitConfig:
    """時間切分設定。"""

    train_start: str
    valid_start: str
    test_start: str


@dataclass
class TrainConfig:
    """訓練與驗證必要參數。"""

    initial_cash: float
    fee_bps: float
    slippage_bps: float
    checkpoint_dir: str
    seed: int = 42


def time_split(df: pd.DataFrame, cfg: SplitConfig):
    """依時間切 train/valid/test。"""
    dates = pd.to_datetime(df["date"])
    train = df[(dates >= pd.to_datetime(cfg.train_start)) & (dates < pd.to_datetime(cfg.valid_start))]
    valid = df[(dates >= pd.to_datetime(cfg.valid_start)) & (dates < pd.to_datetime(cfg.test_start))]
    test = df[dates >= pd.to_datetime(cfg.test_start)]
    return train, valid, test


def run_training_gate(feature_daily: pd.DataFrame, cfg: TrainConfig) -> dict:
    """訓練前閘門：若環境檢查失敗，直接拒絕進入訓練。"""
    results = run_all_checks(
        feature_daily=feature_daily,
        initial_cash=cfg.initial_cash,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
    )
    summary = summarize_results(results)
    if not summary["all_passed"]:
        failed = [r["name"] for r in summary["results"] if not r["passed"]]
        raise ValueError(f"訓練前驗證失敗，禁止訓練。failed_checks={failed}")
    return summary


def train_ppo_placeholder(feature_daily: pd.DataFrame, cfg: TrainConfig) -> str:
    """建立可重現訓練輸出（占位實作）；可替換成真實 PPO。"""
    gate_summary = run_training_gate(feature_daily=feature_daily, cfg=cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    artifact = {
        "algo": "PPO-placeholder",
        "seed": cfg.seed,
        "n_samples": int(len(feature_daily)),
        "status": "trained",
        "gate_summary": gate_summary,
    }
    ckpt_path = os.path.join(cfg.checkpoint_dir, "ppo_baseline_checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return ckpt_path
