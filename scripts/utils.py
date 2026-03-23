"""utils.py — Loracle 공통 유틸리티"""

import glob
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

PREPROCESSED_DIR = "../preprocessed"
MODELS_DIR       = "../models"
FINAL_MODEL_DIR  = os.path.join(MODELS_DIR, "final")

# 04_patch_labels.py 기준: 1=버프  2=너프
LABEL_MAP = {0: "유지", 1: "버프", 2: "너프"}

MINORITY_THRESHOLD = 0.28

NON_FEATURE_COLS = {
    "champion", "label", "top_item_combo",
    "primary_position", "patch_name",
    "pick_count", "win_count", "ban_count",
}


def load_all_patches() -> pd.DataFrame:
    """preprocessed/ 의 전체 패치 CSV를 로드하고 patch_num(1~N)을 부여. label=3(조정) 제외."""
    paths = sorted(glob.glob(os.path.join(PREPROCESSED_DIR, "ml_dataset_*.csv")))
    if not paths:
        raise FileNotFoundError(f"{PREPROCESSED_DIR} 에 데이터가 없습니다.")

    dfs = []
    for i, path in enumerate(paths, start=1):
        df = pd.read_csv(path)
        df["patch_num"]  = i
        df["patch_name"] = os.path.basename(path).replace("ml_dataset_", "").replace(".csv", "")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["label"] != 3].copy()

    print(f"[데이터] {len(paths)}개 패치 로드 | 총 {len(combined)}행 | "
          f"라벨: {combined['label'].value_counts().to_dict()}")
    return combined


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """파생 피처 추가: 직전 패치 라벨 원-핫, 패치 순서 정규화.

    제거된 피처 (절약성 원칙):
        flag_wr_high/low, flag_ban_high, flag_presence_high — 임의 역치 이진화 (win_rate 등과 중복)
        wr_x_presence  — op_index(승률×픽률)와 중복
        delta_momentum — 이론적 근거 불명확한 복합 상호작용
    """
    df = df.copy()

    # 직전 패치 라벨 (patch_num 기준 조인)
    df = df.sort_values(["patch_num", "champion"]).reset_index(drop=True)
    prev = df[["champion", "patch_num", "label"]].copy()
    prev["patch_num"] += 1
    prev = prev.rename(columns={"label": "last_label"})
    df = df.merge(prev, on=["champion", "patch_num"], how="left")
    df["last_label"] = df["last_label"].fillna(0).astype(int)
    for v in [0, 1, 2]:
        df[f"last_label_{v}"] = (df["last_label"] == v).astype(float)
    df.drop(columns=["last_label"], inplace=True)

    df["patch_norm"] = df["patch_num"] / df["patch_num"].max()

    return df


def prepare_features(df: pd.DataFrame, fit_columns: list = None) -> tuple:
    """수치 컬럼 선택 + primary_position 원-핫 인코딩 + NaN 처리.

    fit_columns 지정 시 해당 컬럼으로 reindex (OOT 예측에서 컬럼 정렬에 사용).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS]

    X = pd.get_dummies(
        df[feature_cols + ["primary_position"]],
        columns=["primary_position"],
        drop_first=False,
    )

    # 포지션별 승률 NaN → 0.5 (중립값), 나머지 → 중앙값
    wr_cols = [c for c in X.columns if c.startswith("win_rate_") and c != "win_rate_delta"]
    X[wr_cols] = X[wr_cols].fillna(0.5)
    X = X.fillna(X.median(numeric_only=True))

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0)

    return X, df["label"]


def make_sampler(min_class_count: int, n_splits: int):
    """fold 내 소수 클래스 샘플 수를 추정해 SMOTE 또는 RandomOverSampler 반환."""
    fold_min = max(1, int(min_class_count * (n_splits - 1) / n_splits))
    if fold_min > 1:
        return SMOTE(random_state=42, k_neighbors=min(5, fold_min - 1))
    return RandomOverSampler(random_state=42)


def predict_with_threshold(model, X: pd.DataFrame, threshold: float = MINORITY_THRESHOLD) -> np.ndarray:
    """버프(1)/너프(2) 소수 클래스에 낮은 임계값을 적용한 예측. 후보 없으면 유지(0) 반환."""
    proba   = model.predict_proba(X)
    classes = list(model.classes_)
    preds   = []

    for p in proba:
        candidates = {
            lbl: p[classes.index(lbl)]
            for lbl in [1, 2]
            if lbl in classes and p[classes.index(lbl)] >= threshold
        }
        preds.append(max(candidates, key=candidates.get) if candidates else 0)

    return np.array(preds)


def load_api_key(path: str = "../default_info/api.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
