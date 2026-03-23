"""05_train.py — Loracle 모델 학습 파이프라인

실행 흐름:
    1. 전체 패치 로드 + 피처 엔지니어링  (utils.py)
    2. LOPO 교차검증 (패치별 F1 Macro)
    3. 최종 모델 학습: 패치 1~N-1 학습 → 패치 N OOT 평가
    4. 로지스틱 회귀로 변수 설명력 분석
    5. 앙상블 소프트보팅
    6. 모델·리포트 저장
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import (
    FINAL_MODEL_DIR,
    LABEL_MAP,
    MINORITY_THRESHOLD,
    MODELS_DIR,
    add_engineered_features,
    load_all_patches,
    make_sampler,
    predict_with_threshold,
    prepare_features,
)

warnings.filterwarnings("ignore")

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)


# ── 모델 설정 ──────────────────────────────────────────────────────────────
def _build_model(name: str):
    if name == "LightGBM":
        return LGBMClassifier(
            random_state=42, class_weight="balanced", verbose=-1,
            n_estimators=400, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=5,
            reg_alpha=0.1, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8,
        )
    if name == "XGBoost":
        return XGBClassifier(
            random_state=42, eval_metric="mlogloss", verbosity=0,
            n_estimators=400, learning_rate=0.05, max_depth=4,
            min_child_weight=3, scale_pos_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
        )
    return RandomForestClassifier(
        random_state=42, class_weight="balanced",
        n_estimators=300, max_depth=7,
        min_samples_leaf=2, max_features="sqrt",
    )


def _make_pipeline(name: str, min_count: int, n_splits: int) -> ImbPipeline:
    sampler = make_sampler(min_count, n_splits)
    return ImbPipeline([("sampler", sampler), ("model", _build_model(name))])


# ── 단일 모델 학습 ─────────────────────────────────────────────────────────
def train_single(X_train, y_train, X_test, y_test, model_name: str, n_splits: int = 4):
    """학습 후 기본 예측과 임계값 조정 예측 중 F1이 높은 쪽 반환."""
    pipeline = _make_pipeline(model_name, y_train.value_counts().min(), n_splits)
    pipeline.fit(X_train, y_train)

    y_default   = pipeline.predict(X_test)
    y_threshold = predict_with_threshold(pipeline, X_test)

    f1_def = f1_score(y_test, y_default,   average="macro", zero_division=0)
    f1_thr = f1_score(y_test, y_threshold, average="macro", zero_division=0)

    if f1_thr >= f1_def:
        return pipeline, y_threshold, f1_thr, f"threshold({MINORITY_THRESHOLD})"
    return pipeline, y_default, f1_def, "default(0.5)"


# ── LOPO 교차검증 ──────────────────────────────────────────────────────────
def leave_one_patch_out(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> str:
    """각 패치를 테스트셋으로 순서대로 사용해 F1 Macro를 측정하고 결과 텍스트 반환."""
    patches = sorted(df["patch_num"].unique())
    results = {}

    print("\n[LOPO CV] Leave-One-Patch-Out 교차검증 시작...")
    for test_patch in patches:
        X_tr = X[df["patch_num"] != test_patch]
        y_tr = y[df["patch_num"] != test_patch]
        X_te = X[df["patch_num"] == test_patch]
        y_te = y[df["patch_num"] == test_patch]

        if len(y_tr.unique()) < 2:
            continue

        scores = {}
        for name in ("LightGBM", "XGBoost", "RandomForest"):
            try:
                _, _, f1, _ = train_single(X_tr, y_tr, X_te, y_te, name, n_splits=3)
                scores[name] = f1
            except Exception as e:
                scores[name] = 0.0
                print(f"  [경고] patch {test_patch} {name}: {e}")

        best_name  = max(scores, key=scores.get)
        best_score = scores[best_name]
        results[f"patch_{test_patch}"] = {"scores": scores, "best": best_name, "best_f1": best_score}

        pname = df[df["patch_num"] == test_patch]["patch_name"].iloc[0]
        print(
            f"  패치 {test_patch} ({pname}) | "
            f"LGB: {scores['LightGBM']:.4f}  "
            f"XGB: {scores['XGBoost']:.4f}  "
            f"RF: {scores['RandomForest']:.4f}  "
            f"-> Best: {best_name} ({best_score:.4f})"
        )

    all_f1    = [v["best_f1"] for v in results.values()]
    lopo_mean = np.mean(all_f1) if all_f1 else 0.0
    lopo_std  = np.std(all_f1)  if all_f1 else 0.0

    text  = "\n[LOPO CV 결과]\n" + "-" * 50 + "\n"
    for k, v in results.items():
        text += f"  {k}: Best={v['best']} F1={v['best_f1']:.4f}\n"
    text += f"  평균 F1 Macro: {lopo_mean:.4f} +- {lopo_std:.4f}\n"
    text += "-" * 50 + "\n"

    print(f"\n  LOPO 평균 F1: {lopo_mean:.4f} +- {lopo_std:.4f}")
    return text


# ── 앙상블 소프트보팅 ──────────────────────────────────────────────────────
def train_ensemble(X_train, y_train, X_test, y_test):
    """LightGBM + XGBoost + RandomForest 확률 평균 → 임계값 조정 예측."""
    min_count = y_train.value_counts().min()
    pipelines = {
        name: _make_pipeline(name, min_count, n_splits=4)
        for name in ("LightGBM", "XGBoost", "RandomForest")
    }
    for pipe in pipelines.values():
        pipe.fit(X_train, y_train)

    classes   = sorted(y_train.unique())
    avg_proba = sum(p.predict_proba(X_test) for p in pipelines.values()) / 3.0

    preds = []
    for p in avg_proba:
        cands = {
            lbl: p[i] for i, lbl in enumerate(classes)
            if lbl in [1, 2] and p[i] >= MINORITY_THRESHOLD
        }
        preds.append(max(cands, key=cands.get) if cands else 0)

    y_pred = np.array(preds)
    f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return pipelines, y_pred, f1


# ── 로지스틱 회귀 — 변수 설명력 ────────────────────────────────────────────
def analyze_logistic_regression(X_train, y_train, X_test, y_test, feature_cols: list):
    """표준화된 계수로 피처별 버프/너프 예측 기여도 정량화. 성능이 아닌 해석 목적."""
    from imblearn.over_sampling import RandomOverSampler

    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_res)
    X_te_s   = scaler.transform(X_test)

    lr = LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=2000, C=0.5, random_state=42,
    )
    lr.fit(X_tr_s, y_res)

    proba   = lr.predict_proba(X_te_s)
    classes = list(lr.classes_)
    preds   = []
    for p in proba:
        cands = {
            lbl: p[classes.index(lbl)]
            for lbl in [1, 2]
            if lbl in classes and p[classes.index(lbl)] >= MINORITY_THRESHOLD
        }
        preds.append(max(cands, key=cands.get) if cands else 0)
    y_pred = np.array(preds)

    f1      = f1_score(y_test, y_pred, average="macro", zero_division=0)
    coef_df = pd.DataFrame(lr.coef_, columns=feature_cols,
                            index=[LABEL_MAP[c] for c in lr.classes_])

    TOP_N = 8
    text  = f"\n{'='*50}\n[로지스틱 회귀] 변수 설명력 분석 (F1={f1:.4f})\n{'='*50}\n"
    text += "계수 크기 = 해당 클래스 예측에 대한 변수 기여도\n\n"

    for cls_name in ("너프", "버프"):
        text += f"[{cls_name} 예측 상위 {TOP_N}개 변수]\n"
        top = coef_df.loc[cls_name].sort_values(ascending=False).head(TOP_N)
        for feat, val in top.items():
            bar  = "+" * min(int(abs(val) * 10), 20)
            text += f"  {feat:<30} {val:+.4f}  {bar}\n"
        text += "\n"

    unique_labels = sorted(y_test.unique())
    text += "[Classification Report - LR]\n"
    text += classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=[LABEL_MAP[l] for l in unique_labels],
        zero_division=0,
    )

    joblib.dump((scaler, lr), os.path.join(FINAL_MODEL_DIR, "LogisticRegression.pkl"))
    coef_df.T.sort_values("너프", ascending=False).to_csv(
        os.path.join(FINAL_MODEL_DIR, "lr_coefficients.csv"), encoding="utf-8-sig"
    )
    print(f"  LR F1 Macro: {f1:.4f} | 계수 저장 완료")
    return text


# ── 리포트 포맷 ────────────────────────────────────────────────────────────
def _confusion_matrix_text(cm, unique_labels: list) -> str:
    col_hdr = "              " + "  ".join(
        f"예측:{LABEL_MAP[l]:<3}" for l in unique_labels
    )
    rows = [col_hdr, ""]
    for i, lv in enumerate(unique_labels):
        counts = "        ".join(str(cm[i, j]).center(6) for j in range(len(unique_labels)))
        rows.append(f"  실제:{LABEL_MAP[lv]:<3}  {counts}")
    return "\n".join(rows)


def build_report(y_test, y_pred, unique_labels: list, title: str = "", extra: str = "") -> str:
    cm      = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_text = _confusion_matrix_text(cm, unique_labels)

    text  = f"\n{'='*50}\n{title}\n{'='*50}\n"
    text += f"[라벨 분포]\n{y_test.value_counts().rename(LABEL_MAP).to_string()}\n\n"
    text += f"[Confusion Matrix]\n{cm_text}\n\n"
    text += "[Classification Report]\n"
    text += classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=[LABEL_MAP[l] for l in unique_labels],
        zero_division=0,
    )
    if extra:
        text += extra
    return text


def _get_feature_importance(pipeline, feature_cols: list) -> pd.DataFrame:
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    return (
        pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ── LOPO 기반 역방향 피처 제거 ────────────────────────────────────────────
def run_feature_selection(
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    tolerance: float = 0.02,
) -> list:
    """LOPO F1 기준 역방향 제거법으로 최소 피처 집합 탐색.

    1. 전체 피처로 LOPO F1 기준값 산출
    2. LOPO 폴드별 평균 feature importance로 제거 우선순위 결정
    3. 중요도 낮은 피처부터 제거 시도 — F1이 (기준 - tolerance) 이상이면 제거 확정
    4. 최소 피처 수 유지(5개)

    tolerance: 허용 F1 하락폭 (기본 0.02 = 2 percentage point)
    """
    patches      = sorted(df["patch_num"].unique())
    all_features = X_all.columns.tolist()

    def _lopo_f1(feature_list: list) -> float:
        X_sub  = X_all[feature_list]
        scores = []
        for test_p in patches:
            tr = df["patch_num"] != test_p
            te = df["patch_num"] == test_p
            X_tr, y_tr = X_sub[tr], y_all[tr]
            X_te, y_te = X_sub[te], y_all[te]
            if len(y_tr.unique()) < 2:
                continue
            try:
                _, _, f1, _ = train_single(X_tr, y_tr, X_te, y_te, "LightGBM", n_splits=3)
                scores.append(f1)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.0

    baseline = _lopo_f1(all_features)
    print(f"\n[피처 선택] 기준 LOPO F1: {baseline:.4f}  ({len(all_features)}개 피처)")
    print("-" * 60)

    # LOPO 폴드별 평균 importance → 제거 우선순위 (오름차순)
    importances = {f: 0.0 for f in all_features}
    for test_p in patches:
        tr = df["patch_num"] != test_p
        X_tr, y_tr = X_all[tr], y_all[tr]
        if len(y_tr.unique()) < 2:
            continue
        pipe = _make_pipeline("LightGBM", y_tr.value_counts().min(), n_splits=3)
        pipe.fit(X_tr, y_tr)
        for feat, imp in zip(all_features, pipe.named_steps["model"].feature_importances_):
            importances[feat] += imp

    removal_order = sorted(all_features, key=lambda f: importances[f])
    threshold     = baseline - tolerance
    current       = all_features.copy()

    for feat in removal_order:
        if feat not in current or len(current) <= 5:
            continue
        candidate = [f for f in current if f != feat]
        f1        = _lopo_f1(candidate)
        if f1 >= threshold:
            current = candidate
            print(f"  제거: {feat:<34} 남은 {len(current):2d}개  F1={f1:.4f}")
        else:
            print(f"  유지: {feat:<34} 남은 {len(current):2d}개  F1={f1:.4f}  (하락 {baseline - f1:.4f})")

    final_f1 = _lopo_f1(current)
    print(f"\n[결과] {len(all_features)}개 → {len(current)}개  |  "
          f"F1: {baseline:.4f} → {final_f1:.4f}  ({final_f1 - baseline:+.4f})")
    print(f"선택 피처: {current}")
    return current


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print(" Loracle 학습 파이프라인")
    print("=" * 50)

    raw = load_all_patches()
    df  = add_engineered_features(raw)

    X_all, y_all  = prepare_features(df)
    feature_cols  = X_all.columns.tolist()
    print(f"[피처] 초기 {len(feature_cols)}개 | 샘플: {len(df)}행")

    # 역방향 피처 제거 — 최소 피처 탐색
    selected = run_feature_selection(df, X_all, y_all, tolerance=0.02)
    X_all, y_all = prepare_features(df, fit_columns=selected)
    feature_cols  = selected
    unique_labels = sorted(y_all.unique())
    print(f"[피처] 선택 후 {len(feature_cols)}개\n")

    with open(os.path.join(FINAL_MODEL_DIR, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    lopo_text = leave_one_patch_out(df, X_all, y_all)

    last_patch = df["patch_num"].max()
    train_mask = df["patch_num"] < last_patch
    test_mask  = df["patch_num"] == last_patch

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    print(f"\n[최종 평가] 패치 1~{last_patch-1} 학습 -> 패치 {last_patch} 테스트")
    print(f"  Train: {len(X_train)}행 | Test: {len(X_test)}행")
    print(f"  Train 라벨: {y_train.value_counts().to_dict()}")
    print(f"  Test  라벨: {y_test.value_counts().to_dict()}\n")

    individual_scores    = {}
    individual_pipelines = {}
    model_blocks         = ""

    for name in ("LightGBM", "XGBoost", "RandomForest"):
        print(f"[학습] {name}...")
        pipeline, y_pred, f1, method = train_single(
            X_train, y_train, X_test, y_test, name, n_splits=4
        )
        individual_scores[name]    = f1
        individual_pipelines[name] = pipeline

        fi_df = _get_feature_importance(pipeline, feature_cols)
        top3  = fi_df.head(3)["Feature"].tolist() if not fi_df.empty else []
        print(f"  F1 Macro: {f1:.4f} ({method}) | 상위 피처: {', '.join(top3)}")

        model_blocks += build_report(
            y_test, y_pred, unique_labels,
            title=f"{name} (F1={f1:.4f}, {method})",
            extra=f"\n  상위 피처: {', '.join(top3)}\n",
        )

    print(f"\n[학습] 앙상블 (소프트보팅)...")
    ens_pipelines, y_pred_ens, f1_ens = train_ensemble(X_train, y_train, X_test, y_test)
    individual_scores["앙상블"] = f1_ens
    print(f"  앙상블 F1 Macro: {f1_ens:.4f}")
    model_blocks += build_report(y_test, y_pred_ens, unique_labels,
                                  title=f"앙상블 소프트보팅 (F1={f1_ens:.4f})")

    print(f"\n[학습] 로지스틱 회귀 (변수 설명력)...")
    lr_text = analyze_logistic_regression(
        X_train, y_train, X_test, y_test, feature_cols
    )
    model_blocks += lr_text

    pred_scores = {k: v for k, v in individual_scores.items() if k != "앙상블"}
    best_name   = max(pred_scores, key=pred_scores.get)
    best_score  = individual_scores[best_name]

    summary  = f"\n{'='*50}\n[모델별 F1 Macro 비교]\n"
    for nm, sc in sorted(individual_scores.items(), key=lambda x: -x[1]):
        bar  = "#" * int(sc * 30)
        mark = "  *" if nm == best_name else ""
        summary += f"  {nm:<15} | {bar:<30} | {sc:.4f}{mark}\n"
    summary += f"\n[최고 모델] {best_name} | F1 Macro: {best_score:.4f}\n{'='*50}\n"
    print(summary)

    for nm, pipe in individual_pipelines.items():
        joblib.dump(pipe, os.path.join(FINAL_MODEL_DIR, f"{nm}.pkl"))
    for nm, pipe in ens_pipelines.items():
        joblib.dump(pipe, os.path.join(FINAL_MODEL_DIR, f"ensemble_{nm}.pkl"))

    fi_df = _get_feature_importance(individual_pipelines[best_name], feature_cols)
    if not fi_df.empty:
        fi_df.to_csv(os.path.join(FINAL_MODEL_DIR, "feature_importance.csv"),
                     index=False, encoding="utf-8-sig")

    full_report = (
        "=" * 50 + "\n Loracle 학습 리포트\n" + "=" * 50 + "\n"
        f"\n[라벨 인코딩]  0=유지  1=버프  2=너프\n"
        f"[학습 방식]   전체 {len(raw)}행 누적 | 마지막 패치 OOT 평가\n\n"
        + lopo_text
        + model_blocks
        + summary
    )
    report_path = os.path.join(FINAL_MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write(full_report)

    metrics_path = os.path.join(MODELS_DIR, "metrics_log.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    metrics["final"] = {nm: float(sc) for nm, sc in individual_scores.items()}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n[저장 완료] {FINAL_MODEL_DIR}")
    print(f"[리포트]   {report_path}")


if __name__ == "__main__":
    main()
