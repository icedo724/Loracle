"""06_report.py — 최종 모델 성능 요약 출력

models/final/ 에 저장된 결과물을 읽어 콘솔에 요약을 출력하고
models/summary_report.txt 로 저장.

라벨 인코딩: 0=유지  1=버프  2=너프  (04_patch_labels.py / 05_train.py 기준)
"""

import os
import re

import pandas as pd

FINAL_DIR  = "../models/final"
MODELS_DIR = "../models"
REPORT_IN  = os.path.join(FINAL_DIR, "evaluation_report.txt")
REPORT_OUT = os.path.join(MODELS_DIR, "summary_report.txt")
LABEL_MAP  = {0: "유지", 1: "버프", 2: "너프"}


def make_bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val == 0:
        return ""
    ratio = min(value / max_val, 1.0)
    return "#" * int(ratio * width)


def parse_report(text: str) -> dict:
    """evaluation_report.txt 에서 핵심 수치 파싱."""
    result = {
        "lopo_mean":     None,
        "lopo_std":      None,
        "lopo_patches":  {},
        "model_scores":  {},
        "best_model":    None,
        "best_f1":       None,
        "class_metrics": {},
    }

    m = re.search(r"평균 F1 Macro:\s*([\d.]+)\s*\+-\s*([\d.]+)", text)
    if m:
        result["lopo_mean"] = float(m.group(1))
        result["lopo_std"]  = float(m.group(2))

    for m in re.finditer(r"patch_(\d+):\s*Best=(\w+)\s+F1=([\d.]+)", text):
        result["lopo_patches"][int(m.group(1))] = {
            "best": m.group(2),
            "f1":   float(m.group(3)),
        }

    for m in re.finditer(r"^\s{2}(\w+)\s+\|\s+[#]+\s+\|\s+([\d.]+)", text, re.MULTILINE):
        result["model_scores"][m.group(1)] = float(m.group(2))

    m = re.search(r"\[최고 모델\]\s+(\S+)\s+\|\s+F1 Macro:\s+([\d.]+)", text)
    if m:
        result["best_model"] = m.group(1)
        result["best_f1"]    = float(m.group(2))

    if result["best_model"]:
        name    = re.escape(result["best_model"])
        # 리포트 블록 형식: "\n====\n{name} (F1=...)\n====\n...내용..."
        block_m = re.search(rf"^{name}\b[^\n]*\n=+\n(.*?)(?=\n=+\n|\Z)", text, re.DOTALL | re.MULTILINE)
        if block_m:
            block = block_m.group(1)
            for lm in re.finditer(
                r"^\s*(유지|버프|너프)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
                block, re.MULTILINE,
            ):
                result["class_metrics"][lm.group(1)] = {
                    "precision": float(lm.group(2)),
                    "recall":    float(lm.group(3)),
                    "f1":        float(lm.group(4)),
                    "support":   int(lm.group(5)),
                }

    return result


def generate_summary():
    if not os.path.exists(REPORT_IN):
        print(f"[오류] {REPORT_IN} 없음. 먼저 05_train.py를 실행하세요.")
        return

    with open(REPORT_IN, "r", encoding="utf-8-sig") as f:
        text = f.read()

    parsed = parse_report(text)

    fi_df = None
    fi_path = os.path.join(FINAL_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)

    lr_df = None
    lr_path = os.path.join(FINAL_DIR, "lr_coefficients.csv")
    if os.path.exists(lr_path):
        lr_df = pd.read_csv(lr_path, index_col=0)

    sep   = "=" * 60
    lines = [sep, "  Loracle 최종 모델 성능 요약", sep, ""]

    lines.append("[1] 패치 단위 교차검증 결과")
    lines.append("-" * 40)
    for patch_n, info in sorted(parsed["lopo_patches"].items()):
        bar = make_bar(info["f1"], 1.0, width=20)
        lines.append(f"  패치 {patch_n}  | {bar:<20} | F1={info['f1']:.4f}  [{info['best']}]")
    if parsed["lopo_mean"] is not None:
        lines.append(f"\n  평균 F1 Macro: {parsed['lopo_mean']:.4f} +- {parsed['lopo_std']:.4f}")
    lines.append("")

    lines.append("[2] 모델별 F1 Macro (최종 OOT 평가)")
    lines.append("-" * 40)
    if parsed["model_scores"]:
        max_f1 = max(parsed["model_scores"].values())
        for nm, sc in sorted(parsed["model_scores"].items(), key=lambda x: -x[1]):
            bar  = make_bar(sc, max_f1, width=25)
            mark = "  * 최고" if nm == parsed["best_model"] else ""
            lines.append(f"  {nm:<15} | {bar:<25} | {sc:.4f}{mark}")
    lines.append("")

    best = parsed["best_model"] or "?"
    lines.append(f"[3] 클래스별 성능 (최고 모델: {best})")
    lines.append("-" * 40)
    cm = parsed["class_metrics"]
    if cm:
        lines.append(
            f"  {'클래스':<6} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Support':>7}"
        )
        lines.append(f"  {'─'*6}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*7}")
        for label in ["유지", "버프", "너프"]:
            m_data = cm.get(label)
            if m_data:
                warn = " [!]" if m_data["recall"] < 0.5 and label in ("버프", "너프") else ""
                lines.append(
                    f"  {label:<6} | {m_data['precision']:>10.4f} | {m_data['recall']:>8.4f} |"
                    f" {m_data['f1']:>8.4f} | {m_data['support']:>7}{warn}"
                )
        lines.append("  [!] = 소수 클래스 recall 0.5 미만")
    else:
        lines.append("  [파싱 실패] classification_report 블록을 찾지 못했습니다.")
    lines.append("")

    lines.append("[4] Feature Importance 상위 10개")
    lines.append("-" * 40)
    if fi_df is not None and not fi_df.empty:
        top10   = fi_df.head(10)
        max_imp = top10["Importance"].max()
        for _, row in top10.iterrows():
            bar = make_bar(row["Importance"], max_imp, width=20)
            lines.append(f"  {row['Feature']:<30} {bar:<20} {row['Importance']:.4f}")
    else:
        lines.append("  feature_importance.csv 없음")
    lines.append("")

    lines.append("[5] 로지스틱 회귀 계수 (변수 설명력)")
    lines.append("-" * 40)
    if lr_df is not None and not lr_df.empty:
        for cls_name in ("너프", "버프"):
            if cls_name in lr_df.columns:
                lines.append(f"  [{cls_name} 예측 상위 5개 변수]")
                top5 = lr_df[cls_name].sort_values(ascending=False).head(5)
                for feat, val in top5.items():
                    bar = "+" * min(int(abs(val) * 10), 20)
                    lines.append(f"    {feat:<30} {val:+.4f}  {bar}")
                lines.append("")
    else:
        lines.append("  lr_coefficients.csv 없음")
    lines.append("")
    lines.append(sep)

    output = "\n".join(lines)
    print(output)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8-sig") as f:
        f.write(output)
    print(f"\n[저장] -> {REPORT_OUT}")


if __name__ == "__main__":
    generate_summary()
