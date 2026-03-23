"""03_preprocess.py — 매치 데이터 전처리 및 피처 생성

raw_data/ 의 매치 CSV와 patch_data/ 의 라벨 CSV를 읽어
챔피언별 통계 피처와 패치 라벨을 결합한 학습용 데이터셋을 생성.
"""

import ast
import os
import re

import numpy as np
import pandas as pd
import requests
from scipy.stats import entropy as scipy_entropy


def get_available_versions() -> list:
    """raw_data/ 에서 수집된 패치 버전 목록 반환."""
    raw_data_dir = "../raw_data"
    if not os.path.exists(raw_data_dir):
        print(f"[오류] {raw_data_dir} 폴더가 없습니다.")
        return []

    versions = []
    for filename in os.listdir(raw_data_dir):
        match = re.match(r"loracle_matches_v(.+)\.csv", filename)
        if match:
            versions.append(match.group(1))
    return sorted(versions)


def get_champion_mappings() -> tuple:
    """DDragon API에서 챔피언 이름/ID 매핑 데이터 로드. (영어→한글, ID→한글)"""
    print("DDragon API에서 챔피언 매핑 데이터를 불러옵니다...")
    latest_ver = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
    champ_data = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{latest_ver}/data/ko_KR/champion.json"
    ).json()["data"]

    id_to_name, eng_to_kr = {}, {}
    for eng_name, info in champ_data.items():
        id_to_name[int(info["key"])] = info["name"]
        eng_to_kr[info["id"]] = info["name"]
    return id_to_name, eng_to_kr


def extract_stats(filepath: str, id_to_name: dict, eng_to_kr: dict) -> pd.DataFrame:
    """매치 CSV에서 챔피언별 통계(승률, 픽률, 밴률, OP지수 등)를 집계."""
    df = pd.read_csv(filepath)
    total_games = df["match_id"].nunique()

    valid_positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    df_valid = df[df["position"].isin(valid_positions)].copy()
    df_valid["champion_kr"] = df_valid["champion"].map(eng_to_kr).fillna(df_valid["champion"])

    # 기본 통계
    stats = df_valid.groupby("champion_kr").agg(
        pick_count=("win", "count"),
        win_count=("win", "sum"),
        primary_position=("position", lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN"),
    ).reset_index().rename(columns={"champion_kr": "champion"})

    stats["win_rate"]  = (stats["win_count"] / stats["pick_count"]).round(4)
    # 픽률/밴률 분모: 총 게임 수 × 10명
    stats["pick_rate"] = (stats["pick_count"] / (total_games * 10)).round(4)

    # 밴률
    ban_counts = {}
    for bans_str in df.drop_duplicates("match_id")["bans"].dropna():
        try:
            for cid in ast.literal_eval(bans_str):
                name = id_to_name.get(cid)
                if name:
                    ban_counts[name] = ban_counts.get(name, 0) + 1
        except Exception:
            continue

    ban_df = pd.DataFrame(list(ban_counts.items()), columns=["champion", "ban_count"])
    stats = stats.merge(ban_df, on="champion", how="left")
    stats["ban_count"] = stats["ban_count"].fillna(0)
    stats["ban_rate"]  = (stats["ban_count"] / (total_games * 10)).round(4)

    # 존재감(픽+밴), OP지수(승률×픽률), 밴압박(밴률×승률)
    stats["presence_rate"] = (stats["pick_rate"] + stats["ban_rate"]).round(4)
    stats["op_index"]      = (stats["win_rate"] * stats["pick_rate"]).round(6)
    stats["ban_pressure"]  = (stats["ban_rate"] * stats["win_rate"]).round(6)

    # 포지션 분산도 (높을수록 멀티포지션 챔피언)
    def calc_entropy(grp):
        counts = grp["position"].value_counts(normalize=True)
        return float(scipy_entropy(counts))

    pos_ent = (
        df_valid.groupby("champion_kr")
        .apply(calc_entropy)
        .reset_index()
    )
    pos_ent.columns = ["champion", "position_entropy"]
    stats = stats.merge(pos_ent, on="champion", how="left")
    stats["position_entropy"] = stats["position_entropy"].fillna(0).round(4)

    # 포지션별 세분화 승률
    pos_win = (
        df_valid.groupby(["champion_kr", "position"])["win"]
        .mean()
        .unstack(fill_value=None)
        .round(4)
    )
    pos_win.columns = [f"win_rate_{c.lower()}" for c in pos_win.columns]
    pos_win = pos_win.reset_index().rename(columns={"champion_kr": "champion"})
    stats = stats.merge(pos_win, on="champion", how="left")

    # 최다 픽 코어 아이템 조합(첫 두 아이템) 승률
    item_records = []
    for _, row in df_valid.iterrows():
        try:
            item_list = ast.literal_eval(row["items"])
            if len(item_list) >= 2:
                core_combo = str(tuple(sorted(item_list[:2])))
                item_records.append({
                    "champion":   row["champion_kr"],
                    "core_combo": core_combo,
                    "win":        row["win"],
                })
        except Exception:
            continue

    if item_records:
        item_df     = pd.DataFrame(item_records)
        combo_stats = item_df.groupby(["champion", "core_combo"]).agg(
            combo_count=("win", "count"),
            combo_wins=("win", "sum"),
        ).reset_index()
        combo_stats["combo_win_rate"] = (
            combo_stats["combo_wins"] / combo_stats["combo_count"]
        ).round(4)

        top_combo = (
            combo_stats.sort_values("combo_count", ascending=False)
            .groupby("champion").first()
            .reset_index()
            [["champion", "core_combo", "combo_win_rate"]]
            .rename(columns={"core_combo": "top_item_combo"})
        )
        stats = stats.merge(top_combo, on="champion", how="left")

    return stats


def main():
    available_versions = get_available_versions()
    if not available_versions:
        return

    print("\n" + "=" * 50)
    print("Loracle 데이터 전처리")
    print("=" * 50)
    print(f"사용 가능 버전: {', '.join(available_versions)}")

    FEATURE_PATCH = input("학습 바탕 패치 [N-1] (예: 16_3): ").strip()
    PREV_PATCH    = input("Delta 계산 패치 [N-2] (없으면 엔터): ").strip()
    LABEL_PATCH   = input("예측 정답지 패치 [N]  (예: 16_4): ").strip()

    id_to_name, eng_to_kr = get_champion_mappings()

    print(f"\n[1/3] {FEATURE_PATCH} 패치 통계 추출 중...")
    stats_current = extract_stats(
        f"../raw_data/loracle_matches_v{FEATURE_PATCH}.csv",
        id_to_name, eng_to_kr,
    )

    DELTA_COLS = ["win_rate", "pick_rate", "ban_rate", "presence_rate", "op_index", "ban_pressure"]

    if PREV_PATCH and PREV_PATCH in available_versions:
        print(f"[2/3] {PREV_PATCH} 대비 Delta 계산 중...")
        stats_prev = extract_stats(
            f"../raw_data/loracle_matches_v{PREV_PATCH}.csv",
            id_to_name, eng_to_kr,
        )[["champion"] + DELTA_COLS]

        merged = stats_current.merge(stats_prev, on="champion", how="left", suffixes=("", "_prev"))
        for col in DELTA_COLS:
            prev_col = f"{col}_prev"
            # 신규 챔피언(이전 패치 미등장)은 변화량 0으로 처리
            merged[f"{col}_delta"] = (
                merged[col] - merged[prev_col].fillna(merged[col])
            ).round(4)
            merged.drop(columns=[prev_col], inplace=True)
        final_stats = merged
    else:
        print("[2/3] 과거 패치 미지정 -> Delta 0으로 초기화")
        final_stats = stats_current.copy()
        for col in DELTA_COLS:
            final_stats[f"{col}_delta"] = 0.0

    print(f"[3/3] {LABEL_PATCH} 라벨 병합 중...")
    label_file = f"../raw_data/patch_data/patch_labels_{LABEL_PATCH}.csv"
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        final_stats = final_stats.merge(
            labels_df[["champion", "label"]], on="champion", how="left"
        )
        final_stats["label"] = final_stats["label"].fillna(0).astype(int)
    else:
        print(f"  [경고] 라벨 파일 없음 -> label=0 으로 초기화")
        final_stats["label"] = 0

    # pick_count_log: 샘플 신뢰도 (로그 스케일)
    if "pick_count" in final_stats.columns:
        final_stats["pick_count_log"] = np.log1p(final_stats["pick_count"]).round(4)
    final_stats = final_stats.sort_values(
        ["label", "win_rate"], ascending=[False, False]
    ).drop(columns=["pick_count", "win_count", "ban_count"], errors="ignore")

    OUTPUT = f"../preprocessed/ml_dataset_{FEATURE_PATCH}_to_{LABEL_PATCH}.csv"
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    final_stats.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\n[완료] -> {OUTPUT}")
    print(f"피처 수: {len(final_stats.columns) - 1}개 | 챔피언 수: {len(final_stats)}명")
    print("\n[미리보기]")
    print(final_stats.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
