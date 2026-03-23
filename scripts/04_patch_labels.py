"""04_patch_labels.py — 패치 노트 크롤링 및 라벨 생성

리그 오브 레전드 공식 패치 노트를 크롤링하여 챔피언별 변경 사항을 분석하고
버프 / 너프 / 조정 라벨을 자동으로 생성.

라벨 인코딩 (05_train.py LABEL_MAP 과 일치):
    0 = 유지  (변경 없음)
    1 = 버프  (능력치 상향)
    2 = 너프  (능력치 하향)
    3 = 조정  (버프·너프 혼재, 학습 시 제외)
"""

import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_champion_list() -> list:
    """DDragon API에서 현재 패치 기준 한국어 챔피언 이름 목록 반환."""
    try:
        latest = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
        data   = requests.get(
            f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/ko_KR/champion.json"
        ).json()["data"]
        return [info["name"] for info in data.values()]
    except Exception as e:
        print(f"[오류] 챔피언 리스트 로드 실패: {e}")
        return []


def scrape_patch_notes(url: str, champion_list: list) -> tuple:
    """패치 노트 URL에서 챔피언별 스탯 변경 텍스트를 추출.

    Returns:
        (stat_rows_df, mentioned_champions_set)
    """
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup     = BeautifulSoup(response.text, "html.parser")

    stat_rows           = []
    mentioned_champions = set()
    in_champion_section = False
    current_champion    = None

    SECTION_END_KEYWORDS = ["아이템", "무작위 총력전", "체계", "버그", "스킨"]

    for tag in soup.find_all(["h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue

        if tag.name == "h2":
            if "챔피언" in text:
                in_champion_section = True
                continue
            if in_champion_section and any(kw in text for kw in SECTION_END_KEYWORDS):
                break

        if not in_champion_section:
            continue

        if tag.name in ["h3", "h4"]:
            first_word = text.split(" ")[0]
            if text in champion_list:
                current_champion = text
                mentioned_champions.add(current_champion)
            elif first_word in champion_list:
                current_champion = first_word
                mentioned_champions.add(current_champion)
            continue

        if current_champion and "⇒" in text:
            stat_rows.append({"champion": current_champion, "raw_stat_text": text})

    df = pd.DataFrame(stat_rows)
    if not df.empty:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, mentioned_champions


def classify_stat_change(line: str) -> int:
    """'이전값 ⇒ 이후값' 형태의 스탯 변경 텍스트를 분석해 라벨 반환.

    Returns:
        1=버프  2=너프  3=조정(혼재)  0=변화 없음
    """
    if "⇒" not in line:
        return 0

    before, after = line.split("⇒", 1)

    # 특수 케이스: "최대 체력 → 추가 체력"은 버프
    if "최대 체력" in before and "추가 체력" in after:
        return 1

    num_pattern = r"[-+]?\d*\.?\d+"
    before_nums = [float(n) for n in re.findall(num_pattern, before)]
    after_nums  = [float(n) for n in re.findall(num_pattern, after)]

    if not before_nums or not after_nums:
        return 3  # 수치 파싱 불가 → 조정

    # 낮을수록 유리한 스탯(쿨타임, 소모 등)
    NEGATIVE_STATS = ["재사용", "소모", "쿨타임", "대기시간", "충전", "지연", "페널티"]
    is_negative = any(kw in before for kw in NEGATIVE_STATS)

    if len(before_nums) == len(after_nums):
        diffs    = [a - b for a, b in zip(after_nums, before_nums)]
        has_buff = any((d < 0 if is_negative else d > 0) for d in diffs)
        has_nerf = any((d > 0 if is_negative else d < 0) for d in diffs)

        if has_buff and has_nerf:
            return 3
        if has_buff:
            return 1
        if has_nerf:
            return 2
        return 0

    before_avg = sum(before_nums) / len(before_nums)
    after_avg  = sum(after_nums)  / len(after_nums)

    if before_avg == after_avg:
        return 0
    increased = after_avg > before_avg
    if is_negative:
        return 1 if not increased else 2
    return 1 if increased else 2


def generate_labels(stat_df: pd.DataFrame, mentioned: set, version: str) -> pd.DataFrame:
    """챔피언별 스탯 변경 텍스트 목록에서 최종 라벨 결정.

    버프 수 > 너프 수 → 버프(1), 너프 수 > 버프 수 → 너프(2), 동수 → 조정(3).
    """
    champ_results = {}
    for _, row in stat_df.iterrows():
        champ  = row["champion"]
        result = classify_stat_change(row["raw_stat_text"])
        champ_results.setdefault(champ, []).append(result)

    records = []
    for champ, results in champ_results.items():
        valid = [r for r in results if r != 0]
        if not valid:
            label = 3
        else:
            buffs = valid.count(1)
            nerfs = valid.count(2)
            if buffs > nerfs:
                label = 1
            elif nerfs > buffs:
                label = 2
            else:
                label = 3
        records.append({"version": version, "champion": champ, "label": label})

    # 스탯 변경 텍스트 없이 언급만 된 챔피언 → 조정
    for champ in mentioned - set(champ_results.keys()):
        records.append({"version": version, "champion": champ, "label": 3})

    df = pd.DataFrame(records)
    return df.sort_values(["label", "champion"]).reset_index(drop=True)


def main():
    # 패치마다 아래 두 값만 수정
    target_url     = "https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-6-notes/"
    target_version = "16.6"

    champ_list = get_champion_list()
    if not champ_list:
        return

    print(f"패치 노트 크롤링 중: {target_url}")
    stat_df, mentioned = scrape_patch_notes(target_url, champ_list)

    if stat_df.empty and not mentioned:
        print("[경고] 챔피언 변경 사항을 찾지 못했습니다. URL 또는 페이지 구조를 확인하세요.")
        return

    labels_df = generate_labels(stat_df, mentioned, target_version)

    version_key = target_version.replace(".", "_")
    os.makedirs("../raw_data/patch_data", exist_ok=True)
    output = f"../raw_data/patch_data/patch_labels_{version_key}.csv"
    labels_df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n[완료] 저장 -> {output}")
    print(f"라벨 분포: {labels_df['label'].value_counts().to_dict()}")
    print("\n[미리보기]")
    print(labels_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
