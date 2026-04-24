"""01_collect_users.py — 최상위권 유저 PUUID 수집

Riot API에서 Challenger / Grandmaster / Master 티어 플레이어의
PUUID 목록을 수집하여 CSV로 저장.
"""

import os
from pathlib import Path

import pandas as pd
import requests

BASE_DIR     = Path(__file__).resolve().parent.parent
API_FILE     = BASE_DIR / "default_info" / "api.txt"
OUTPUT_FILE  = BASE_DIR / "raw_data" / "target_users.csv"
TARGET_TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER"]


def fetch_tier_users(tier: str, headers: dict) -> list:
    url = (
        f"https://kr.api.riotgames.com/lol/league/v4/"
        f"{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5"
    )
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"[{tier}] API 에러: {r.status_code}")
            return []
        entries = r.json().get("entries", [])
        users = [
            {
                "tier":           tier,
                "summoner_name":  e.get("summonerName", "Unknown"),
                "puuid":          e["puuid"],
            }
            for e in entries
            if "puuid" in e
        ]
        print(f"[{tier}] {len(users)}명 확보")
        return users
    except Exception as e:
        print(f"[{tier}] 요청 실패: {e}")
        return []


def main():
    with open(API_FILE, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    headers = {"X-Riot-Token": api_key}

    print("최상위권 유저 명단 수집을 시작합니다...")
    all_users = []
    for tier in TARGET_TIERS:
        all_users.extend(fetch_tier_users(tier, headers))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(all_users)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 총 {len(df)}명 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
