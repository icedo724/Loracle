"""02_collect_matches.py — 패치 매치 데이터 수집

현재 패치(자동 감지) 또는 특정 과거 패치의 랭크 솔로 매치 데이터를 수집.
이어하기(incremental) 지원.

실행 시 현재 패치 여부를 선택:
    y — 최신 패치 자동 감지, 이전 패치 도달 시 즉시 종료
    n — 패치 버전 직접 입력, PATCH_SCHEDULE 날짜 범위로 필터링
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests

from utils import load_api_key

API_FILE   = "../default_info/api.txt"
USERS_FILE = "../raw_data/target_users.csv"
QUEUE_ID   = 420  # 랭크 솔로/듀오

# 새 패치 추가 시 날짜 범위 등록
PATCH_SCHEDULE = {
    "16_1": ("2026-01-07", "2026-01-21"),
    "16_2": ("2026-01-21", "2026-02-03"),
    "16_3": ("2026-02-03", "2026-02-18"),
    "16_4": ("2026-02-18", "2026-03-03"),
    "16_5": ("2026-03-04", "2026-03-17"),
}


def handle_rate_limit(response) -> bool:
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", 10))
        print(f"\n[대기] API 호출 한도 초과. {wait}초 대기...")
        time.sleep(wait)
        return True
    return False


def date_to_timestamp(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def get_latest_patch() -> tuple:
    latest = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
    major, minor = map(int, latest.split(".")[:2])
    return major, minor


def extract_match_rows(data: dict, match_id: str) -> list:
    banned_ids = [
        ban["championId"]
        for team in data["info"]["teams"]
        for ban in team["bans"]
        if ban["championId"] != -1
    ]
    rows = []
    for p in data["info"]["participants"]:
        items = [p[f"item{i}"] for i in range(6) if p[f"item{i}"] != 0]
        rows.append({
            "match_id":     match_id,
            "game_version": data["info"].get("gameVersion", ""),
            "champion":     p["championName"],
            "position":     p["teamPosition"],
            "items":        str(items),
            "bans":         str(banned_ids),
            "win":          1 if p["win"] else 0,
        })
    return rows


def collect_user_matches(
    puuid: str,
    headers: dict,
    t_major: int,
    t_minor: int,
    skip_ids: set,
    output_file: str,
    start_ts: int = None,
    end_ts: int = None,
    early_stop: bool = False,
) -> set:
    """단일 유저의 대상 패치 매치를 수집하고 skip_ids를 갱신해 반환.

    early_stop=True: 현재 패치 수집 모드. 이전 패치 도달 시 즉시 탐색 중단.
    start_ts/end_ts:  과거 패치 수집 모드. API 날짜 필터 적용 후 버전 재검증.
    """
    start_index = 0
    count = 100 if (start_ts and end_ts) else 20

    while True:
        url = (
            f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid"
            f"/{puuid}/ids?queue={QUEUE_ID}&start={start_index}&count={count}"
        )
        if start_ts and end_ts:
            url += f"&startTime={start_ts}&endTime={end_ts}"

        r = requests.get(url, headers=headers)
        if handle_rate_limit(r):
            continue
        if r.status_code != 200:
            break

        match_ids = r.json()
        if not match_ids:
            break

        batch     = []
        stop_user = False

        for match_id in match_ids:
            if match_id in skip_ids:
                continue

            r_d = requests.get(
                f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}",
                headers=headers,
            )
            if handle_rate_limit(r_d):
                time.sleep(1.2)
                continue
            if r_d.status_code != 200:
                continue

            data        = r_d.json()
            version_str = data["info"].get("gameVersion", "")
            try:
                m_major, m_minor = map(int, version_str.split(".")[:2])
            except (ValueError, AttributeError):
                continue

            # 현재 패치 모드: 이전 패치 도달 시 이 유저 탐색 종료
            if early_stop and (
                m_major < t_major or (m_major == t_major and m_minor < t_minor)
            ):
                stop_user = True
                break

            if m_major == t_major and m_minor == t_minor:
                batch.extend(extract_match_rows(data, match_id))
                skip_ids.add(match_id)
                print(f"  -> {match_id} ({t_major}.{t_minor}) 추출", end="\r")

            time.sleep(1.2)

        if batch:
            df_new = pd.DataFrame(batch)
            mode   = "w" if not os.path.exists(output_file) else "a"
            df_new.to_csv(
                output_file, index=False, encoding="utf-8-sig",
                mode=mode, header=(mode == "w"),
            )

        if stop_user:
            break
        start_index += count

    return skip_ids


def main():
    api_key = load_api_key(API_FILE)
    headers = {"X-Riot-Token": api_key}

    c_major, c_minor = get_latest_patch()
    print("=" * 50)
    print(f"현재 라이엇 본섭 최신 패치: {c_major}.{c_minor}")
    print("=" * 50)

    mode_input = input("현재 패치 수집? (y=현재 / n=특정 패치 지정): ").strip().lower()

    if mode_input == "y":
        t_major, t_minor = c_major, c_minor
        start_ts = end_ts = None
        early_stop = True
        print(f"수집 대상 패치: {t_major}.{t_minor}")
    else:
        raw = input("수집할 패치 버전을 입력하세요 (예: 16_1): ").strip().replace(".", "_")
        try:
            t_major, t_minor = map(int, raw.split("_"))
        except ValueError:
            print("[오류] 올바른 패치 버전 형식이 아닙니다. (예: 16_1)")
            return

        version_key = f"{t_major}_{t_minor}"
        early_stop  = False
        start_ts = end_ts = None
        print(f"\n수집 대상 패치: {t_major}.{t_minor}")

        if version_key in PATCH_SCHEDULE:
            start_date, end_date = PATCH_SCHEDULE[version_key]
            start_ts = date_to_timestamp(start_date)
            end_ts   = date_to_timestamp(end_date)
            print(f"기간 필터링 적용: {start_date} ~ {end_date}")
        else:
            print(f"{version_key} 패치 일정이 등록되어 있지 않습니다.")
            manual_start = input("시작 날짜 (예: 2026-01-07, 모르면 엔터): ").strip()
            manual_end   = input("종료 날짜 (예: 2026-01-21, 모르면 엔터): ").strip()
            if manual_start and manual_end:
                start_ts = date_to_timestamp(manual_start)
                end_ts   = date_to_timestamp(manual_end)
                print("수동 기간 필터링 적용 완료.")
            else:
                print("[경고] 기간 미지정. 최근 게임부터 역탐색합니다.")

    version_key = f"{t_major}_{t_minor}"
    output_file = f"../raw_data/loracle_matches_v{version_key}.csv"
    log_file    = f"../raw_data/user_progress_log_{version_key}.txt"

    processed_users = set()
    if os.path.exists(log_file):
        resume = input("\n기존 수집 로그가 있습니다. 이어서 작업하시겠습니까? (y/n): ").strip().lower()
        if resume == "y":
            with open(log_file, "r", encoding="utf-8") as f:
                processed_users = {int(line.strip()) for line in f if line.strip().isdigit()}
            print(f"{len(processed_users)}명 스킵. 이어서 수집 시작.")
        else:
            os.remove(log_file)
            if os.path.exists(output_file):
                os.remove(output_file)
            print("로그 및 데이터 파일 초기화.")

    skip_ids = set()
    if os.path.exists(output_file):
        try:
            skip_ids = set(pd.read_csv(output_file, usecols=["match_id"])["match_id"].unique())
        except Exception:
            pass

    df_users    = pd.read_csv(USERS_FILE)
    total_users = len(df_users)

    for idx, row in df_users.iterrows():
        if idx in processed_users:
            continue
        print(f"\n[{idx + 1}/{total_users}] {row['tier']} 유저 탐색 중...")
        skip_ids = collect_user_matches(
            puuid=row["puuid"],
            headers=headers,
            t_major=t_major,
            t_minor=t_minor,
            skip_ids=skip_ids,
            output_file=output_file,
            start_ts=start_ts,
            end_ts=end_ts,
            early_stop=early_stop,
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{idx}\n")

    print(f"\n[완료] {t_major}.{t_minor} 패치 매치 데이터 수집 완료.")


if __name__ == "__main__":
    main()
