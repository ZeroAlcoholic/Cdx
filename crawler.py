"""新聞爬取與去重流程（合法來源假設由設定提供）。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable, List, Dict

import pandas as pd


# 去重規則：以 normalized title + published_at 產生 content_hash，再去除重複。
def _content_hash(title: str, summary: str, published_at: str) -> str:
    payload = f"{title.strip().lower()}|{summary.strip().lower()}|{published_at}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_news_records(records: Iterable[Dict[str, str]], source: str) -> pd.DataFrame:
    """將來源新聞轉為 `news_raw` 結構，保留來源、發佈時間、抓取時間。"""
    now = datetime.now(timezone.utc).isoformat()
    rows: List[Dict[str, str]] = []
    for rec in records:
        pub = pd.to_datetime(rec["published_at"], utc=True).isoformat()
        c_hash = _content_hash(rec.get("title", ""), rec.get("summary", ""), pub)
        rows.append(
            {
                "news_id": c_hash,
                "source": source,
                "title": rec.get("title", ""),
                "summary": rec.get("summary", ""),
                "url": rec.get("url", ""),
                "published_at": pub,
                "fetched_at": now,
                "content_hash": c_hash,
            }
        )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["content_hash", "published_at"], keep="first").reset_index(drop=True)


def merge_news_raw(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """合併舊資料與新抓取結果，確保可重跑且結果一致。"""
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values("published_at").drop_duplicates(subset=["content_hash", "published_at"], keep="first")
    return merged.reset_index(drop=True)


def scheduled_ingest(source_name: str, records: Iterable[Dict[str, str]], existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """提供可排程呼叫的單次擷取流程入口。"""
    existing_df = existing if existing is not None else pd.DataFrame()
    normalized = normalize_news_records(records=records, source=source_name)
    if existing_df.empty:
        return normalized
    return merge_news_raw(existing_df, normalized)
