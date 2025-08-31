from functools import lru_cache
from typing import Iterable, Dict, Optional
import yfinance as yf
import time

def _fetch_name_once(ticker: str, timeout_sec: float = 5.0) -> Optional[str]:
    """
    yfinance를 통해 회사명(longName/shortName)을 1회 조회.
    실패/없음이면 None 반환.
    """
    try:
        t = yf.Ticker(ticker)
        # 최신 yfinance는 get_info() 권장
        info = t.get_info()
        if not isinstance(info, dict):
            return None
        name = info.get("longName") or info.get("shortName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        # 야후 응답 변동/레이트리밋/네트워크 이슈 등은 조용히 무시하고 None
        return None
    return None

@lru_cache(maxsize=2048)
def get_name(ticker: str, fallback_to_ticker: bool = True, retries: int = 2, backoff_sec: float = 0.8) -> str:
    """
    단일 티커의 회사명을 반환.
    - yfinance API로 자동 조회 (longName > shortName)
    - 실패 시 None -> fallback_to_ticker=True면 티커 그대로 반환
    - 간단한 재시도(retries, backoff) 내장
    """
    if not ticker:
        return ""
    key = ticker.strip()

    name = _fetch_name_once(key)
    attempt = 0
    while name is None and attempt < retries:
        attempt += 1
        time.sleep(backoff_sec * attempt)
        name = _fetch_name_once(key)

    if name:
        return name
    return key if fallback_to_ticker else ""

def get_names_bulk(tickers: Iterable[str], fallback_to_ticker: bool = True) -> Dict[str, str]:
    """
    여러 티커를 한 번에 조회해서 {ticker: name} dict 반환.
    내부적으로 get_name(캐시 사용)을 호출하므로 반복 사용에 효율적.
    """
    result: Dict[str, str] = {}
    for t in tickers:
        result[t] = get_name(t, fallback_to_ticker=fallback_to_ticker)
    return result