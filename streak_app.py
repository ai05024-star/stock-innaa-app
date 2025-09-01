# filename: streak_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 회사명 자동 조회 모듈 (없으면 간단 대체)
try:
    from ticker_name_api import get_name
except Exception:
    def get_name(t: str) -> str:
        return t

st.set_page_config(page_title="연속 상승/하락 현황", page_icon="📈", layout="wide")
st.title("📈 연속 상승/하락 현황")
st.caption("기준: 패턴 시작 '전날' 종가 대비 누적등락(%). 데이터: Yahoo Finance(일부 지연/누락 가능)")

# ====== 연속 계산: 기준 = 패턴 시작 '전날' 종가 ======
def streak_from_close_baseline_prevday(close: pd.Series, tol: float = 1e-6):
    """
    return:
      - streak, last_close, pct_since_baseline, baseline_date, baseline_price
    """
    close = close.dropna()
    n = close.size
    if n < 2:
        last = float(close.iloc[-1]) if n == 1 else None
        date = close.index[-1] if n == 1 else None
        return 0, last, 0.0, date, last

    a = close.to_numpy()
    idx = close.index.to_numpy()
    diff = a[1:] - a[:-1]
    changes = np.where(diff > tol, 1, np.where(diff < -tol, -1, 0))

    i = len(changes) - 1
    while i >= 0 and changes[i] == 0:
        i -= 1

    last_close = float(a[-1])
    if i < 0:
        return 0, last_close, 0.0, close.index[-1], last_close

    last_sign = changes[i]
    count, j = 0, i
    while j >= 0:
        if changes[j] == 0:
            j -= 1; continue
        if changes[j] == last_sign:
            count += 1; j -= 1
        else:
            break

    start_idx = i - count + 1
    baseline_idx = start_idx
    baseline_price = float(a[baseline_idx])
    baseline_date = pd.Timestamp(idx[baseline_idx])

    pct_since = (last_close / baseline_price - 1.0) * 100.0 if baseline_price > 0 else None
    streak = count if last_sign > 0 else -count
    return streak, last_close, pct_since, baseline_date, baseline_price

# ====== 데이터 가져오기 ======
@st.cache_data(ttl=60*60*6)
def fetch_daily(ticker: str, period: str):
    return yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

def fetch_live_price(ticker: str):
    try:
        info = yf.Ticker(ticker).fast_info
        if hasattr(info, "get"):
            return info.get("last_price", None)
    except Exception:
        pass
    return None

# ====== 기본 설정 ======
DEFAULT_TICKERS = "003490.KS,005930.KS,AAPL"

# (1) 사이드바 설정 — 차트 관련 옵션 제거
with st.sidebar:
    st.header("설정")
    tickers_text = st.text_area("종목들(쉼표로 구분)", value=DEFAULT_TICKERS, height=100)
    period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    min_abs_streak = st.slider("최소 연속일수(절댓값)", 0, 10, 0)

tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

# (2) 데이터 수집
rows, details = [], {}
prog = st.progress(0.0)
for i, t in enumerate(tickers):
    df = fetch_daily(t, period)
    if df.empty or "Close" not in df.columns:
        live_price = fetch_live_price(t)
        rows.append({
            "티커": t, "이름": get_name(t), "연속일수": None, "기준일(전날)": None,
            "기준가": None, "현재가(일봉)": None, "현재가(실시간)": live_price,
            "누적등락(%)": None, "비고": "데이터 없음"
        })
    else:
        streak, last_close, pct, base_date, base_price = streak_from_close_baseline_prevday(df["Close"])
        live_price = fetch_live_price(t)
        rows.append({
            "티커": t,
            "이름": get_name(t),
            "연속일수": streak,
            "기준일(전날)": base_date.date().isoformat() if base_date is not None else None,
            "기준가": base_price,
            "현재가(일봉)": last_close,
            "현재가(실시간)": live_price,
            "누적등락(%)": None if pct is None else round(pct, 2),
            "비고": None
        })
        details[t] = df
    prog.progress((i + 1) / max(1, len(tickers)))

# (3) 요약 테이블
summary = pd.DataFrame(rows)
if not summary.empty and "연속일수" in summary.columns:
    if min_abs_streak > 0:
        summary = summary[summary["연속일수"].abs() >= min_abs_streak]
    summary = summary.sort_values(by="연속일수", key=lambda s: s.abs(), ascending=False, na_position="last")

def _fmt_two(v):
    return None if pd.isna(v) else float(f"{float(v):.2f}")

display_cols = ["티커", "이름", "연속일수", "기준일(전날)", "기준가", "현재가(일봉)", "현재가(실시간)", "누적등락(%)", "비고"]
show_df = summary[[c for c in display_cols if c in summary.columns]].copy()
for col in ["기준가", "현재가(일봉)", "현재가(실시간)"]:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(_fmt_two)
if "누적등락(%)" in show_df.columns:
    show_df["누적등락(%)"] = show_df["누적등락(%)"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else None)

st.subheader("전체 현황")
st.dataframe(show_df, use_container_width=True)

# (4) 검증 테이블 사이드바 (summary 이후)
with st.sidebar:
    st.divider()
    st.subheader("검증 테이블")
    enable_verif = st.checkbox("검증용 테이블 보기", value=False)

    try:
        ok = isinstance(summary, pd.DataFrame) and (not summary.empty) and ("티커" in summary.columns)
    except NameError:
        ok = False

    if ok:
        selectable_tickers = summary["티커"].dropna().astype(str).tolist()
    else:
        selectable_tickers = []

    label_map = {f"{get_name(t)} ({t})": t for t in selectable_tickers}
    selected_labels = (
        st.multiselect("표시할 종목 선택", options=list(label_map.keys()), default=[], placeholder="종목을 선택하세요")
        if (enable_verif and label_map) else []
    )
    selected_tickers = [label_map[l] for l in selected_labels]

# (5) 검증 테이블 본문
if enable_verif and selected_tickers:
    st.subheader("검증용: 최근 10영업일 (전일대비 등락률 %)")
    for t in selected_tickers:
        df_ = details.get(t)
        if df_ is None or df_.empty:
            continue
        tmp = df_[["Close"]].copy()
        tmp["전일대비 등락률(%)"] = tmp["Close"].pct_change() * 100

        name = get_name(t)
        st.write(f"**{name} ({t})**")

        out = tmp.tail(10).copy()
        out["Close"] = out["Close"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        out["전일대비 등락률(%)"] = out["전일대비 등락률(%)"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        st.dataframe(out, use_container_width=True)
elif enable_verif:
    st.info("사이드바에서 종목을 선택하세요.")

st.caption("※ '현재가(실시간)'은 거래소/종목에 따라 지연일 수 있습니다. 제작 : Jeon InHwa")
