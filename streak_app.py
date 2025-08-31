from ticker_name_api import get_name
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="연속 상승/하락 현황", page_icon="📈", layout="wide")
st.title("📈 연속 상승/하락 현황")
st.caption("기준: 패턴 시작 '전날' 종가 대비 누적등락(%). 데이터: Yahoo Finance(지연/누락 가능)")

# ====== 핵심 계산 함수 (기준=패턴 시작 전날) ======
def streak_from_close_baseline_prevday(close: pd.Series, tol=1e-6):
    close = close.dropna()
    n = close.size
    if n < 2:
        last = float(close.iloc[-1]) if n==1 else None
        date = close.index[-1] if n==1 else None
        return 0, last, 0.0, date, last

    a = close.to_numpy()
    idx = close.index.to_numpy()
    diff = a[1:] - a[:-1]
    changes = np.where(diff > tol, 1, np.where(diff < -tol, -1, 0))

    # 마지막 비보합 변화 찾기
    i = len(changes) - 1
    while i >= 0 and changes[i] == 0:
        i -= 1

    last_close = float(a[-1])
    if i < 0:
        return 0, last_close, 0.0, close.index[-1], last_close

    last_sign = changes[i]
    # 연속 카운트
    count, j = 0, i
    while j >= 0:
        if changes[j] == 0:
            j -= 1; continue
        if changes[j] == last_sign:
            count += 1; j -= 1
        else:
            break

    start_idx = i - count + 1
    baseline_idx = start_idx               # 시작 '전날' 종가 위치
    baseline_price = float(a[baseline_idx])
    baseline_date  = pd.Timestamp(idx[baseline_idx])

    pct_since = (last_close / baseline_price - 1.0) * 100.0 if baseline_price > 0 else None
    streak = count if last_sign > 0 else -count
    return streak, last_close, pct_since, baseline_date, baseline_price

# ====== UI ======
DEFAULT_TICKERS = "003490.KS,005930.KS,AAPL"
with st.sidebar:
    st.header("설정")
    tickers_text = st.text_area("종목들(쉼표로 구분)", value=DEFAULT_TICKERS, height=100)
    period = st.selectbox("조회 기간", ["1mo","3mo","6mo","1y"], index=1)
    min_abs_streak = st.slider("최소 연속일수(절댓값)", 0, 10, 0)
    show_charts = st.checkbox("최근 종가 차트 보기", value=False)

# ---- 요약 테이블(summary)까지 기존 코드 그대로 진행 ----

# ---- 검증 테이블을 위한 추가 사이드바 ----
with st.sidebar:
    st.divider()
    st.subheader("검증 테이블")
    enable_verif = st.checkbox("검증용 테이블 보기", value=False)
    selectable_tickers = list(summary["티커"]) if not summary.empty else []
    # 라벨을 '이름 (티커)'로 변환
    label_map = {}
    for t in selectable_tickers:
        try:
            name = get_name(t)
        except Exception:
            name = t
        label_map[f"{name} ({t})"] = t

    selected_labels = []
    if enable_verif and selectable_tickers:
        selected_labels = st.multiselect(
            "표시할 종목 선택",
            options=list(label_map.keys()),
            default=[],
            placeholder="종목을 선택하세요"
        )
    selected_tickers = [label_map[l] for l in selected_labels]

tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
rows, details = [], {}

@st.cache_data(ttl=60*60*6)
def fetch(ticker, period):
    return yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

prog = st.progress(0.0)
for i, t in enumerate(tickers):
    df = fetch(t, period)
    if df.empty or "Close" not in df.columns:
        rows.append({"티커": t, "연속일수": None, "기준일(전날)": None, "기준가": None,
                     "현재가": None, "누적등락(%)": None, "비고": "데이터 없음"})
    else:
        streak, last_close, pct, base_date, base_price = streak_from_close_baseline_prevday(df["Close"])
        rows.append({
            "티커": t,
            "이름": get_name(t),
            "연속일수": streak,
            "기준일(전날)": base_date.date().isoformat() if base_date is not None else None,
            "기준가": base_price,
            "현재가": last_close,
            "누적등락(%)": None if pct is None else round(pct,2),
            "비고": None
        })
        details[t] = df
    prog.progress((i+1)/max(1,len(tickers)))

summary = pd.DataFrame(rows)
if not summary.empty and "연속일수" in summary.columns:
    if min_abs_streak>0:
        summary = summary[summary["연속일수"].abs()>=min_abs_streak]
    summary = summary.sort_values(by="연속일수", key=lambda s: s.abs(), ascending=False, na_position="last")

st.subheader("전체 현황")
st.dataframe(summary, use_container_width=True)

if show_charts and not summary.empty:
    st.subheader("최근 종가 차트")
    for t in summary["티커"]:
        df = details.get(t)
        if df is not None and not df.empty:
            st.line_chart(df["Close"].dropna(), height=180)
            st.caption(f"{t} 최근 종가 ({period})")

if enable_verif and selected_tickers:
    st.subheader("검증용: 최근 10영업일 (전일대비 등락률 %)")
    for t in selected_tickers:
        df = details.get(t)
        if df is None or df.empty:
            continue

        tmp = df[["Close"]].copy()
        tmp["전일대비 등락률(%)"] = tmp["Close"].pct_change() * 100

        # 제목: 회사명 (티커)
        name = get_name(t)
        st.write(f"**{name} ({t})**")

        # 포맷: 둘째 자리까지
        out = tmp.tail(10).copy()
        out["Close"] = out["Close"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        out["전일대비 등락률(%)"] = out["전일대비 등락률(%)"].map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else ""
        )
        st.dataframe(out, use_container_width=True)

elif enable_verif:
    # 토글은 켰지만 아직 선택한 종목이 없을 때
    st.info("사이드바에서 종목을 선택하세요.")
        
st.caption("※ 무료 데이터 특성상 지연/누락 가능. 투자 판단은 본인 책임입니다. 제작 : 전인화")


