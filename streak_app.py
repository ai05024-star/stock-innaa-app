import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# (회사명 자동 조회 모듈) 없으면 간단 대체함수로 대체 가능
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
    close: 종가 Series (DatetimeIndex 권장)
    tol  : 보합 허용오차 (전일대비 절댓값이 tol 이하면 보합=0)
    return:
      - streak               : 양수=연속 상승일수, 음수=연속 하락일수, 0=판정불가/전구간 보합
      - last_close           : 마지막 종가(float)
      - pct_since_baseline   : '연속 시작 전날' 종가 대비 누적 등락률(%)
      - baseline_date        : '연속 시작 전날' 날짜 (Timestamp)
      - baseline_price       : '연속 시작 전날' 종가 (float)
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
    changes = np.where(diff > tol, 1, np.where(diff < -tol, -1, 0))  # +1/-1/0

    # 뒤에서부터 보합 제거 → 마지막 비보합 변화 인덱스 i
    i = len(changes) - 1
    while i >= 0 and changes[i] == 0:
        i -= 1

    last_close = float(a[-1])
    if i < 0:
        # 전 기간 보합
        return 0, last_close, 0.0, close.index[-1], last_close

    last_sign = changes[i]
    # 현재 연속 길이 count
    count, j = 0, i
    while j >= 0:
        if changes[j] == 0:
            j -= 1
            continue
        if changes[j] == last_sign:
            count += 1
            j -= 1
        else:
            break

    # 현재 연속구간의 시작 변화 인덱스
    start_idx = i - count + 1
    # 기준=시작 '전날' 종가
    baseline_idx = start_idx
    baseline_price = float(a[baseline_idx])
    baseline_date = pd.Timestamp(idx[baseline_idx])

    pct_since = (last_close / baseline_price - 1.0) * 100.0 if baseline_price > 0 else None
    streak = count if last_sign > 0 else -count
    return streak, last_close, pct_since, baseline_date, baseline_price

# ====== 캐시된 일봉 데이터 가져오기 ======
@st.cache_data(ttl=60*60*6)
def fetch_daily(ticker: str, period: str):
    return yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

# ====== 현재가(지연/실시간) 조회 ======
def fetch_live_price(ticker: str):
    """
    yfinance.fast_info에서 현재가(보통 지연)를 가져옵니다.
    실패/미지원이면 None 반환.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        # 일부 환경에서 dict가 아닐 수 있어 안전 처리
        if hasattr(info, "get"):
            return info.get("last_price", None)
    except Exception:
        pass
    return None

# ====== 기본 설정 ======
DEFAULT_TICKERS = "003490.KS,005930.KS,AAPL,TSLA,360750.KS,069500.KS,000370.KS,329200.KS,292560.KS,0072R0.KS,138910.KS,017670.KS"

# ---------------------------
# (1) 기본 사이드바 설정
# ---------------------------
with st.sidebar:
    st.header("설정")
    tickers_text = st.text_area("종목들(쉼표로 구분)", value=DEFAULT_TICKERS, height=100)
    period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    min_abs_streak = st.slider("최소 연속일수(절댓값)", 0, 10, 0)
    show_charts = st.checkbox("최근 종가 차트 보기", value=False)
    apply_filter_to_charts = st.checkbox("차트에 요약 필터 적용", value=False)

# 입력 파싱
tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

# ---------------------------
# (2) 데이터 수집 → rows, details
# ---------------------------
rows, details = [], {}
prog = st.progress(0.0)
for i, t in enumerate(tickers):
    df = fetch_daily(t, period)
    if df.empty or "Close" not in df.columns:
        live_price = fetch_live_price(t)  # 일봉이 비어도 현재가는 시도
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

# ---------------------------
# (3) 요약 테이블 summary 만들기
# ---------------------------
summary = pd.DataFrame(rows)

# 필터/정렬
if not summary.empty and "연속일수" in summary.columns:
    if min_abs_streak > 0:
        summary = summary[summary["연속일수"].abs() >= min_abs_streak]
    summary = summary.sort_values(
        by="연속일수",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last"
    )

# 표시용 포맷(소수점 둘째자리)
def _fmt_two(v):
    return None if pd.isna(v) else float(f"{float(v):.2f}")

display_cols = [
    "티커", "이름", "연속일수", "기준일(전날)", "기준가", "현재가(일봉)", "현재가(실시간)", "누적등락(%)", "비고"
]
show_df = summary[[c for c in display_cols if c in summary.columns]].copy()

for col in ["기준가", "현재가(일봉)", "현재가(실시간)"]:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(_fmt_two)

if "누적등락(%)" in show_df.columns:
    show_df["누적등락(%)"] = show_df["누적등락(%)"].apply(
        lambda v: f"{v:.2f}%" if pd.notna(v) else None
    )

st.subheader("전체 현황")
st.dataframe(show_df, use_container_width=True)

# ---------------------------
# (4) 검증 테이블용 사이드바 (※ summary 생성 '이후'에 위치)
# ---------------------------
with st.sidebar:
    st.divider()
    st.subheader("검증 테이블")
    enable_verif = st.checkbox("검증용 테이블 보기", value=False)

    # 안전가드: summary가 비었거나 '티커' 없으면 빈 리스트
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
        st.multiselect(
            "표시할 종목 선택",
            options=list(label_map.keys()),
            default=[],
            placeholder="종목을 선택하세요"
        )
        if (enable_verif and label_map) else []
    )
    selected_tickers = [label_map[l] for l in selected_labels]

# ---------------------------
# (5) 검증 테이블 본문 (선택된 종목만 출력)
# ---------------------------
if enable_verif and selected_tickers:
    st.subheader("검증용: 최근 10영업일 (전일대비 등락률 %)")
    for t in selected_tickers:
        df_ = details.get(t)
        if df_ is None or df_.empty:
            continue
        tmp = df_[["Close"]].copy()
        tmp["전일대비 등락률(%)"] = tmp["Close"].pct_change() * 100

        # 제목: 회사명 (티커)
        name = get_name(t)
        st.write(f"**{name} ({t})**")

        out = tmp.tail(10).copy()
        # 종가/등락률 모두 소수점 둘째자리
        out["Close"] = out["Close"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        out["전일대비 등락률(%)"] = out["전일대비 등락률(%)"].map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else ""
        )
        st.dataframe(out, use_container_width=True)

elif enable_verif:
    st.info("사이드바에서 종목을 선택하세요.")

# ---------------------------
# (6) 선택: 차트
# ---------------------------
if show_charts:
    st.subheader("최근 종가 차트")

    # ✅ 요약 필터 적용 여부 선택
    if apply_filter_to_charts and (not summary.empty) and ("티커" in summary.columns):
        chart_tickers = summary["티커"].tolist()
    else:
        chart_tickers = list(details.keys())

    for t in chart_tickers:
        df_ = details.get(t)
        if df_ is None or df_.empty:
            continue
        s = df_["Close"].dropna()
        if s.empty:
            continue
        st.write(f"**{get_name(t)} ({t})**")
        st.line_chart(s, height=180)
        
st.caption("※ '현재가(실시간)'은 거래소/종목에 따라 지연일 수 있습니다. 제작 : 전인화")


