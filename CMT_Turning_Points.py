import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ---- Sidebar 설정 ----
maturity_options = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30"
}

selected_maturity = st.sidebar.selectbox("Select Treasury Maturity", list(maturity_options.keys()))
fred_id = maturity_options[selected_maturity]

@st.cache_data
def load_data(fred_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", fred_id: "Rate"})
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

df = load_data(fred_id)

# 날짜 설정
min_date = df["Date"].min()
max_date = df["Date"].max()
def_start = pd.to_datetime("2015-01-01")

start_date = st.sidebar.date_input("Start Date", value=def_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.stop()

# 파라미터
frac = st.sidebar.slider("LOESS Smoothing (frac)", 0.001, 0.2, 0.05, step=0.005)
threshold = st.sidebar.slider("Slope Threshold", min_value=0.0001, max_value=0.02, value=0.005, step=0.0005, format="%.4f")
window = st.sidebar.slider("Turning Point Window (days)", 5, 90, 30, step=5)

# 분석 범위 제한
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()

# LOESS 스무딩
smoothed = lowess(df['Rate'], df['Date'], frac=frac)
smoothed_dates = pd.to_datetime(smoothed[:, 0])
smoothed_values = smoothed[:, 1]

# 기울기 계산
slopes = np.diff(smoothed_values)
slope_dates = smoothed_dates[1:]
candidate_idxs = np.where((np.abs(slopes) > 0) & (np.abs(slopes) < threshold))[0]

def find_turning_points(values, dates, candidate_idxs, window):
    peaks, troughs = [], []
    for idx in candidate_idxs:
        start = max(0, idx - window)
        end = min(len(values), idx + window + 1)
        seg = values[start:end]
        val = values[idx]
        if val == np.max(seg):
            peaks.append(idx)
        elif val == np.min(seg):
            troughs.append(idx)
    return peaks, troughs

peak_idxs, trough_idxs = find_turning_points(smoothed_values, smoothed_dates, candidate_idxs, window)

# 전환점 시각화
st.title(f"{selected_maturity} CMT Rate Turning Point Analyzer")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Rate'], label='Raw Rate', alpha=0.4)
ax.plot(smoothed_dates, smoothed_values, color='red', label='LOESS Smoothed')
ax.scatter(slope_dates[peak_idxs], smoothed_values[1:][peak_idxs], color='blue', label='Peaks')
ax.scatter(slope_dates[trough_idxs], smoothed_values[1:][trough_idxs], color='green', label='Troughs')
ax.legend()
ax.grid(True)
st.pyplot(fig)

def analyze_segment(df, dates, window):
    rows = []
    for dt in dates:
        seg = df[(df["Date"] <= dt) & (df["Date"] > dt - pd.Timedelta(days=window))]
        if len(seg) < 2:
            continue
        diff = seg["Rate"].diff().dropna()
        rows.append({
            "Turning Point": dt.date(),
            "Start Date": seg["Date"].min().date(),
            "End Date": seg["Date"].max().date(),
            "Mean Rate": seg["Rate"].mean(),
            "Std Dev": seg["Rate"].std(),
            "Rate Change": seg["Rate"].iloc[-1] - seg["Rate"].iloc[0],
            "Max Daily Change": diff.max(),
            "Min Daily Change": diff.min()
        })
    return pd.DataFrame(rows)

peak_df = analyze_segment(df, slope_dates[peak_idxs], window)
trough_df = analyze_segment(df, slope_dates[trough_idxs], window)

exclude_dates = set()
for idx in peak_idxs + trough_idxs:
    ref = slope_dates[idx]
    rng = pd.date_range(ref - pd.Timedelta(days=window), ref)
    exclude_dates.update(rng.date)

control_df = df[~df["Date"].dt.date.isin(exclude_dates)].copy()
control_df.reset_index(drop=True, inplace=True)

rolling_stats = []
for i in range(len(control_df) - window):
    sub = control_df.iloc[i:i+window]
    changes = sub["Rate"].diff().dropna()
    rolling_stats.append({
        "Window Start": sub["Date"].iloc[0].date(),
        "Window End": sub["Date"].iloc[-1].date(),
        "Mean Rate": sub["Rate"].mean(),
        "Std Dev": sub["Rate"].std(),
        "Rate Change": sub["Rate"].iloc[-1] - sub["Rate"].iloc[0],
        "Max Daily Change": changes.max(),
        "Min Daily Change": changes.min()
    })

control_rolling_df = pd.DataFrame(rolling_stats)

# 테이블 및 다운로드
st.markdown("### 📋 Peak Analysis")
st.dataframe(peak_df)
st.download_button("Download Peak Stats", data=peak_df.to_csv(index=False), file_name="peak_stats.csv")

st.markdown("### 📋 Trough Analysis")
st.dataframe(trough_df)
st.download_button("Download Trough Stats", data=trough_df.to_csv(index=False), file_name="trough_stats.csv")

st.markdown("### 📋 Control Period (Non-Turning Points)")
st.dataframe(control_rolling_df)
st.download_button("Download Control Period Stats", data=control_rolling_df.to_csv(index=False), file_name="control_stats.csv")

# 통계 비교 함수
def compare_groups(label, a_df, b_df):
    if not a_df.empty and not b_df.empty:
        t_stat, p_val = ttest_ind(a_df["Rate Change"].dropna(), b_df["Rate Change"].dropna(), equal_var=False)
        st.write(f"#### {label} vs Control")
        st.write(f"Mean: {a_df['Rate Change'].mean():.4f} vs {b_df['Rate Change'].mean():.4f}")
        st.write(f"T-test p-value: {p_val:.4f}")
        if p_val < 0.05:
            st.success("Significant difference.")
        else:
            st.info("No significant difference.")

compare_groups("Peak", peak_df, control_rolling_df)
compare_groups("Trough", trough_df, control_rolling_df)

# 분포 시각화
st.markdown("### 📊 Distribution of Rate Changes")
fig2, ax2 = plt.subplots()
sns.kdeplot(peak_df["Rate Change"], label="Peak", fill=True, ax=ax2)
sns.kdeplot(trough_df["Rate Change"], label="Trough", fill=True, ax=ax2)
sns.kdeplot(control_rolling_df["Rate Change"], label="Control", fill=True, ax=ax2)
ax2.legend()
st.pyplot(fig2)
