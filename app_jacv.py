# -*- coding: utf-8 -*-
"""
JAC 台股分析版 (app_jac.py)
============================
以 app.py 為基礎，整合「JAC台股分析版.txt」分析心法：
1. OBV 能量潮強化：加入 OBV MA10、量價齊揚/齊跌、頂/底背離偵測（優先級 1）
2. BIAS 乖離率強化：BIAS10/BIAS20 雙週期門檻（±7% / ±10%）、
   趨勢股門檻放寬 1.5 倍、與 KD/RSI/OBV 的共振驗證
3. 支撐壓力規則：近 20 日高低點、突破/跌破/接近判斷
4. 量能規則：2 倍量放大、50% 量萎縮
5. JAC 智能警示報告：優先級 1/2/3 警示清單、綜合評分（-100 ~ +100）、
   關鍵數據、操作建議、風險提示
"""

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import pandas as pd
import numpy as np
import matplotlib
import os
from matplotlib import font_manager
import matplotlib.patches as mpatches

# ==========================================
# 0. 網頁基本配置與字型處理
# ==========================================
st.set_page_config(page_title="JAC 台股智能分析版", layout="wide")

# 處理中文字型 (解決雲端 Linux 亂碼問題)
font_path = "NotoSansTC-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()]
else:
    # 本機環境嘗試使用正黑體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

plt.rcParams['axes.unicode_minus'] = False

st.title("📊 JAC 台股智能分析版")
st.markdown("""
本專案整合 **JAC 分析心法**：從資料獲取、6 大指標計算（均線/布林帶、OBV、KDJ、MACD、RSI、BIAS），
到 **JAC 智能警示報告**（優先級警示、OBV 量價背離、BIAS 乖離門檻、綜合評分與操作建議）的完整流程。
""")

# ==========================================
# 1. 側邊欄與參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
stock_id = st.sidebar.text_input("股票代號", "2330.TW")

default_end = datetime.today().date()
default_start = (datetime.today()-timedelta(180)).date()
target_start = st.sidebar.date_input("觀測起始日", default_start)
target_end = st.sidebar.date_input("觀測結束日", default_end)
warmup_days = st.sidebar.slider("指標預熱天數 (用於 EMA/RSI 準確度)", 30, 100, 60)

st.sidebar.divider()
st.sidebar.subheader("🎯 JAC 心法參數")
bias10_limit = st.sidebar.slider("BIAS10 乖離警戒 (±%)", 3.0, 15.0, 7.0, 0.5)
bias20_limit = st.sidebar.slider("BIAS20 乖離警戒 (±%)", 5.0, 20.0, 10.0, 0.5)
trend_relax = st.sidebar.checkbox("趨勢股乖離門檻放寬 1.5 倍（JAC 心法）", value=True)

# 顯示字型狀態診斷
if os.path.exists(font_path):
    st.sidebar.success("✅ 已載入 NotoSansTC 中文字型")
else:
    st.sidebar.warning("⚠️ 使用系統預設中文字型 (雲端環境請確認已上傳 ttf 檔)")

# ==========================================
# 2. 步驟 1：資料獲取與「預熱」邏輯
# ==========================================
st.header("Step 1: 資料獲取與預熱處理")
with st.expander("📖 為什麼需要預熱資料？"):
    st.write("""
    - **預熱機制 (Warm-up)**：EMA、MACD 與 RSI 都是具備「延續性」的指標。如果直接從觀測日開始計算，初始值會產生嚴重的偏差。
    - 本程式自動向前抓取（預設 60 天）的資料進行「預熱」計算，確保在進入使用者選定的觀測區間時，所有指標已趨於穩定準確。
    - **避免格式錯誤**：強制將日期轉為 `YYYY-MM-DD` 格式再向 Yahoo Finance 請求，確保網頁一開啟就能正確載入資料。
    """)

@st.cache_data
def load_stock_data(symbol, start_dt, end_dt, warmup):
    # 向前推 warmup 天數
    fetch_start = start_dt - timedelta(days=warmup)

    # 強制轉換為字串格式，避免 datetime 物件造成 yfinance 解析異常
    start_str = fetch_start.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')

    df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False)

    if not df.empty:
        # 展平欄位名稱 (若 yfinance 回傳 MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 清洗壞資料：Yahoo 偶爾回傳 OHLC 為 NaN 或 0 的異常列（如 0050.TW），
        # 會造成 K 線插到 0、支撐位 = 0 與後續 NoneType 運算錯誤
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        df = df.dropna(subset=ohlc_cols)
        df = df[(df[ohlc_cols] > 0).all(axis=1)]
    return df

df_all = load_stock_data(stock_id, target_start, target_end, warmup_days)

if df_all.empty:
    st.error("找不到資料，請檢查代號或網路連線。")
    st.stop()

# ==========================================
# 3. 步驟 2：技術指標運算 (6大指標 + JAC 強化)
# ==========================================
st.header("Step 2: 技術指標運算 (Indicator Math)")
with st.expander("📖 查看 6 大指標運算邏輯說明"):
    st.markdown("""
    1. **均線與布林通道 (SMA & BBands)**：5日、10日、20日均線，以及 20日均線上下 2 倍標準差的軌道。
    2. **KDJ**：透過 9 日最高/最低價計算 RSV，再以 EWM (指數加權移動平均) 平滑出 K、D、J 值。
    3. **OBV (能量潮)**：利用成交量與股價漲跌的累計值，觀察資金進出動向。
       **JAC 強化**：加入 OBV MA10 趨勢基準線，用於偵測量能轉強/轉弱與量價背離。
    4. **MACD**：計算 12日與 26日 EMA 之差 (DIF)，以及其 9日訊號線 (MACD)。
    5. **RSI (相對強弱指標)**：使用 Yahoo Finance 官方標準公式 (修正平滑移動平均法) 計算 5日與 10日 RSI。
    6. **BIAS (乖離率)**：計算股價偏離 10日與 20日均線的百分比。
       **JAC 強化**：以 ±7% (BIAS10) / ±10% (BIAS20) 為超漲超跌警戒線，趨勢股門檻放寬 1.5 倍。
    """)

with st.spinner('各項指標計算中...'):
    df_calculated = df_all.copy()

    # 1. SMA & BBands
    df_calculated['SMA_5'] = df_calculated['Close'].rolling(window=5).mean()
    df_calculated['SMA_10'] = df_calculated['Close'].rolling(window=10).mean()
    df_calculated['SMA_20'] = df_calculated['Close'].rolling(window=20).mean()
    df_calculated['std_dev'] = df_calculated['Close'].rolling(window=20).std()
    df_calculated['upper_band'] = df_calculated['SMA_20'] + (df_calculated['std_dev'] * 2)
    df_calculated['lower_band'] = df_calculated['SMA_20'] - (df_calculated['std_dev'] * 2)

    # 2. KDJ (EWM 快速法)
    n = 9
    low_min = df_calculated['Low'].rolling(window=n).min()
    high_max = df_calculated['High'].rolling(window=n).max()
    df_calculated['RSV'] = ((df_calculated['Close'] - low_min) / (high_max - low_min)) * 100
    df_calculated['K'] = df_calculated['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['D'] = df_calculated['K'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['J'] = 3 * df_calculated['D'] - 2 * df_calculated['K']

    # 3. OBV + OBV MA10 (JAC 心法：以 10 日均線作為量能趨勢基準)
    df_calculated['OBV'] = np.where(df_calculated['Close'] > df_calculated['Close'].shift(1), df_calculated['Volume'], -df_calculated['Volume']).cumsum()
    df_calculated['OBV_MA10'] = df_calculated['OBV'].rolling(window=10).mean()

    # 4. MACD
    df_calculated['EMA12'] = df_calculated['Close'].ewm(span=12, adjust=False).mean()
    df_calculated['EMA26'] = df_calculated['Close'].ewm(span=26, adjust=False).mean()
    df_calculated['DIF'] = df_calculated['EMA12'] - df_calculated['EMA26']
    df_calculated['MACD'] = df_calculated['DIF'].ewm(span=9, adjust=False).mean()
    df_calculated['MACD Histogram'] = df_calculated['DIF'] - df_calculated['MACD']

    # 5. RSI (Yahoo 靈魂公式)
    def yahoo_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df_calculated['RSI5'] = yahoo_rsi(df_calculated['Close'], 5)
    df_calculated['RSI10'] = yahoo_rsi(df_calculated['Close'], 10)

    # 6. BIAS 乖離率
    df_calculated['BIAS10'] = ((df_calculated['Close'] - df_calculated['SMA_10']) / df_calculated['SMA_10']) * 100
    df_calculated['BIAS20'] = ((df_calculated['Close'] - df_calculated['SMA_20']) / df_calculated['SMA_20']) * 100
    df_calculated['B10-B20'] = df_calculated['BIAS10'] - df_calculated['BIAS20']

    # 7. JAC 支撐壓力（近 20 日高低點）與量能基準
    df_calculated['P20_High'] = df_calculated['High'].rolling(window=20).max()
    df_calculated['P20_Low'] = df_calculated['Low'].rolling(window=20).min()
    df_calculated['Vol_MA20'] = df_calculated['Volume'].rolling(window=20).mean()

# --- 過濾預熱資料 ---
# 確保 index 格式為 datetime，以利於進行時間過濾
df_calculated.index = pd.to_datetime(df_calculated.index)
mask_start = pd.Timestamp(target_start)
df = df_calculated.loc[mask_start:].copy()

# 將最終繪圖用的 DataFrame 索引轉為字串格式
df.index = df.index.map(lambda x: x.strftime('%y-%m-%d'))

with st.expander("🔍 查看已計算的指標數據 (觀測區間末七筆)"):
    st.dataframe(df.tail(7))

# ==========================================
# 4. 步驟 3：專業多圖層視覺化
# ==========================================
st.header("Step 3: 綜合技術指標儀表板")
with st.expander("📖 查看圖表排版設計說明"):
    st.markdown("""
    本圖表使用 Matplotlib 的 `add_subplot(8, 1, ...)` 將畫布切分為 8 個單位高度：
    - **區塊 1-3 (主圖)**：顯示 K 線、5/10/20 日均線與布林通道，並標示近 20 日壓力/支撐位。
    - **區塊 4 (OBV & Volume)**：能量潮曲線 + **OBV MA10 基準線**（JAC 心法）與成交量柱狀圖 (雙 Y 軸)。
    - **區塊 5 (KDJ)**：K、D、J 三線交叉觀察，標示 80/20 超買超賣線。
    - **區塊 6 (MACD)**：DIF、MACD 指標及其紅綠柱狀圖。
    - **區塊 7 (RSI)**：觀察 RSI5 與 RSI10 是否觸及超買 (70) 或超賣 (30) 虛線區間。
    - **區塊 8 (BIAS)**：10日與 20日乖離率，**標示 ±7% / ±10% JAC 警戒線**。
    """)

# 建立圖表畫布 (高度拉高到 16 以容納 6 個圖表)
fig = plt.figure(figsize=(14, 16), layout='constrained')

# 定義 6 大區塊 (總共 8 個單位)
ax1 = fig.add_subplot(8,1,(1,3)) # 主圖 (佔 3 單位)
ax2 = fig.add_subplot(8,1,4)     # OBV (佔 1 單位)
ax3 = fig.add_subplot(8,1,5)     # KDJ (佔 1 單位)
ax4 = fig.add_subplot(8,1,6)     # MACD (佔 1 單位)
ax5 = fig.add_subplot(8,1,7)     # RSI (佔 1 單位)
ax6 = fig.add_subplot(8,1,8)     # BIAS (佔 1 單位)

# 定義 X 軸刻度間隔 (每 15 根 K 棒顯示一次)
x_ticks_pos = range(0, len(df.index), 15)
x_ticks_labels = df.index[::15]

# --- Ax1: K線 + 均線 + 布林帶 + 壓力/支撐 ---
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels(x_ticks_labels)
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'],
                       width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(df['SMA_5'],label='5日均線', color='cyan', lw=1)
ax1.plot(df['SMA_10'],label='10日均線', color='purple', lw=1)
ax1.plot(df['SMA_20'],label='20日均線', color='orange', lw=1)
ax1.plot(df['upper_band'], label='布林上軌', color='g', ls=':', lw=1)
ax1.plot(df['lower_band'], label='布林下軌', color='g', ls=':', lw=1)
# JAC 心法：近 20 日壓力/支撐水平線
pressure = float(df['P20_High'].iloc[-1])
support = float(df['P20_Low'].iloc[-1])
ax1.axhline(pressure, color='red', ls='--', lw=1.2, alpha=0.7, label=f'壓力位 {pressure:.1f}')
ax1.axhline(support, color='green', ls='--', lw=1.2, alpha=0.7, label=f'支撐位 {support:.1f}')
ax1.legend(loc='upper left', fontsize='small')
ax1.set_title(f"【{stock_id}】JAC 綜合技術分析", fontsize=16)

# --- Ax2: OBV + OBV MA10 與 成交量 ---
ax2.set_xticks(x_ticks_pos)
ax2.set_xticklabels([])
conditions = [
    df['Close'] > df['Close'].shift(1),  # 漲 -> 紅
    df['Close'] < df['Close'].shift(1)   # 跌 -> 綠
]
choices = ['r', 'g']
vol_colors = np.select(conditions, choices, default='gray')

ax2.plot(df['OBV'], color='purple', ls='--', label='OBV')
ax2.plot(df['OBV_MA10'], color='orange', lw=1, label='OBV MA10')
ax2_v = ax2.twinx()
ax2_v.bar(df.index, df['Volume'], color=vol_colors, alpha=0.3, width=0.8)
ax2.set_title("OBV 能量潮（含 MA10 趨勢基準）")
ax2.legend(loc='upper right', fontsize='small')
red_patch = mpatches.Patch(color='red', label='紅色漲')
green_patch = mpatches.Patch(color='green', label='綠色跌')
gray_patch = mpatches.Patch(color='gray', label='灰持平')
ax2_v.legend(handles=[red_patch, green_patch,gray_patch],loc=2)
# --- Ax3: KDJ ---
ax3.plot(df['K'], label='K線', color='cyan', lw=1)
ax3.plot(df['D'], label='D線', color='purple', lw=1)
ax3.plot(df['J'], label='J線', color='orange', ls='--')
ax3.axhline(80, color='r', ls='--', lw=0.8, alpha=0.5)  # 超買線
ax3.axhline(20, color='g', ls='--', lw=0.8, alpha=0.5)  # 超賣線
ax3.set_xticks(x_ticks_pos)
ax3.set_xticklabels(x_ticks_labels)
ax3.set_title("KDJ 指標")
ax3.legend(loc='upper left', fontsize='small')

# --- Ax4: MACD ---
ax4.plot(df['DIF'], label='DIF', color='purple')
ax4.plot(df['MACD'], label='MACD', color='skyblue')
m_hist_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.bar(df.index, df['MACD Histogram'], color=m_hist_colors, alpha=0.6)
ax4.axhline(0, color='gray', ls='--', lw=1)
ax4.set_xticks(x_ticks_pos)
ax4.set_xticklabels([])
ax4.set_title("MACD 指標")
ax4.legend(loc='upper left', fontsize='small')

# --- Ax5: RSI ---
ax5.plot(df['RSI5'], label='RSI5', color='cyan', lw=1)
ax5.plot(df['RSI10'], label='RSI10', color='purple', lw=1)
ax5.axhline(70, color='r', ls='--', lw=0.8, alpha=0.5) # 超買線
ax5.axhline(30, color='g', ls='--', lw=0.8, alpha=0.5) # 超賣線
ax5.set_ylim(0, 100)
ax5.set_xticks(x_ticks_pos)
ax5.set_xticklabels(x_ticks_labels)
ax5.set_title("RSI 相對強弱指標")
ax5.legend(loc='upper left', fontsize='small')

# --- Ax6: BIAS (含 JAC ±7% / ±10% 警戒線) ---
ax6.plot(df['BIAS10'], label='BIAS10', color='cyan', lw=1)
ax6.plot(df['BIAS20'], label='BIAS20', color='purple', lw=1)
bias_diff_colors = np.where(df['B10-B20'] >= 0, 'r', 'g')
ax6.bar(df.index, df['B10-B20'], color=bias_diff_colors, alpha=0.6)
ax6.axhline(0, color='gray', ls='--', lw=1)
# JAC 心法：乖離警戒線
ax6.axhline(bias10_limit, color='red', ls=':', lw=1, alpha=0.7)
ax6.axhline(-bias10_limit, color='green', ls=':', lw=1, alpha=0.7)
ax6.axhline(bias20_limit, color='red', ls='--', lw=1, alpha=0.5)
ax6.axhline(-bias20_limit, color='green', ls='--', lw=1, alpha=0.5)
ax6.set_xticks(x_ticks_pos)
ax6.set_xticklabels([])
ax6.set_title(f"BIAS 乖離率（警戒：BIAS10 ±{bias10_limit}%、BIAS20 ±{bias20_limit}%）")
ax6.legend(loc='upper left', fontsize='small')

# 渲染到網頁
st.pyplot(fig)

st.divider()

# ==========================================
# 5. 步驟 4：進場 / 獲利了結 綜合分析
# ==========================================
st.header("Step 4: 進場 / 獲利了結 綜合分析")
st.caption("以下根據觀測區間「最後一個交易日」的指標數值自動研判，僅供技術面教學參考，非投資建議。")


def _f(value):
    """安全取出純量數值，遇到 NaN 回傳 None。"""
    try:
        v = float(value)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# 取最後一筆與前一筆，用於判斷交叉與趨勢方向
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) >= 2 else last

close = _f(last['Close'])
sma5, sma10, sma20 = _f(last['SMA_5']), _f(last['SMA_10']), _f(last['SMA_20'])
upper, lower = _f(last['upper_band']), _f(last['lower_band'])
k, d = _f(last['K']), _f(last['D'])
k_prev, d_prev = _f(prev['K']), _f(prev['D'])
dif, macd_sig = _f(last['DIF']), _f(last['MACD'])
dif_prev, macd_prev = _f(prev['DIF']), _f(prev['MACD'])
rsi5 = _f(last['RSI5'])
bias20 = _f(last['BIAS20'])
obv_last, obv_prev5 = _f(last['OBV']), _f(df['OBV'].iloc[-6]) if len(df) >= 6 else _f(prev['OBV'])

signals = []  # 每項 (指標, 訊號文字, 分數)；分數 +偏多 / -偏空


# 1. 均線排列與布林位置
if None not in (sma5, sma10, sma20):
    if sma5 > sma10 > sma20:
        signals.append(("均線排列", "🔴 多頭排列（5>10>20），趨勢偏多", 1))
    elif sma5 < sma10 < sma20:
        signals.append(("均線排列", "🟢 空頭排列（5<10<20），趨勢偏空", -1))
    else:
        signals.append(("均線排列", "⚪ 均線糾結，方向未明", 0))
if None not in (close, upper, lower):
    if close >= upper:
        signals.append(("布林通道", "🟢 股價觸及或突破上軌，短線過熱、留意獲利了結", -1))
    elif close <= lower:
        signals.append(("布林通道", "🔴 股價觸及或跌破下軌，超跌可留意反彈進場", 1))
    else:
        signals.append(("布林通道", "⚪ 股價於布林通道中段，無極端訊號", 0))

# 2. KDJ 交叉與超買超賣
if None not in (k, d, k_prev, d_prev):
    if k_prev <= d_prev and k > d:
        signals.append(("KDJ", f"🔴 K 線由下往上穿越 D 線（黃金交叉，K={k:.1f}），偏多", 1))
    elif k_prev >= d_prev and k < d:
        signals.append(("KDJ", f"🟢 K 線由上往下跌破 D 線（死亡交叉，K={k:.1f}），偏空", -1))
    elif k < 20 and d < 20:
        signals.append(("KDJ", f"🔴 K、D 同處超賣區（<20），可留意反彈進場", 1))
    elif k > 80 and d > 80:
        signals.append(("KDJ", f"🟢 K、D 同處超買區（>80），短線過熱可獲利了結", -1))
    else:
        signals.append(("KDJ", f"⚪ KDJ 無明顯交叉訊號（K={k:.1f}, D={d:.1f}）", 0))

# 3. OBV 資金流向
if None not in (obv_last, obv_prev5):
    if obv_last > obv_prev5:
        signals.append(("OBV 能量潮", "🔴 OBV 近 5 日走高，量能支撐、資金流入", 1))
    elif obv_last < obv_prev5:
        signals.append(("OBV 能量潮", "🟢 OBV 近 5 日走低，量能轉弱、資金流出", -1))
    else:
        signals.append(("OBV 能量潮", "⚪ OBV 持平，量能無明顯方向", 0))

# 4. MACD 交叉與多空軸
if None not in (dif, macd_sig, dif_prev, macd_prev):
    if dif_prev <= macd_prev and dif > macd_sig:
        signals.append(("MACD", "🔴 DIF 向上穿越訊號線（黃金交叉），動能轉強", 1))
    elif dif_prev >= macd_prev and dif < macd_sig:
        signals.append(("MACD", "🟢 DIF 向下跌破訊號線（死亡交叉），動能轉弱", -1))
    elif dif > 0 and dif > macd_sig:
        signals.append(("MACD", "🔴 DIF 位於零軸之上且高於訊號線，多方動能延續", 1))
    elif dif < 0 and dif < macd_sig:
        signals.append(("MACD", "🟢 DIF 位於零軸之下且低於訊號線，空方動能延續", -1))
    else:
        signals.append(("MACD", "⚪ MACD 動能中性", 0))

# 5. RSI 超買超賣
if rsi5 is not None:
    if rsi5 >= 70:
        signals.append(("RSI", f"🟢 RSI5={rsi5:.1f} 進入超買區（≥70），漲多過熱、可獲利了結", -1))
    elif rsi5 <= 30:
        signals.append(("RSI", f"🔴 RSI5={rsi5:.1f} 進入超賣區（≤30），跌深可留意進場", 1))
    else:
        signals.append(("RSI", f"⚪ RSI5={rsi5:.1f} 位於中性區間（30~70）", 0))

# 6. BIAS 乖離率
if bias20 is not None:
    if bias20 >= 8:
        signals.append(("BIAS 乖離率", f"🟢 20日乖離率 {bias20:+.1f}% 過大，股價偏離均線過遠、回檔風險升高", -1))
    elif bias20 <= -8:
        signals.append(("BIAS 乖離率", f"🔴 20日乖離率 {bias20:+.1f}% 過低，超跌反彈機會浮現", 1))
    else:
        signals.append(("BIAS 乖離率", f"⚪ 20日乖離率 {bias20:+.1f}% 在合理範圍，無極端訊號", 0))

# --- 綜合評分與結論 ---
step4_score = sum(s[2] for s in signals)
bull = sum(1 for s in signals if s[2] > 0)
bear = sum(1 for s in signals if s[2] < 0)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("綜合多空評分", f"{step4_score:+d}", help="各指標偏多 +1、偏空 -1 的加總")
    st.write(f"🔴 偏多訊號：**{bull}** 項　🟢 偏空訊號：**{bear}** 項")
with col_b:
    if step4_score >= 3:
        # 台股紅漲綠跌：偏多用紅框 (st.error 為紅色)
        st.error("**綜合研判：偏多 → 可留意進場 / 續抱**\n\n多數指標站在多方，趨勢與動能一致向上。若尚未持有可分批佈局，已持有者續抱，並設好停損（如跌破 20 日均線或布林下軌）。")
    elif step4_score <= -3:
        # 台股紅漲綠跌：偏空用綠框 (st.success 為綠色)
        st.success("**綜合研判：偏空 → 留意獲利了結 / 觀望**\n\n多數指標轉弱或進入過熱區，上漲動能衰竭。持有者可考慮分批獲利了結或減碼，空手者宜觀望、勿追高。")
    else:
        st.warning("**綜合研判：中性 → 區間整理 / 等待訊號**\n\n多空訊號互有拉鋸，方向尚未明朗。建議等待均線、MACD、KDJ 出現一致訊號後再決定進出場。")

st.subheader("📋 各指標逐項研判")
signal_df = pd.DataFrame(
    [(name, text, "偏多" if sc > 0 else ("偏空" if sc < 0 else "中性")) for name, text, sc in signals],
    columns=["指標", "目前研判", "方向"],
)
st.table(signal_df)

with st.expander("📖 如何用這 6 大指標判斷進場與獲利了結？"):
    st.markdown("""
    技術指標沒有單一萬靈丹，重點是**多項指標互相驗證（共振）**。以下是常見的判讀原則：

    **🔴 偏向「進場 / 加碼」的訊號**
    - **均線多頭排列**：5 日 > 10 日 > 20 日，且股價站上均線，代表趨勢向上。
    - **KDJ 黃金交叉**：K 線於低檔（<20）由下往上穿越 D 線。
    - **MACD 黃金交叉**：DIF 向上突破訊號線，柱狀圖由綠翻紅，最好發生在零軸附近或之上。
    - **RSI 由超賣區（<30）回升**：跌深出現買盤。
    - **OBV 持續走高**：股價上漲伴隨量能支撐，較不易是假突破。
    - **布林下軌 + BIAS 負乖離過大**：超跌後的反彈機會。

    **🟢 偏向「獲利了結 / 減碼」的訊號**
    - **均線空頭排列**或股價跌破 20 日均線：趨勢轉弱。
    - **KDJ 在高檔（>80）死亡交叉**：短線過熱、動能反轉。
    - **MACD 死亡交叉**：DIF 跌破訊號線，柱狀圖由紅翻綠。
    - **RSI 進入超買區（>70）**：漲多過熱，續抱風險升高。
    - **股價觸及布林上軌 + BIAS 正乖離過大（如 >8%）**：偏離均線太遠，易拉回。
    - **OBV 背離**：股價創新高但 OBV 未同步走高，量能不支持，留意假突破。

    **⚠️ 實務提醒**
    - 指標多為「落後 / 同步」訊號，無法預測未來；務必搭配**停損紀律**（例如跌破 20 日均線或布林下軌出場）。
    - 不同指標訊號矛盾時，以**趨勢方向（均線、MACD）為主，擺盪指標（KDJ、RSI、BIAS）為輔**。
    - 本頁面的「綜合多空評分」是把各項訊號簡單加總的教學示範，實際操作仍需考量基本面、籌碼面與大盤環境。
    """)

st.divider()

# ==========================================
# 6. 步驟 5：ECF 智能警示報告
# ==========================================
st.header("Step 5: 📊 ECF 智能警示報告")
st.caption("依 ECF 分析心法，以觀測區間「最後一個交易日」數值自動研判。僅供技術面教學參考，非投資建議。")

if len(df) < 21:
    st.warning("觀測區間資料不足 21 筆，無法完整執行 ECF 警示規則（背離偵測需近 20 日資料），請拉長觀測區間。")
    st.stop()

# --- 取數 ---
last = df.iloc[-1]
prev = df.iloc[-2]

close = _f(last['Close'])
close_prev = _f(prev['Close'])
sma5, sma10, sma20 = _f(last['SMA_5']), _f(last['SMA_10']), _f(last['SMA_20'])
sma5_p, sma20_p = _f(prev['SMA_5']), _f(prev['SMA_20'])
upper, lower = _f(last['upper_band']), _f(last['lower_band'])
k, d = _f(last['K']), _f(last['D'])
k_prev, d_prev = _f(prev['K']), _f(prev['D'])
dif, macd_sig = _f(last['DIF']), _f(last['MACD'])
dif_prev, macd_prev = _f(prev['DIF']), _f(prev['MACD'])
osc, osc_prev = _f(last['MACD Histogram']), _f(prev['MACD Histogram'])
rsi5, rsi10 = _f(last['RSI5']), _f(last['RSI10'])
bias10, bias20 = _f(last['BIAS10']), _f(last['BIAS20'])
obv, obv_prev = _f(last['OBV']), _f(prev['OBV'])
obv_ma10, obv_ma10_prev = _f(last['OBV_MA10']), _f(prev['OBV_MA10'])
volume, vol_ma20 = _f(last['Volume']), _f(last['Vol_MA20'])
pressure = _f(last['P20_High'])
support = _f(last['P20_Low'])

# 近 20 日視窗（不含今日與含今日各取一份，供背離/突破判斷）
win20 = df.iloc[-20:]
win20_ex_today = df.iloc[-21:-1]

# --- 警示引擎 ---
# 每筆警示：(priority 1/2/3, tag 'bullish'/'bearish'/'warning'/'neutral', 標題, 說明, 分數)
alerts = []
TAG_ICON = {'bullish': '🔴', 'bearish': '🟢', 'warning': '🟡', 'neutral': '⚪'}
TAG_TEXT = {'bullish': '看漲', 'bearish': '看跌', 'warning': '警告', 'neutral': '中性'}
P_SCORE = {1: 30, 2: 15, 3: 5}  # 優先級對應分數權重


def add_alert(priority, tag, title, detail):
    sign = 1 if tag == 'bullish' else (-1 if tag == 'bearish' else 0)
    alerts.append((priority, tag, title, detail, sign * P_SCORE[priority]))


# ========== 一、均線分析規則 ==========
trend_state = "糾結"
if None not in (sma5, sma10, sma20):
    if sma5 > sma10 > sma20:
        trend_state = "多頭排列"
        add_alert(2, 'bullish', "均線多頭排列", f"MA5({sma5:.1f}) > MA10({sma10:.1f}) > MA20({sma20:.1f})，趨勢偏多")
    elif sma5 < sma10 < sma20:
        trend_state = "空頭排列"
        add_alert(2, 'bearish', "均線空頭排列", f"MA5({sma5:.1f}) < MA10({sma10:.1f}) < MA20({sma20:.1f})，趨勢偏空")

if None not in (sma5, sma20, sma5_p, sma20_p):
    if sma5_p <= sma20_p and sma5 > sma20:
        add_alert(1, 'bullish', "均線黃金交叉", f"MA5 從下方穿越 MA20（{sma5:.1f} > {sma20:.1f}），買進訊號")
    elif sma5_p >= sma20_p and sma5 < sma20:
        add_alert(1, 'bearish', "均線死亡交叉", f"MA5 從上方跌破 MA20（{sma5:.1f} < {sma20:.1f}），賣出訊號")

# ========== 二、支撐壓力規則（近20日） ==========
if None not in (close, close_prev, pressure, support):
    prev_pressure = _f(prev['P20_High'])
    prev_support = _f(prev['P20_Low'])
    if close > pressure * 0.995 and not (prev_pressure and close_prev > prev_pressure * 0.995):
        add_alert(1, 'bullish', "突破 20 日壓力位", f"收盤 {close:.1f} 突破壓力位 {pressure:.1f}（× 0.995 容忍帶），買進訊號")
    elif close < support * 1.005 and not (prev_support and close_prev < prev_support * 1.005):
        add_alert(1, 'bearish', "跌破 20 日支撐位", f"收盤 {close:.1f} 跌破支撐位 {support:.1f}（× 1.005 容忍帶），賣出訊號")
    elif pressure * 0.97 <= close <= pressure:
        add_alert(3, 'neutral', "接近壓力位", f"收盤 {close:.1f} 位於壓力位 {pressure:.1f} 的 97%~100% 區間，觀察是否突破")
    elif support <= close <= support * 1.03:
        add_alert(3, 'neutral', "接近支撐位", f"收盤 {close:.1f} 位於支撐位 {support:.1f} 的 100%~103% 區間，觀察是否守穩")

# ========== 三、成交量規則 ==========
vol_ratio = None
if None not in (volume, vol_ma20) and vol_ma20 > 0:
    vol_ratio = volume / vol_ma20
    if vol_ratio >= 2:
        if close is not None and close_prev is not None and close > close_prev:
            add_alert(2, 'bullish', "量增價漲", f"成交量為 20 日均量 {vol_ratio:.1f} 倍且價漲，多方強勢")
        else:
            add_alert(2, 'warning', "量增價跌", f"成交量為 20 日均量 {vol_ratio:.1f} 倍但價跌，留意賣壓")
    elif vol_ratio < 0.5:
        add_alert(3, 'neutral', "量能萎縮", f"成交量僅 20 日均量 {vol_ratio:.1f} 倍（<50%），市場觀望")

# ========== 四、KD 指標規則（超買80、超賣20） ==========
if None not in (k, d, k_prev, d_prev):
    if k_prev <= d_prev and k > d:
        add_alert(2, 'bullish', "KD 黃金交叉", f"K({k:.1f}) 從下方穿越 D({d:.1f})，短線買進訊號")
    elif k_prev >= d_prev and k < d:
        add_alert(2, 'bearish', "KD 死亡交叉", f"K({k:.1f}) 從上方跌破 D({d:.1f})，短線賣出訊號")
    if k > 80 and d > 80:
        add_alert(2, 'warning', "KD 超買", f"K({k:.1f})、D({d:.1f}) 均 > 80，可能面臨回檔")
    elif k < 20 and d < 20:
        add_alert(2, 'warning', "KD 超賣", f"K({k:.1f})、D({d:.1f}) 均 < 20，可能出現反彈")

# ========== 五、RSI 指標規則（超買80、超賣20 + 背離） ==========
rsi_overbought = rsi_oversold = False
if rsi10 is not None:
    if rsi10 > 80:
        rsi_overbought = True
        add_alert(2, 'warning', "RSI 超買", f"RSI10 = {rsi10:.1f} > 80，股價可能過熱")
    elif rsi10 < 20:
        rsi_oversold = True
        add_alert(2, 'warning', "RSI 超賣", f"RSI10 = {rsi10:.1f} < 20，股價可能超跌")

    # RSI 背離（以近 20 日為觀察窗）
    price_new_high = close is not None and close >= _f(win20['Close'].max())
    price_new_low = close is not None and close <= _f(win20['Close'].min())
    rsi_20d_max = _f(win20_ex_today['RSI10'].max())
    rsi_20d_min = _f(win20_ex_today['RSI10'].min())
    if price_new_high and rsi_20d_max is not None and rsi10 < rsi_20d_max and rsi10 > 60:
        add_alert(2, 'bearish', "RSI 頂背離", f"價格創近 20 日新高但 RSI10({rsi10:.1f}) 未創高，趨勢轉弱信號")
    if price_new_low and rsi_20d_min is not None and rsi10 > rsi_20d_min and rsi10 < 40:
        add_alert(2, 'bullish', "RSI 底背離", f"價格創近 20 日新低但 RSI10({rsi10:.1f}) 未創低，趨勢轉強信號")

# ========== 六、MACD 指標規則 ==========
if None not in (dif, macd_sig, dif_prev, macd_prev):
    if dif_prev <= macd_prev and dif > macd_sig:
        add_alert(1, 'bullish', "MACD 黃金交叉", f"DIF({dif:.2f}) 從下方穿越 MACD({macd_sig:.2f})，中期趨勢轉多")
    elif dif_prev >= macd_prev and dif < macd_sig:
        add_alert(1, 'bearish', "MACD 死亡交叉", f"DIF({dif:.2f}) 從上方跌破 MACD({macd_sig:.2f})，中期趨勢轉空")
if None not in (osc, osc_prev):
    if osc_prev <= 0 < osc:
        add_alert(3, 'bullish', "MACD 柱狀體翻正", f"OSC 由負轉正（{osc:+.2f}），動能轉強")
    elif osc_prev >= 0 > osc:
        add_alert(3, 'bearish', "MACD 柱狀體翻負", f"OSC 由正轉負（{osc:+.2f}），動能轉弱")

# ========== 七、布林通道規則 ==========
if None not in (close, upper, lower):
    if close >= upper:
        add_alert(3, 'warning', "觸及布林上軌", f"收盤 {close:.1f} ≥ 上軌 {upper:.1f}，短線可能過熱")
    elif close <= lower:
        add_alert(3, 'warning', "觸及布林下軌", f"收盤 {close:.1f} ≤ 下軌 {lower:.1f}，短線可能超跌")

# ========== 八、OBV 能量潮規則（ECF 心法） ==========
obv_state = "中性"
obv_top_div = obv_bottom_div = False
if obv is not None:
    obv_20d_max = _f(win20_ex_today['OBV'].max())
    obv_20d_min = _f(win20_ex_today['OBV'].min())
    price_new_high = close is not None and close >= _f(win20['Close'].max())
    price_new_low = close is not None and close <= _f(win20['Close'].min())
    obv_new_high = obv_20d_max is not None and obv >= obv_20d_max
    obv_new_low = obv_20d_min is not None and obv <= obv_20d_min

    # 1/2. 量價齊揚 / 量價齊跌
    if price_new_high and obv_new_high:
        obv_state = "量價齊揚"
        add_alert(2, 'bullish', "OBV 量價齊揚", "股價與 OBV 同步創近 20 日新高，多方動能確認")
    elif price_new_low and obv_new_low:
        obv_state = "量價齊跌"
        add_alert(2, 'bearish', "OBV 量價齊跌", "股價與 OBV 同步創近 20 日新低，空方動能確認")
    # 3/4. 頂背離 / 底背離（ECF 心法：優先級 1）
    elif price_new_high and not obv_new_high:
        obv_state = "頂背離"
        obv_top_div = True
        add_alert(1, 'bearish', "OBV 頂背離", "股價創近 20 日新高但 OBV 未創新高，上漲缺乏量能支撐，趨勢轉弱信號")
    elif price_new_low and not obv_new_low:
        obv_state = "底背離"
        obv_bottom_div = True
        add_alert(1, 'bullish', "OBV 底背離", "股價創近 20 日新低但 OBV 未創新低，賣壓減輕，趨勢轉強信號")

    # 5/6. OBV 突破 / 跌破 MA10
    if None not in (obv_ma10, obv_prev, obv_ma10_prev):
        if obv_prev <= obv_ma10_prev and obv > obv_ma10:
            add_alert(3, 'bullish', "OBV 突破均線", "OBV 從下方穿越 OBV MA10，量能轉強，偏多觀察")
        elif obv_prev >= obv_ma10_prev and obv < obv_ma10:
            add_alert(3, 'bearish', "OBV 跌破均線", "OBV 從上方跌破 OBV MA10，量能轉弱，偏空觀察")

    # 7/8. 價平量增 / 價平量減（近 5 日股價盤整 ±2% 內，OBV 明顯走向）
    if len(df) >= 6:
        close_5ago = _f(df['Close'].iloc[-6])
        obv_5ago = _f(df['OBV'].iloc[-6])
        obv_range = (_f(win20['OBV'].max()) or 0) - (_f(win20['OBV'].min()) or 0)
        if None not in (close, obv, close_5ago, obv_5ago) and close_5ago != 0 and obv_range > 0:
            price_chg = abs(close / close_5ago - 1)
            obv_chg = (obv - obv_5ago) / obv_range
            if price_chg < 0.02:
                if obv_chg > 0.15:
                    add_alert(3, 'bullish', "價平量增", "股價盤整但 OBV 持續走升，主力可能默默吸籌（偏多）")
                elif obv_chg < -0.15:
                    add_alert(3, 'bearish', "價平量減", "股價盤整但 OBV 持續下滑，籌碼鬆動，留意變盤（偏空）")

# ========== 九、BIAS 乖離率規則（ECF 心法） ==========
# 趨勢股門檻放寬 1.5 倍：多頭/空頭排列明確時，避免在趨勢行進中過早反向操作
relax = 1.5 if (trend_relax and trend_state in ("多頭排列", "空頭排列")) else 1.0
b10_lim = bias10_limit * relax
b20_lim = bias20_limit * relax
relax_note = f"（趨勢股門檻已放寬 1.5 倍 → ±{b10_lim:.1f}% / ±{b20_lim:.1f}%）" if relax > 1 else ""

bias_state10 = bias_state20 = "正常"
bias_over = bias_under = False
if None not in (bias10, bias20):
    # 1. 正乖離過大
    if bias10 > b10_lim or bias20 > b20_lim:
        bias_over = True
        add_alert(2, 'bearish', "BIAS 正乖離過大",
                  f"BIAS10={bias10:+.1f}%、BIAS20={bias20:+.1f}%，超過警戒（+{b10_lim:.1f}%/+{b20_lim:.1f}%），短線過熱、回檔修正機率高{relax_note}")
    # 2. 負乖離過大
    elif bias10 < -b10_lim or bias20 < -b20_lim:
        bias_under = True
        add_alert(2, 'bullish', "BIAS 負乖離過大",
                  f"BIAS10={bias10:+.1f}%、BIAS20={bias20:+.1f}%，低於警戒（-{b10_lim:.1f}%/-{b20_lim:.1f}%），短線超跌、反彈機率高{relax_note}")
    # 3/4. 乖離偏高（警戒值的 5/7 比例區間）
    elif bias10 > b10_lim * 5 / 7:
        add_alert(3, 'warning', "BIAS 正乖離偏高", f"BIAS10={bias10:+.1f}% 接近警戒區，追高風險升高")
    elif bias10 < -b10_lim * 5 / 7:
        add_alert(3, 'warning', "BIAS 負乖離偏高", f"BIAS10={bias10:+.1f}% 接近超跌區，留意反彈契機")

    # 狀態標籤
    def bias_label(v, lim):
        if v > lim: return "過熱"
        if v > lim * 5 / 7: return "偏高"
        if v < -lim: return "超跌"
        if v < -lim * 5 / 7: return "偏低"
        return "正常"
    bias_state10 = bias_label(bias10, b10_lim)
    bias_state20 = bias_label(bias20, b20_lim)

# ========== 十、指標共振驗證（ECF 心法） ==========
kd_overbought = (k is not None and d is not None and k > 80 and d > 80)
if bias_over and (kd_overbought or rsi_overbought):
    add_alert(1, 'bearish', "共振：乖離過大 + KD/RSI 超買", "BIAS 正乖離過大且擺盪指標同步超買，回檔訊號可信度提升")
if bias_under and obv_bottom_div:
    add_alert(1, 'bullish', "共振：負乖離過大 + OBV 底背離", "BIAS 負乖離過大且 OBV 底背離，反彈訊號可信度提升")

# ==========================================
# 7. 報告輸出（ECF 格式）
# ==========================================
bull_n = sum(1 for a in alerts if a[1] == 'bullish')
bear_n = sum(1 for a in alerts if a[1] == 'bearish')
warn_n = sum(1 for a in alerts if a[1] == 'warning')
total_score = int(np.clip(sum(a[4] for a in alerts), -100, 100))

st.markdown(f"""
═══════════════════════════════════════
### 📊 {stock_id} 智能警示報告
═══════════════════════════════════════
""")

# --- 警示摘要 ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 看漲訊號", f"{bull_n} 個")
c2.metric("🟢 看跌訊號", f"{bear_n} 個")
c3.metric("🟡 警告訊號", f"{warn_n} 個")
c4.metric("綜合評分", f"{total_score:+d}", help="優先級1 ±30、優先級2 ±15、優先級3 ±5 加總，範圍 -100 至 +100")

# --- 詳細警示清單 ---
st.subheader("📋 詳細警示清單")
P_TITLE = {1: "🔴 優先級 1（重要）", 2: "🟡 優先級 2（注意）", 3: "🟢 優先級 3（觀察）"}
for p in (1, 2, 3):
    group = [a for a in alerts if a[0] == p]
    if not group:
        continue
    st.markdown(f"**{P_TITLE[p]}**")
    for _, tag, title, detail, _ in group:
        st.markdown(f"- {TAG_ICON[tag]} **{title}** `{TAG_TEXT[tag]}`\n   - {detail}")
if not alerts:
    st.info("目前無任何警示訊號，盤勢平穩。")

# --- 關鍵數據 ---
st.subheader("📊 關鍵數據")
key_rows = [
    ("壓力位（近20日高）", f"{pressure:.1f} 元" if pressure else "N/A"),
    ("支撐位（近20日低）", f"{support:.1f} 元" if support else "N/A"),
    ("成交量比（今日/20日均量）", f"{vol_ratio:.1f} 倍" if vol_ratio else "N/A"),
    ("OBV 狀態", obv_state),
    ("BIAS10", f"{bias10:+.1f}%（{bias_state10}）" if bias10 is not None else "N/A"),
    ("BIAS20", f"{bias20:+.1f}%（{bias_state20}）" if bias20 is not None else "N/A"),
    ("均線狀態", trend_state),
]
st.table(pd.DataFrame(key_rows, columns=["項目", "數值"]))

# --- 操作建議 ---
st.subheader("💡 操作建議")
sl_pct = (support / close - 1) * 100 if (support and close) else None
tp_pct = (pressure / close - 1) * 100 if (pressure and close) else None
support_txt = f"{support:.1f}" if support is not None else "N/A"
pressure_txt = f"{pressure:.1f}" if pressure is not None else "N/A"
level_text = ""
if None not in (close, support, pressure, sl_pct, tp_pct):
    level_text = (f"\n- 參考進場：{close:.1f} 元附近（現價）"
                  f"\n- 參考停損：{support:.1f} 元（{sl_pct:+.1f}%，20 日支撐）"
                  f"\n- 參考停利：{pressure:.1f} 元（{tp_pct:+.1f}%，20 日壓力）")

if total_score >= 40:
    st.error(f"**綜合研判：偏多 → 可留意進場 / 續抱**\n\n"
             f"多數指標站在多方（評分 {total_score:+d}），量能（OBV：{obv_state}）與乖離（BIAS10 {bias_state10}）"
             f"顯示動能與價位尚屬健康。空手者可分批佈局，已持有者續抱並守住支撐。{level_text}")
elif total_score >= 15:
    st.error(f"**綜合研判：偏多觀察 → 分批佈局、勿追高**\n\n"
             f"多方訊號略佔上風（評分 {total_score:+d}），但共振強度不足。"
             f"留意 BIAS 乖離（{bias_state10}）勿在乖離偏高時追價，等回測均線再進場。{level_text}")
elif total_score <= -40:
    st.success(f"**綜合研判：偏空 → 獲利了結 / 觀望**\n\n"
               f"多數指標轉弱（評分 {total_score:+d}），量能（OBV：{obv_state}）不支持續漲。"
               f"持有者分批獲利了結或減碼，空手者觀望勿接刀。{level_text}")
elif total_score <= -15:
    st.success(f"**綜合研判：偏空觀察 → 減碼防守**\n\n"
               f"空方訊號略佔上風（評分 {total_score:+d}）。持有者拉高警覺、跌破支撐 {support_txt} 元宜出場；"
               f"若 BIAS 負乖離過大（{bias_state10}）可留意超跌反彈，但僅宜短打。")
else:
    st.warning(f"**綜合研判：中性 → 區間整理 / 等待訊號**\n\n"
               f"多空訊號拉鋸（評分 {total_score:+d}），方向未明。建議等待均線/MACD 與 OBV 量能出現一致訊號"
               f"（共振）後再決定進出場，區間操作以支撐 {support_txt}、壓力 {pressure_txt} 為界。")

# --- 風險提示 ---
st.subheader("⚠️ 風險提示")
risks = []
if obv_state == "頂背離":
    risks.append("OBV 頂背離：股價新高缺乏量能支撐，留意假突破與快速拉回。")
if obv_state == "量價齊跌":
    risks.append("量價齊跌：資金持續流出，反彈宜視為逃命波而非回升。")
if bias_over:
    risks.append(f"BIAS 正乖離過大（BIAS10 {bias10:+.1f}%）：短線漲幅偏離均線過遠，追高風險極高。")
if kd_overbought or rsi_overbought:
    risks.append("KD/RSI 處於超買區：擺盪指標過熱，隨時可能技術性回檔。")
if vol_ratio is not None and vol_ratio >= 2 and close is not None and close_prev is not None and close < close_prev:
    risks.append("爆量下跌：大量賣壓出籠，若隔日無法收復失地，趨勢恐轉空。")
if trend_state == "空頭排列":
    risks.append("均線空頭排列：任何反彈均面臨均線反壓，操作以短打為限。")
risks.append("本報告僅依技術面規則自動產生，未涵蓋籌碼面（三大法人）與消息面，重大事件（財報、法說會）前後波動可能加大。")
risks.append("指標多為落後/同步訊號，務必搭配停損紀律（如跌破 20 日均線或支撐位出場）。")
for i, r in enumerate(risks, 1):
    st.markdown(f"{i}. {r}")

st.divider()
st.info("💡 ECF 心法提醒：單一指標不可盡信，重點在**多指標共振**——趨勢（均線/MACD）為主、擺盪（KD/RSI/BIAS）為輔、量能（OBV）驗證。"
        "BIAS 乖離在強趨勢股應放寬門檻，避免過早反向操作。")
