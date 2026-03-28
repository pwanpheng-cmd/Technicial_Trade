"""
TECHSCAN SET — Full App v2.0
==============================
ติดตั้ง:
    pip install streamlit yfinance pandas_ta plotly anthropic

รัน:
    streamlit run techscan_app.py

ตั้งค่า API Key:
    สร้างไฟล์ .streamlit/secrets.toml แล้วใส่:
    ANTHROPIC_API_KEY = "sk-ant-xxxx"
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
from datetime import datetime
import time

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(page_title="TECHSCAN SET", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Sarabun:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'Sarabun','IBM Plex Mono',monospace; background:#080c14; color:#c8d8e8; }
.main-title  { font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:700; color:#00d4ff; letter-spacing:.1em; }
.main-sub    { font-size:11px; color:#4a7a9a; letter-spacing:.15em; }
.sec-label   { font-size:10px; color:#4a6a8a; letter-spacing:.15em; margin:14px 0 8px; }
.badge-strong-buy  { background:#00e67633; border:2px solid #00e676; color:#00e676; padding:6px 18px; border-radius:20px; font-weight:700; font-size:15px; display:inline-block; }
.badge-buy         { background:#00e67618; border:1px solid #00e67660; color:#00e676; padding:6px 18px; border-radius:20px; font-weight:700; font-size:15px; display:inline-block; }
.badge-hold        { background:#ffd60018; border:1px solid #ffd60060; color:#ffd600; padding:6px 18px; border-radius:20px; font-weight:700; font-size:15px; display:inline-block; }
.badge-sell        { background:#ff525218; border:1px solid #ff525260; color:#ff5252; padding:6px 18px; border-radius:20px; font-weight:700; font-size:15px; display:inline-block; }
.badge-strong-sell { background:#ff525233; border:2px solid #ff5252;   color:#ff5252; padding:6px 18px; border-radius:20px; font-weight:700; font-size:15px; display:inline-block; }
.sig-row { display:flex; justify-content:space-between; align-items:center; padding:8px 14px; border-radius:6px; margin-bottom:6px; background:#0a1120; border:1px solid #1a2535; }
.ai-box  { background:linear-gradient(135deg,#1a0a2e,#0a0f1c); border:1px solid #9c4dcc44; border-radius:12px; padding:20px 24px; }
section[data-testid="stSidebar"] { background:#0a1120 !important; border-right:1px solid #1e3045 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════
WATCHLIST = {
    "AOT":"AOT.BK","PTT":"PTT.BK","CPALL":"CPALL.BK","SCB":"SCB.BK",
    "KBANK":"KBANK.BK","BBL":"BBL.BK","GULF":"GULF.BK","DELTA":"DELTA.BK",
    "TRUE":"TRUE.BK","MINT":"MINT.BK","AWC":"AWC.BK","CPN":"CPN.BK",
    "ADVANC":"ADVANC.BK","BDMS":"BDMS.BK","SCC":"SCC.BK","BH":"BH.BK",
}
PERIOD_MAP = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y"}
INTERVAL_MAP = {"รายวัน (1D)":"1d","รายชั่วโมง (1H)":"1h"}
PERIOD_MAP_HOURLY = {"5D":"5d","1M":"1mo","3M":"3mo","6M":"6mo"}

# ══════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_50"]    = ta.ema(df["Close"], length=50)
    df["EMA_200"]   = ta.ema(df["Close"], length=200)
    df["RSI_14"]    = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df["MACD"]      = macd["MACD_12_26_9"]
    df["MACD_sig"]  = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df["Stoch_K"]   = stoch["STOCHk_14_3_3"]
    df["Stoch_D"]   = stoch["STOCHd_14_3_3"]
    bb = ta.bbands(df["Close"], length=20)
    df["BB_upper"]  = bb["BBU_20_2.0_2.0"]
    df["BB_mid"]    = bb["BBM_20_2.0_2.0"]
    df["BB_lower"]  = bb["BBL_20_2.0_2.0"]
    df["ATR_14"]    = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_SMA20"]
    df["TP"]        = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"]      = (df["TP"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    df["OBV"]       = ta.obv(df["Close"], df["Volume"])
    try:
        adx_df       = ta.adx(df["High"], df["Low"], df["Close"])
        df["ADX"]    = adx_df["ADX_14"]
    except Exception:
        df["ADX"]    = 20.0
    fib_high       = df["High"].rolling(60).max()
    fib_low        = df["Low"].rolling(60).min()
    diff           = fib_high - fib_low
    df["Fib_382"]  = fib_high - diff * 0.382
    df["Fib_500"]  = fib_high - diff * 0.500
    df["Fib_618"]  = fib_high - diff * 0.618
    df["Fib_high"] = fib_high
    df["Fib_low"]  = fib_low
    return df


def detect_patterns(df: pd.DataFrame) -> list:
    patterns = []
    for i in range(max(0, len(df)-5), len(df)):
        c    = df.iloc[i]
        body = abs(c["Close"] - c["Open"])
        rng  = c["High"] - c["Low"]
        bull = c["Close"] > c["Open"]
        uw   = c["High"] - max(c["Open"], c["Close"])
        lw   = min(c["Open"], c["Close"]) - c["Low"]
        date = df.index[i].strftime("%d/%m")
        if rng > 0 and body/rng < 0.1:
            patterns.append({"date":date,"name":"Doji","signal":"ลังเล รอยืนยัน","color":"#ffd600"})
        elif not bull and lw > body*2 and uw < body*0.5:
            patterns.append({"date":date,"name":"Hammer","signal":"Bullish Reversal ↑","color":"#00e676"})
        elif bull and uw > body*2 and lw < body*0.5:
            patterns.append({"date":date,"name":"Shooting Star","signal":"Bearish Reversal ↓","color":"#ff5252"})
        elif i > 0:
            p = df.iloc[i-1]
            if bull and c["Close"] > p["Open"] and c["Open"] < p["Close"] and body > abs(p["Close"]-p["Open"]):
                patterns.append({"date":date,"name":"Bullish Engulfing","signal":"แรงซื้อครอบ — BUY","color":"#00e676"})
            elif not bull and c["Close"] < p["Open"] and c["Open"] > p["Close"] and body > abs(p["Close"]-p["Open"]):
                patterns.append({"date":date,"name":"Bearish Engulfing","signal":"แรงขายครอบ — SELL","color":"#ff5252"})
    return patterns


def get_regime(df: pd.DataFrame) -> tuple:
    last   = df.iloc[-1]
    adx_v  = last["ADX"] if pd.notna(last["ADX"]) else 20
    atr_p  = last["ATR_14"] / last["Close"] * 100 if last["Close"] > 0 and pd.notna(last["ATR_14"]) else 2
    bb_w   = (last["BB_upper"] - last["BB_lower"]) / last["BB_mid"] * 100 if pd.notna(last["BB_mid"]) and last["BB_mid"] > 0 else 5
    if adx_v > 30 and atr_p > 1.5:
        return "VOLATILE TREND","#ff9100","Trend แข็ง + Volatile สูง — EMA/MACD ดี แต่ Stop กว้างขึ้น"
    elif adx_v > 25:
        return "TRENDING","#00e676","Trend ชัด — ใช้ EMA Cross / MACD ได้ผลดีที่สุด"
    elif bb_w < 3.5:
        return "SQUEEZE","#ea80fc","BB แคบ — กำลังสะสมพลัง Breakout กำลังมา รอสัญญาณ"
    else:
        return "SIDEWAY","#ffd600","ไม่มีทิศทาง — ใช้ RSI Bounce + BB Range"


def detect_divergence(df: pd.DataFrame) -> list:
    divs = []
    w    = min(30, len(df))
    sub  = df.tail(w).copy()
    try:
        ph1 = sub["High"].iloc[:w//2].idxmax()
        ph2 = sub["High"].iloc[w//2:].idxmax()
        if sub.loc[ph2,"High"] > sub.loc[ph1,"High"]:
            if pd.notna(sub.loc[ph2,"RSI_14"]) and sub.loc[ph2,"RSI_14"] < sub.loc[ph1,"RSI_14"]:
                divs.append({"type":"Bearish","ind":"RSI","action":"SELL","color":"#ff5252","desc":"ราคาทำ Higher High แต่ RSI ทำ Lower High — แรงซื้อลดลง"})
            if pd.notna(sub.loc[ph2,"MACD"]) and sub.loc[ph2,"MACD"] < sub.loc[ph1,"MACD"]:
                divs.append({"type":"Bearish","ind":"MACD","action":"SELL","color":"#ff5252","desc":"ราคาทำ Higher High แต่ MACD ทำ Lower High"})
        pl1 = sub["Low"].iloc[:w//2].idxmin()
        pl2 = sub["Low"].iloc[w//2:].idxmin()
        if sub.loc[pl2,"Low"] < sub.loc[pl1,"Low"]:
            if pd.notna(sub.loc[pl2,"RSI_14"]) and sub.loc[pl2,"RSI_14"] > sub.loc[pl1,"RSI_14"]:
                divs.append({"type":"Bullish","ind":"RSI","action":"BUY","color":"#00e676","desc":"ราคาทำ Lower Low แต่ RSI ทำ Higher Low — แรงขายหมดแล้ว"})
            if pd.notna(sub.loc[pl2,"MACD"]) and sub.loc[pl2,"MACD"] > sub.loc[pl1,"MACD"]:
                divs.append({"type":"Bullish","ind":"MACD","action":"BUY","color":"#00e676","desc":"ราคาทำ Lower Low แต่ MACD ทำ Higher Low"})
    except Exception:
        pass
    return divs


def detect_breakout(df: pd.DataFrame) -> list:
    if len(df) < 22:
        return [{"type":"ข้อมูลน้อยเกิน","desc":"ต้องการข้อมูลอย่างน้อย 22 วัน","color":"#4a6a8a","action":"HOLD","confirmed":False}]
    last      = df.iloc[-1]
    prev20    = df.iloc[-21:-1]
    res20     = prev20["High"].max()
    sup20     = prev20["Low"].min()
    vol_avg   = last["Vol_SMA20"] if pd.notna(last["Vol_SMA20"]) else last["Volume"]
    vol_ratio = last["Volume"] / vol_avg if vol_avg > 0 else 1
    alerts    = []
    if last["Close"] > res20:
        if vol_ratio >= 1.5:
            alerts.append({"type":"✅ BREAKOUT UP","desc":f"ราคา {last['Close']:.2f} ทะลุ Resistance {res20:.2f} + Volume {vol_ratio:.1f}×","color":"#00e676","action":"BUY","confirmed":True})
        else:
            alerts.append({"type":"⚠ FALSE BREAK?","desc":f"ราคาทะลุ {res20:.2f} แต่ Volume แห้ง ({vol_ratio:.1f}×) — ระวัง False Breakout","color":"#ffd600","action":"WATCH","confirmed":False})
    elif last["Close"] < sup20 and vol_ratio >= 1.5:
        alerts.append({"type":"🔴 BREAKDOWN","desc":f"ราคา {last['Close']:.2f} หลุด Support {sup20:.2f} + Volume หนาแน่น","color":"#ff5252","action":"SELL","confirmed":True})
    elif vol_ratio >= 2:
        alerts.append({"type":"📊 VOLUME SPIKE","desc":f"Volume {vol_ratio:.1f}× สูงผิดปกติ ไม่มี Breakout — อาจ Smart Money","color":"#ea80fc","action":"WATCH","confirmed":False})
    else:
        alerts.append({"type":"— ปกติ","desc":f"ไม่มี Breakout · Volume {vol_ratio:.1f}×","color":"#4a6a8a","action":"HOLD","confirmed":False})
    return alerts


def gen_signals(df: pd.DataFrame, ticker: str) -> dict:
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = last["Close"]
    signals, score = [], 0

    rsi = last["RSI_14"] if pd.notna(last["RSI_14"]) else 50
    if   rsi < 30: signals.append({"g":"Momentum","i":"RSI(14)","v":f"{rsi:.1f}","s":"OVERSOLD","a":"BUY","sc":+2}); score+=2
    elif rsi < 40: signals.append({"g":"Momentum","i":"RSI(14)","v":f"{rsi:.1f}","s":"ใกล้ Oversold","a":"WATCH","sc":+1}); score+=1
    elif rsi > 70: signals.append({"g":"Momentum","i":"RSI(14)","v":f"{rsi:.1f}","s":"OVERBOUGHT","a":"SELL","sc":-2}); score-=2
    elif rsi > 60: signals.append({"g":"Momentum","i":"RSI(14)","v":f"{rsi:.1f}","s":"ใกล้ Overbought","a":"WATCH","sc":-1}); score-=1
    else:          signals.append({"g":"Momentum","i":"RSI(14)","v":f"{rsi:.1f}","s":"NEUTRAL","a":"HOLD","sc":0})

    if pd.notna(last["MACD"]) and pd.notna(prev["MACD"]):
        cu = last["MACD"] > last["MACD_sig"] and prev["MACD"] <= prev["MACD_sig"]
        cd = last["MACD"] < last["MACD_sig"] and prev["MACD"] >= prev["MACD_sig"]
        if   cu: signals.append({"g":"Momentum","i":"MACD","v":f"{last['MACD']:.3f}","s":"Bullish Cross ✦","a":"BUY","sc":+2}); score+=2
        elif cd: signals.append({"g":"Momentum","i":"MACD","v":f"{last['MACD']:.3f}","s":"Bearish Cross ✦","a":"SELL","sc":-2}); score-=2
        elif last["MACD"] > last["MACD_sig"]: signals.append({"g":"Momentum","i":"MACD","v":f"{last['MACD']:.3f}","s":"Bullish Zone","a":"HOLD","sc":+1}); score+=1
        else: signals.append({"g":"Momentum","i":"MACD","v":f"{last['MACD']:.3f}","s":"Bearish Zone","a":"HOLD","sc":-1}); score-=1

    if pd.notna(last["Stoch_K"]):
        sk, sd = last["Stoch_K"], last["Stoch_D"]
        if   sk < 20 and sk > sd: signals.append({"g":"Momentum","i":"Stochastic","v":f"K={sk:.1f}","s":"Oversold+CrossUp","a":"BUY","sc":+2}); score+=2
        elif sk > 80 and sk < sd: signals.append({"g":"Momentum","i":"Stochastic","v":f"K={sk:.1f}","s":"Overbought+CrossDn","a":"SELL","sc":-2}); score-=2
        else: signals.append({"g":"Momentum","i":"Stochastic","v":f"K={sk:.1f}","s":"NEUTRAL","a":"HOLD","sc":0})

    if pd.notna(last["EMA_50"]) and pd.notna(last["EMA_200"]):
        gc = last["EMA_50"] > last["EMA_200"] and prev["EMA_50"] <= prev["EMA_200"]
        dc = last["EMA_50"] < last["EMA_200"] and prev["EMA_50"] >= prev["EMA_200"]
        if   gc: signals.append({"g":"Trend","i":"EMA Cross","v":f"50={last['EMA_50']:.2f}","s":"Golden Cross ★","a":"STRONG BUY","sc":+3}); score+=3
        elif dc: signals.append({"g":"Trend","i":"EMA Cross","v":f"50={last['EMA_50']:.2f}","s":"Death Cross ★","a":"STRONG SELL","sc":-3}); score-=3
        elif last["EMA_50"] > last["EMA_200"]: signals.append({"g":"Trend","i":"EMA Cross","v":f"50={last['EMA_50']:.2f}","s":"Uptrend","a":"BUY","sc":+1}); score+=1
        else: signals.append({"g":"Trend","i":"EMA Cross","v":f"50={last['EMA_50']:.2f}","s":"Downtrend","a":"SELL","sc":-1}); score-=1

    if pd.notna(last["ADX"]):
        adv = last["ADX"]
        ts  = "แข็งมาก" if adv>40 else "แข็ง" if adv>25 else "ปานกลาง" if adv>20 else "Sideway"
        sc  = +1 if adv > 25 else 0
        signals.append({"g":"Trend","i":"ADX","v":f"{adv:.1f}","s":f"Trend {ts}","a":"BUY" if sc else "HOLD","sc":sc})
        score += sc

    if pd.notna(last["BB_lower"]) and pd.notna(last["BB_upper"]) and last["BB_upper"] != last["BB_lower"]:
        bp = (close - last["BB_lower"]) / (last["BB_upper"] - last["BB_lower"]) * 100
        if   close <= last["BB_lower"]: signals.append({"g":"Volatility","i":"Bollinger","v":f"%B={bp:.0f}%","s":"แตะ Lower Band","a":"BUY","sc":+2}); score+=2
        elif close >= last["BB_upper"]: signals.append({"g":"Volatility","i":"Bollinger","v":f"%B={bp:.0f}%","s":"แตะ Upper Band","a":"SELL","sc":-2}); score-=2
        else: signals.append({"g":"Volatility","i":"Bollinger","v":f"%B={bp:.0f}%","s":f"กลาง {bp:.0f}%","a":"HOLD","sc":0})

    if pd.notna(last["VWAP"]):
        pos = "เหนือ VWAP" if close > last["VWAP"] else "ต่ำกว่า VWAP"
        sc  = +1 if close > last["VWAP"] else -1
        signals.append({"g":"Volume","i":"VWAP","v":f"{last['VWAP']:.2f}","s":pos,"a":"BUY" if sc>0 else "SELL","sc":sc}); score+=sc

    if pd.notna(last["OBV"]) and len(df) >= 6:
        op5 = df.iloc[-6]["OBV"]
        ot  = "OBV ขึ้น (Accumulation)" if last["OBV"] > op5 else "OBV ลง (Distribution)"
        sc  = +1 if last["OBV"] > op5 else -1
        signals.append({"g":"Volume","i":"OBV","v":f"{last['OBV']/1e6:.1f}M","s":ot,"a":"BUY" if sc>0 else "SELL","sc":sc}); score+=sc

    overall = "STRONG BUY" if score>=6 else "BUY" if score>=3 else "HOLD" if score>=-2 else "SELL" if score>=-5 else "STRONG SELL"
    recent60 = df.tail(60)
    support    = round(recent60["Low"].min(), 2)
    resistance = round(recent60["High"].max(), 2)
    atr        = last["ATR_14"] if pd.notna(last["ATR_14"]) else (resistance - support) * 0.1
    stop_loss  = round(close - 1.5 * atr, 2)
    target1    = round(close + 1.5 * atr, 2)
    target2    = round(close + 3.0 * atr, 2)
    rr         = round((target1 - close) / (close - stop_loss), 2) if (close - stop_loss) > 0 else 0

    return {
        "ticker":ticker, "date":df.index[-1].strftime("%d/%m/%Y"),
        "close":round(close,2), "overall":overall, "score":score, "signals":signals,
        "support":support, "resistance":resistance, "stop_loss":stop_loss,
        "target1":target1, "target2":target2, "rr_ratio":rr,
        "volume_alert": last["Vol_ratio"] > 1.5 if pd.notna(last["Vol_ratio"]) else False,
        "rsi":round(rsi,1),
        "macd":round(last["MACD"],3) if pd.notna(last["MACD"]) else 0,
        "ema50":round(last["EMA_50"],2) if pd.notna(last["EMA_50"]) else 0,
        "ema200":round(last["EMA_200"],2) if pd.notna(last["EMA_200"]) else 0,
        "stoch_k":round(last["Stoch_K"],1) if pd.notna(last["Stoch_K"]) else 50,
        "adx":round(last["ADX"],1) if pd.notna(last["ADX"]) else 0,
        "vwap":round(last["VWAP"],2) if pd.notna(last["VWAP"]) else 0,
        "fib_382":round(last["Fib_382"],2) if pd.notna(last["Fib_382"]) else 0,
        "fib_500":round(last["Fib_500"],2) if pd.notna(last["Fib_500"]) else 0,
        "fib_618":round(last["Fib_618"],2) if pd.notna(last["Fib_618"]) else 0,
        "fib_high":round(last["Fib_high"],2) if pd.notna(last["Fib_high"]) else 0,
        "fib_low":round(last["Fib_low"],2) if pd.notna(last["Fib_low"]) else 0,
    }


def run_backtest(df: pd.DataFrame) -> dict:
    trades, in_trade, entry_price, entry_idx = [], False, 0, 0
    for i in range(15, len(df)-1):
        c, n, p = df.iloc[i], df.iloc[i+1], df.iloc[i-1]
        if pd.isna(c["RSI_14"]) or pd.isna(c["MACD"]) or pd.isna(c["EMA_50"]):
            continue
        entry    = not in_trade and c["RSI_14"] < 40 and c["MACD"] > c["MACD_sig"] and p["MACD"] <= p["MACD_sig"] and c["Close"] > c["EMA_50"]
        sl_hit   = in_trade and c["Low"] < entry_price * 0.95
        exit_sig = in_trade and (
            c["RSI_14"] > 65 or
            (pd.notna(p["MACD"]) and c["MACD"] < c["MACD_sig"] and p["MACD"] >= p["MACD_sig"]) or
            (pd.notna(c["EMA_50"]) and c["Close"] < c["EMA_50"])
        )
        if entry:
            in_trade, entry_price, entry_idx = True, n["Open"], i+1
        if (sl_hit or exit_sig) and in_trade:
            ep  = entry_price * 0.95 if sl_hit else n["Open"]
            pnl = round((ep - entry_price) / entry_price * 100, 2)
            trades.append({"entry_date":df.index[entry_idx].strftime("%d/%m/%y"),
                           "exit_date":df.index[i+1].strftime("%d/%m/%y"),
                           "entry":round(entry_price,2),"exit":round(ep,2),
                           "pnl":pnl,"win":pnl>0,"sl":sl_hit,"bars":i+1-entry_idx})
            in_trade = False
    if not trades:
        return {"trades":[],"win_rate":0,"avg_win":0,"avg_loss":0,"rr":0,"expectancy":0,"total_return":0}
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    wr     = round(len(wins)/len(trades)*100, 1)
    aw     = round(sum(t["pnl"] for t in wins)/len(wins), 2)   if wins   else 0
    al     = round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0
    rr     = round(abs(aw/al), 2) if al != 0 else 0
    exp    = round((wr/100*aw) + ((1-wr/100)*al), 2)
    tot    = round(sum(t["pnl"] for t in trades), 2)
    return {"trades":trades,"win_rate":wr,"avg_win":aw,"avg_loss":al,"rr":rr,"expectancy":exp,"total_return":tot}


# ══════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════
def make_chart(df, ticker, show_ema, show_bb, show_vwap, show_fib):
    # ── Warning ถ้าข้อมูลสั้นเกิน ──────────────────
    if len(df) < 60:
        st.warning(f"⚠ ข้อมูลมีเพียง {len(df)} วัน — EMA 200 และ Fibonacci อาจไม่แม่น แนะนำใช้ช่วงเวลา 1Y ขึ้นไป")
    nan_cols = [c for c in ["EMA_200","Fib_382"] if df[c].isna().all()]
    if nan_cols:
        st.info(f"ℹ {', '.join(nan_cols)} ต้องการข้อมูลมากกว่านี้ — เปลี่ยนช่วงเวลาเป็น 1Y หรือ 2Y")

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.15, 0.15, 0.14, 0.14],
        vertical_spacing=0.02,
        subplot_titles=("", "MACD", "RSI + Stochastic", "ADX", "Volume"),
    )

    # ── Row 1: Candlestick ──────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_fillcolor="#00e676", increasing_line_color="#00e676",
        decreasing_fillcolor="#ff5252", decreasing_line_color="#ff5252",
    ), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"],  name="EMA50",  line=dict(color="#ffd600",width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_200"], name="EMA200", line=dict(color="#ff9100",width=1.5,dash="dot")), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB↑", line=dict(color="#80d8ff",width=1,dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB↓", line=dict(color="#80d8ff",width=1,dash="dash"), fill="tonexty", fillcolor="rgba(128,216,255,0.04)", showlegend=False), row=1, col=1)
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", line=dict(color="#00bcd4",width=1.5,dash="dashdot")), row=1, col=1)
    if show_fib:
        last = df.iloc[-1]
        for level, color, label in [
            (last["Fib_382"],"#ffd60080","38.2%"),
            (last["Fib_500"],"#00d4ff80","50%"),
            (last["Fib_618"],"#ea80fc80","61.8%"),
        ]:
            if pd.notna(level):
                fig.add_hline(y=level, line_color=color, line_dash="dash", line_width=1,
                              annotation_text=label, annotation_font_color=color, row=1, col=1)

    # ── Row 2: MACD ────────────────────────────────
    colors_macd = ["#00e676" if v >= 0 else "#ff5252" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=colors_macd, opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],     name="MACD",   line=dict(color="#00d4ff",width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal", line=dict(color="#ff9100",width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_color="#1e3045", line_width=1, row=2, col=1)

    # ── Row 3: RSI + Stochastic ────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"],  name="RSI",     line=dict(color="#ea80fc",width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch%K", line=dict(color="#00d4ff",width=1,dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch%D", line=dict(color="#ffd600",width=1,dash="dot")), row=3, col=1)
    fig.add_hline(y=70, line_color="#ff5252", line_dash="dash", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_color="#00e676", line_dash="dash", line_width=1, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", row=3, col=1)

    # ── Row 4: ADX ─────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                              line=dict(color="#00bcd4",width=1.5),
                              fill="tozeroy", fillcolor="rgba(0,188,212,0.06)"), row=4, col=1)
    fig.add_hline(y=25, line_color="#ffd600", line_dash="dash", line_width=1, row=4, col=1)
    fig.add_hline(y=40, line_color="#ff9100", line_dash="dash", line_width=1, row=4, col=1)

    # ── Row 5: Volume ──────────────────────────────
    vol_colors = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=vol_colors, opacity=0.7), row=5, col=1)
    # SMA20 volume line
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_SMA20"], name="Vol SMA20",
                              line=dict(color="#ffd600",width=1.5,dash="dot"), showlegend=True), row=5, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#080c14", plot_bgcolor="#0a1020",
        font=dict(family="IBM Plex Mono", color="#7a9ab0", size=10),
        legend=dict(bgcolor="#0d1828", bordercolor="#1e3045", borderwidth=1,
                    font=dict(size=9), orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=24, b=10),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="#1a2535", row=i, col=1)
        fig.update_yaxes(gridcolor="#1a2535", row=i, col=1)
    fig.update_yaxes(title_text=ticker,   title_font_size=9, row=1, col=1)
    fig.update_yaxes(title_text="MACD",   title_font_size=9, row=2, col=1)
    fig.update_yaxes(title_text="RSI/Stoch", title_font_size=9, row=3, col=1)
    fig.update_yaxes(title_text="ADX",    title_font_size=9, row=4, col=1)
    fig.update_yaxes(title_text="Volume", title_font_size=9, row=5, col=1)
    return fig


# ══════════════════════════════════════════════
# AI SUMMARY
# ══════════════════════════════════════════════
def ai_summary(result, patterns, regime, divs, breakouts, api_key) -> str:
    sig_text = "\n".join([f"  {s['i']}: {s['v']} → {s['a']} ({s['s']})" for s in result["signals"]])
    pat_text = ", ".join([f"{p['name']} ({p['signal']})" for p in patterns]) or "ไม่พบ"
    div_text = ", ".join([f"{d['type']} Divergence จาก {d['ind']}" for d in divs]) or "ไม่พบ"
    brk_text = breakouts[0]["desc"] if breakouts else "-"
    prompt = f"""คุณเป็นนักวิเคราะห์หุ้นเทคนิคมืออาชีพ วิเคราะห์หุ้น {result['ticker']} ณ {result['date']}
ราคา: {result['close']} บาท | สัญญาณ: {result['overall']} | Score: {result['score']:+d}
Market Regime: {regime[0]} — {regime[2]}

Indicators:
{sig_text}

Patterns: {pat_text}
Divergence: {div_text}
Breakout: {brk_text}
Fibonacci: 38.2%={result['fib_382']} | 50%={result['fib_500']} | 61.8%={result['fib_618']}
Support: {result['support']} | Resistance: {result['resistance']}
Stop Loss: {result['stop_loss']} | T1: {result['target1']} | T2: {result['target2']} | R/R: 1:{result['rr_ratio']}

สรุปภาษาไทย 5 หัวข้อ (กระชับ ไม่เกิน 250 คำ):
1. ภาพรวม Regime และสัญญาณ
2. จุดเข้าซื้อ / Stop Loss / เป้าหมาย
3. Fibonacci Level สำคัญ
4. Divergence & Pattern ที่พบ
5. คำแนะนำสำหรับนักลงทุน"""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=1000,
                                   messages=[{"role":"user","content":prompt}])
    return msg.content[0].text


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◈ TECHSCAN SET v2.0")
    st.markdown("---")
    api_key = ""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.success("✓ API Key โหลดแล้ว", icon="🔑")
    except Exception:
        api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.markdown("---")
    st.markdown("### 🔍 เลือกหุ้น")
    mode = st.radio("วิธีเลือก", ["รายการ", "พิมพ์เอง"], horizontal=True)
    if mode == "รายการ":
        name   = st.selectbox("หุ้น", list(WATCHLIST.keys()))
        ticker = WATCHLIST[name]
    else:
        ticker = st.text_input("Ticker", value="AOT.BK").strip().upper()
    interval_label = st.radio("📐 Chart Scale", list(INTERVAL_MAP.keys()), horizontal=True)
    interval = INTERVAL_MAP[interval_label]
    if interval == "1h":
        period_label = st.select_slider("ช่วงเวลา", list(PERIOD_MAP_HOURLY.keys()), value="1M")
        period = PERIOD_MAP_HOURLY[period_label]
    else:
        period_label = st.select_slider("ช่วงเวลา", list(PERIOD_MAP.keys()), value="1Y")
        period = PERIOD_MAP[period_label]
    st.markdown("---")
    st.markdown("### 📊 Overlay")
    show_ema  = st.checkbox("EMA 50/200", value=True)
    show_bb   = st.checkbox("Bollinger Bands", value=True)
    show_vwap = st.checkbox("VWAP", value=True)
    show_fib  = st.checkbox("Fibonacci", value=False)
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (5 นาที)", value=False)
    analyze_btn  = st.button("🔎 วิเคราะห์", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("### 📋 Watchlist")
    for k in list(WATCHLIST.keys())[:10]:
        st.markdown(f"<span style='color:#4a6a8a;font-size:11px'>{k} · {WATCHLIST[k]}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#0a1628,#0d1f3c);border:1px solid #1e3a5f;border-radius:12px;padding:16px 22px;margin-bottom:14px'>
    <div class='main-title'>◈ TECHSCAN SET v2.0</div>
    <div class='main-sub'>REAL DATA · yfinance ~15min delay · RSI · MACD · EMA · BB · Stoch · ADX · VWAP · OBV · Fibonacci · Patterns · Regime · Divergence · Breakout · Backtest · Position Sizing · Claude AI</div>
</div>
""", unsafe_allow_html=True)

if not analyze_btn and "last_ticker" not in st.session_state:
    st.markdown("""<div style='text-align:center;padding:70px 20px;color:#2a4a6a'>
        <div style='font-size:48px;margin-bottom:14px'>📈</div>
        <div style='font-size:18px;color:#4a7a9a;font-family:IBM Plex Mono'>เลือกหุ้นและกด วิเคราะห์</div>
        <div style='font-size:11px;margin-top:8px;color:#2a4a6a'>ครบ 11 Tab · ข้อมูลจริงจาก Yahoo Finance</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if analyze_btn:
    st.session_state["last_ticker"] = ticker
    st.session_state["last_period"] = period
    st.session_state["last_interval"] = interval
    st.cache_data.clear()

use_ticker   = st.session_state.get("last_ticker", ticker)
use_period   = st.session_state.get("last_period", period)
use_interval = st.session_state.get("last_interval", interval)

if auto_refresh:
    st.markdown(f"<div style='font-size:10px;color:#4a6a8a'>🔄 Auto-refresh · อัปเดต: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

with st.spinner(f"🔄 กำลังดึงข้อมูล {use_ticker}..."):
    df_raw = fetch_stock(use_ticker, use_period, use_interval)

if df_raw.empty:
    st.error(f"❌ ไม่พบข้อมูล {use_ticker} — ตรวจสอบ ticker (หุ้น SET ต้องลงท้าย .BK)")
    st.stop()

df        = calc_indicators(df_raw)
result    = gen_signals(df, use_ticker)
patterns  = detect_patterns(df)
regime    = get_regime(df)
divs      = detect_divergence(df)
breakouts = detect_breakout(df)
bt        = run_backtest(df)

# ── HEADER ────────────────────────────────────
h1, h2, h3, h4, h5 = st.columns([2, 1.5, 1.5, 1.5, 1])
with h1:
    st.markdown(f"<div style='font-size:20px;font-weight:700;color:#e8f4ff;font-family:IBM Plex Mono'>{result['ticker']}</div>", unsafe_allow_html=True)
    interval_disp = "1H" if use_interval == "1h" else "1D"
    bar_label = "แท่ง" if use_interval == "1h" else "วัน"
    st.markdown(f"<div style='font-size:10px;color:#4a6a8a'>{period_label} · {interval_disp} · {len(df)} {bar_label} · delay ~15min</div>", unsafe_allow_html=True)
with h2:
    st.markdown(f"<div style='font-size:24px;font-weight:700;color:#00d4ff;font-family:IBM Plex Mono'>{result['close']}</div><div style='font-size:10px;color:#4a6a8a'>THB</div>", unsafe_allow_html=True)
with h3:
    bc = {"STRONG BUY":"badge-strong-buy","BUY":"badge-buy","HOLD":"badge-hold","SELL":"badge-sell","STRONG SELL":"badge-strong-sell"}.get(result["overall"],"badge-hold")
    st.markdown(f"<div class='{bc}'>{result['overall']}</div><div style='font-size:10px;color:#4a6a8a;margin-top:4px'>Score: {result['score']:+d} / 16</div>", unsafe_allow_html=True)
with h4:
    rc = regime[1]
    st.markdown(f"<div style='background:{rc}18;border:1px solid {rc}40;color:{rc};border-radius:10px;padding:4px 10px;font-size:10px;display:inline-block'>{regime[0]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:9px;color:#4a6a8a;margin-top:4px'>ADX={result['adx']}</div>", unsafe_allow_html=True)
with h5:
    st.markdown(f"<div style='font-size:10px;color:#4a6a8a'>อัปเดต<br><span style='color:#c8d8e8'>{result['date']}</span></div>", unsafe_allow_html=True)
    if result["volume_alert"]:
        st.markdown("<div style='color:#ffd600;font-size:10px'>⚠ Volume สูง</div>", unsafe_allow_html=True)

st.markdown("---")

# ── TABS ──────────────────────────────────────
tabs = st.tabs(["📈 กราฟ","◉ Signals","🌊 Regime","⚡ Divergence","🔥 Breakout","⊹ Fibonacci","⬡ Patterns","⚙ Backtest","💰 Position","🔍 Scanner","✦ AI"])

# ── TAB 0: CHART ──────────────────────────────
with tabs[0]:
    st.plotly_chart(make_chart(df, use_ticker, show_ema, show_bb, show_vwap, show_fib), use_container_width=True)
    st.markdown("<div class='sec-label'>▸ OBV — On-Balance Volume</div>", unsafe_allow_html=True)
    obv60 = df[["OBV"]].tail(60)
    fig_o = go.Figure(go.Scatter(x=obv60.index, y=obv60["OBV"], fill="tozeroy",
                                  fillcolor="rgba(178,255,89,0.08)", line=dict(color="#b2ff59",width=1.5)))
    fig_o.update_layout(height=100, paper_bgcolor="#080c14", plot_bgcolor="#0a1020",
                         margin=dict(l=50,r=10,t=4,b=4), showlegend=False,
                         font=dict(color="#7a9ab0",size=9), xaxis=dict(gridcolor="#1a2535"), yaxis=dict(gridcolor="#1a2535"))
    st.plotly_chart(fig_o, use_container_width=True)

# ── TAB 1: SIGNALS ────────────────────────────
with tabs[1]:
    sl, sr = st.columns([3, 2])
    with sl:
        for g in list(dict.fromkeys(s["g"] for s in result["signals"])):
            st.markdown(f"<div class='sec-label'>▸ {g.upper()}</div>", unsafe_allow_html=True)
            for s in [x for x in result["signals"] if x["g"]==g]:
                ac = {"BUY":"#00e676","STRONG BUY":"#00e676","SELL":"#ff5252","STRONG SELL":"#ff5252"}.get(s["a"],"#ffd600")
                st.markdown(f"""<div class='sig-row'>
                    <span style='font-size:11px;color:#7a9ab0;width:110px'>{s['i']}</span>
                    <span style='font-size:10px;color:#4a6a8a;width:100px;font-family:monospace'>{s['v']}</span>
                    <span style='font-size:11px;color:#a0b8c8;flex:1;text-align:center'>{s['s']}</span>
                    <span style='font-size:11px;font-weight:700;color:{ac};width:90px;text-align:right'>{s['a']}</span>
                    <span style='font-size:10px;color:{"#00e676" if s["sc"]>0 else "#ff5252" if s["sc"]<0 else "#4a6a8a"};width:30px;text-align:right'>{s["sc"]:+d}</span>
                </div>""", unsafe_allow_html=True)
    with sr:
        st.markdown("<div class='sec-label'>▸ PRICE LEVELS</div>", unsafe_allow_html=True)
        st.metric("Resistance", result["resistance"])
        st.metric("Target 2",   result["target2"],   delta=f"+{result['target2']-result['close']:.2f}")
        st.metric("Target 1",   result["target1"],   delta=f"+{result['target1']-result['close']:.2f}")
        st.metric("Stop Loss",  result["stop_loss"], delta=f"{result['stop_loss']-result['close']:.2f}", delta_color="inverse")
        st.metric("Support",    result["support"])
        st.metric("R/R Ratio",  f"1:{result['rr_ratio']}")

# ── TAB 2: REGIME ─────────────────────────────
with tabs[2]:
    rc = regime[1]
    st.markdown(f"""<div style='background:{rc}18;border:2px solid {rc}50;border-radius:12px;padding:18px 22px;text-align:center;margin-bottom:14px'>
        <div style='font-size:9px;color:#4a6a8a;letter-spacing:.15em;margin-bottom:6px'>▸ MARKET REGIME</div>
        <div style='font-size:24px;font-weight:700;color:{rc}'>{regime[0]}</div>
        <div style='font-size:12px;color:#c8d8e8;margin-top:6px'>{regime[2]}</div>
    </div>""", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("ADX", f"{result['adx']}", help=">25=Trending, >40=แข็งมาก")
    last_row = df.iloc[-1]
    atr_pct  = last_row["ATR_14"]/last_row["Close"]*100 if pd.notna(last_row["ATR_14"]) else 0
    m2.metric("ATR%", f"{atr_pct:.2f}%", help=">1.5%=Volatile")
    bb_w = (last_row["BB_upper"]-last_row["BB_lower"])/last_row["BB_mid"]*100 if pd.notna(last_row["BB_mid"]) and last_row["BB_mid"]>0 else 0
    m3.metric("BB Width%", f"{bb_w:.1f}%", help="<3.5%=SQUEEZE")
    st.markdown("<div class='sec-label'>▸ กลยุทธ์ที่แนะนำตาม Regime</div>", unsafe_allow_html=True)
    for r, (use, avoid) in {
        "TRENDING":       ("✓ EMA Cross, MACD, ADX Filter","✗ Mean Reversion, RSI Bounce"),
        "SIDEWAY":        ("✓ RSI Bounce, BB Range, Stochastic","✗ Golden Cross, Breakout Entry"),
        "SQUEEZE":        ("✓ รอ Breakout + Volume","✗ เทรดในทิศทางเดิม"),
        "VOLATILE TREND": ("✓ ATR Stop กว้าง, Position เล็ก","✗ Tight Stop, Over-leverage"),
    }.items():
        active = r == regime[0]
        color  = regime[1] if active else "#1e3045"
        st.markdown(f"""<div style='background:{""+regime[1]+"18" if active else "#0d1828"};border:1px solid {color};border-radius:8px;padding:10px 14px;margin-bottom:6px'>
            <div style='font-size:12px;font-weight:700;color:{regime[1] if active else "#4a6a8a"};margin-bottom:4px'>{"▶ " if active else ""}{r}</div>
            <div style='font-size:11px;color:#00e676'>{use}</div>
            <div style='font-size:11px;color:#ff5252;margin-top:2px'>{avoid}</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 3: DIVERGENCE ─────────────────────────
with tabs[3]:
    st.markdown("<div class='sec-label'>▸ DIVERGENCE DETECTION — สัญญาณกลับตัว Price vs Indicator</div>", unsafe_allow_html=True)
    if not divs:
        st.info("ไม่พบ Divergence ใน 30 วันล่าสุด — ราคาและ Indicators เคลื่อนที่สอดคล้องกัน")
    for d in divs:
        st.markdown(f"""<div style='background:{d["color"]}14;border:1px solid {d["color"]}50;border-radius:10px;padding:12px 16px;margin-bottom:8px'>
            <div style='display:flex;justify-content:space-between'>
                <span style='font-size:13px;font-weight:700;color:{d["color"]}'>{d["type"]} Divergence — {d["ind"]}</span>
                <span style='font-size:12px;color:{d["color"]};font-weight:700'>{d["action"]}</span>
            </div>
            <div style='font-size:11px;color:#c8d8e8;margin-top:6px'>{d["desc"]}</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 4: BREAKOUT ───────────────────────────
with tabs[4]:
    st.markdown("<div class='sec-label'>▸ BREAKOUT + VOLUME ALERT</div>", unsafe_allow_html=True)
    lr = df.iloc[-1]
    vr = lr["Vol_ratio"] if pd.notna(lr["Vol_ratio"]) else 1
    st.metric("Volume Ratio vs SMA20", f"{vr:.2f}×",
              delta="⚠ สูงผิดปกติ" if vr > 1.5 else "ปกติ", delta_color="inverse" if vr < 0.8 else "normal")
    for b in breakouts:
        ac_color = {"BUY":"#00e676","SELL":"#ff5252"}.get(b["action"],"#ffd600")
        st.markdown(f"""<div style='background:{b["color"]}14;border:1px solid {b["color"]}50;border-radius:10px;padding:12px 16px;margin-bottom:8px'>
            <div style='display:flex;justify-content:space-between'>
                <span style='font-size:13px;font-weight:700;color:{b["color"]}'>{b["type"]}</span>
                <span style='font-size:12px;color:{ac_color};font-weight:700'>{b["action"]}</span>
            </div>
            <div style='font-size:11px;color:#c8d8e8;margin-top:4px'>{b["desc"]}</div>
        </div>""", unsafe_allow_html=True)
    if len(df) >= 22:
        prev20 = df.iloc[-21:-1]
        b1, b2, b3 = st.columns(3)
        b1.metric("Resistance 20d", f"{prev20['High'].max():.2f}")
        b2.metric("ราคาล่าสุด",     f"{result['close']}")
        b3.metric("Support 20d",    f"{prev20['Low'].min():.2f}")

# ── TAB 5: FIBONACCI ──────────────────────────
with tabs[5]:
    st.markdown("<div class='sec-label'>▸ FIBONACCI RETRACEMENT — 60 วันล่าสุด</div>", unsafe_allow_html=True)
    for label, val, color, desc in [
        ("0% (High)",    result["fib_high"], "#ff5252", "แนวต้านสูงสุด"),
        ("38.2%",        result["fib_382"],  "#ffd600", "แนวรองรับสำคัญ"),
        ("50%",          result["fib_500"],  "#00d4ff", "จุดกึ่งกลาง"),
        ("61.8% Golden", result["fib_618"],  "#00e676", "Golden Ratio — แข็งแกร่ง"),
        ("100% (Low)",   result["fib_low"],  "#ea80fc", "จุดต่ำสุด"),
    ]:
        diff_ = result["fib_high"] - result["fib_low"]
        near  = diff_ > 0 and abs(val - result["close"]) < diff_ * 0.03
        bg    = f'background:{color}18;border:1px solid {color}60' if near else 'background:#0d1828;border:1px solid #1e3045'
        st.markdown(f"""<div style='{bg};border-radius:7px;padding:9px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center'>
            <div><span style='font-size:12px;font-weight:700;color:{color}'>{label}</span>
            <span style='font-size:10px;color:#4a6a8a;margin-left:10px'>{desc}</span>
            {"<span style='font-size:9px;color:"+color+";margin-left:8px'>◀ ราคาใกล้นี้</span>" if near else ""}</div>
            <span style='font-size:14px;font-weight:700;color:{color};font-family:monospace'>{val}</span>
        </div>""", unsafe_allow_html=True)

# ── TAB 6: PATTERNS ───────────────────────────
with tabs[6]:
    st.markdown("<div class='sec-label'>▸ CANDLESTICK PATTERNS — 5 วันล่าสุด</div>", unsafe_allow_html=True)
    if not patterns:
        st.info("ไม่พบ pattern ชัดเจนในช่วงนี้")
    for p in patterns:
        st.markdown(f"""<div style='background:{p["color"]}14;border:1px solid {p["color"]}50;border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between'>
            <span><span style='font-size:13px;font-weight:700;color:{p["color"]}'>{p["name"]}</span>
            <span style='font-size:11px;color:#7a9ab0;margin-left:10px'>{p["signal"]}</span></span>
            <span style='font-size:11px;color:#4a6a8a'>{p["date"]}</span>
        </div>""", unsafe_allow_html=True)

# ── TAB 7: BACKTEST ───────────────────────────
with tabs[7]:
    st.markdown("<div class='sec-label'>▸ BACKTEST — Entry: RSI<40 + MACD Cross Up + Price>EMA50 | Exit: RSI>65 or MACD Cross Down | SL: 5%</div>", unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("เทรดทั้งหมด",  f"{len(bt['trades'])} ไม้")
    m2.metric("Win Rate",     f"{bt['win_rate']}%")
    m3.metric("Avg Win",      f"+{bt['avg_win']}%")
    m4.metric("Avg Loss",     f"{bt['avg_loss']}%")
    m5.metric("R/R",          f"1:{bt['rr']}")
    m6.metric("Total Return", f"{bt['total_return']:+.1f}%")
    if bt["trades"]:
        cum, eq_y = 0, []
        for t in bt["trades"]:
            cum += t["pnl"]; eq_y.append(round(cum,2))
        fig_eq = go.Figure(go.Scatter(x=list(range(len(eq_y))), y=eq_y, fill="tozeroy",
                                       fillcolor="rgba(0,212,255,0.08)", line=dict(color="#00d4ff",width=2)))
        fig_eq.add_hline(y=0, line_color="#1e3045")
        fig_eq.update_layout(height=140, paper_bgcolor="#080c14", plot_bgcolor="#0a1020",
                              margin=dict(l=50,r=10,t=4,b=4), showlegend=False,
                              font=dict(color="#7a9ab0",size=9),
                              xaxis=dict(gridcolor="#1a2535"), yaxis=dict(gridcolor="#1a2535"))
        st.markdown("<div class='sec-label'>▸ EQUITY CURVE (%)</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_eq, use_container_width=True)
        df_t = pd.DataFrame([{
            "วันเข้า":t["entry_date"],"วันออก":t["exit_date"],
            "ซื้อ":t["entry"],"ขาย":t["exit"],"P&L%":t["pnl"],
            "Win":"✅" if t["win"] else "❌","SL":"🛑" if t["sl"] else "","Bars":t["bars"]
        } for t in bt["trades"]])
        st.dataframe(df_t, use_container_width=True, height=250)

# ── TAB 8: POSITION SIZING ────────────────────
with tabs[8]:
    st.markdown("<div class='sec-label'>▸ POSITION SIZING CALCULATOR</div>", unsafe_allow_html=True)
    pi, pr = st.columns(2)
    with pi:
        port   = st.number_input("มูลค่าพอร์ต (฿)", min_value=10000, value=100000, step=10000)
        risk_p = st.number_input("ความเสี่ยงต่อไม้ (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.5)
        buy_px = st.number_input("ราคาซื้อ", min_value=0.01, value=float(result["close"]), step=0.5)
        sl_px  = st.number_input("Stop Loss", min_value=0.01, value=float(result["stop_loss"]), step=0.5)
    with pr:
        risk   = port * risk_p / 100
        sl_d   = max(0.01, buy_px - sl_px)
        shares = int(risk / sl_d)
        cost   = round(shares * buy_px, 2)
        ml     = round(shares * sl_d, 2)
        t1     = round(buy_px + sl_d * 1.5, 2)
        t2     = round(buy_px + sl_d * 3.0, 2)
        rr_c   = round((t1 - buy_px) / sl_d, 2)
        st.metric("จำนวนหุ้น",         f"{shares:,} หุ้น")
        st.metric("เงินที่ใช้",         f"฿{cost:,.0f}", delta=f"{cost/port*100:.1f}% ของพอร์ต")
        st.metric("ยอมขาดทุนสูงสุด",    f"฿{ml:,.0f}", delta=f"-{risk_p}%", delta_color="inverse")
        pp1, pp2 = st.columns(2)
        pp1.metric("Target 1 (1.5R)", f"{t1}", delta=f"+฿{round(shares*(t1-buy_px)):,}")
        pp2.metric("Target 2 (3R)",   f"{t2}", delta=f"+฿{round(shares*(t2-buy_px)):,}")
        st.metric("R/R Ratio", f"1:{rr_c}", delta="ดี" if rr_c >= 1.5 else "ระวัง")
        st.markdown(f"""<div style='background:#0d1828;border:1px solid #1e3045;border-radius:8px;padding:10px 14px;font-size:11px;font-family:monospace;color:#7a9ab0;margin-top:6px'>
            Risk = {port:,} × {risk_p}% = <span style='color:#ffd600'>฿{risk:,.0f}</span><br>
            Shares = {risk:,.0f} ÷ {sl_d:.2f} = <span style='color:#00d4ff'>{shares:,} หุ้น</span>
        </div>""", unsafe_allow_html=True)

# ── TAB 9: SCANNER ────────────────────────────
with tabs[9]:
    st.markdown("<div class='sec-label'>▸ WATCHLIST SCANNER — สแกนทุกหุ้นพร้อมกัน (ใช้เวลา ~30 วินาที)</div>", unsafe_allow_html=True)
    if st.button("🔍 Scan ทั้งหมด", type="primary"):
        rows, prog = [], st.progress(0)
        for idx, (k, v) in enumerate(WATCHLIST.items()):
            prog.progress((idx+1)/len(WATCHLIST), text=f"Scanning {v}...")
            d = fetch_stock(v, "6mo")
            if d.empty: continue
            d = calc_indicators(d)
            r = gen_signals(d, v)
            rg = get_regime(d)
            dv = detect_divergence(d)
            bk = detect_breakout(d)
            rows.append({
                "Ticker":k, "ราคา":r["close"], "สัญญาณ":r["overall"],
                "Score":r["score"], "RSI":r["rsi"], "ADX":r["adx"],
                "Regime":rg[0], "Diverge":"⚡" if dv else "—",
                "Breakout":"🔥" if any(b["confirmed"] for b in bk) else "—",
            })
        prog.empty()
        if rows:
            df_sc = pd.DataFrame(rows).sort_values("Score", ascending=False)
            def color_score(val):
                if val >= 6:   return "background-color:#00e67633;color:#00e676"
                elif val >= 3: return "background-color:#00e67618;color:#00e676"
                elif val >= -2:return "background-color:#ffd60018;color:#ffd600"
                elif val >= -5:return "background-color:#ff525218;color:#ff5252"
                else:          return "background-color:#ff525233;color:#ff5252"
            st.dataframe(df_sc.style.applymap(color_score, subset=["Score"]),
                         use_container_width=True, height=420)
        else:
            st.error("ไม่สามารถดึงข้อมูลได้ — ตรวจสอบการเชื่อมต่อ")
    else:
        st.info("กด 'Scan ทั้งหมด' เพื่อวิเคราะห์ทุกหุ้นใน Watchlist")

# ── TAB 10: AI ────────────────────────────────
with tabs[10]:
    st.markdown("<div class='sec-label'>▸ CLAUDE AI — วิเคราะห์ครบทุก Indicator ภาษาไทย</div>", unsafe_allow_html=True)
    with st.expander("📋 ดูข้อมูลที่จะส่งให้ Claude"):
        for s in result["signals"]:
            ic = {"BUY":"🟢","SELL":"🔴","STRONG BUY":"🟢","STRONG SELL":"🔴"}.get(s["a"],"🟡")
            st.markdown(f"{ic} **{s['i']}** `{s['v']}` → {s['a']} ({s['s']})")
        if patterns: st.markdown("**Patterns:** " + ", ".join(p["name"] for p in patterns))
        if divs:     st.markdown("**Divergence:** " + ", ".join(f"{d['type']} {d['ind']}" for d in divs))
        st.markdown(f"**Regime:** {regime[0]} — {regime[2]}")
    if not api_key:
        st.warning("ใส่ Anthropic API Key ใน Sidebar", icon="🔑")
    elif st.button("⬡ ให้ Claude วิเคราะห์", type="primary"):
        with st.spinner("Claude กำลังวิเคราะห์ครบทุก indicator..."):
            try:
                summary = ai_summary(result, patterns, regime, divs, breakouts, api_key)
                st.markdown(f"""<div class='ai-box'>
                    <div style='font-size:9px;color:#ea80fc;letter-spacing:.15em;margin-bottom:12px'>⬡ CLAUDE AI · {result['ticker']} · {result['date']}</div>
                    <div style='font-size:13px;line-height:1.9;color:#c8d8e8'>{summary.replace(chr(10),'<br>')}</div>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ── FOOTER ────────────────────────────────────
st.markdown("---")
st.markdown(f"""<div style='text-align:center;font-size:9px;color:#2a4a6a;padding:6px'>
    TECHSCAN SET v2.0 · ข้อมูลจาก Yahoo Finance (delay ~15 min) · อัปเดต {datetime.now().strftime("%d/%m/%Y %H:%M")} · ไม่ใช่คำแนะนำการลงทุน
</div>""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(300)
    st.rerun()
