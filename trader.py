"""
trader.py — Paper-trade botu (tek seferlik çalışır, cron tetikler).

Akış:
  1) config.json oku    -> hangi ticker, interval, lookback, k
  2) veri çek (yfinance) -> son KAPANMIŞ bar baz alınır
  3) signals.py          -> kombine skor -> z-score -> AL/SAT/TUT
  4) position.json oku   -> elde pozisyon var mı?
  5) karara göre paper al/sat, position.json güncelle
  6) işlem olduysa Telegram'a mesaj at

Gerçek para YOK. Tamamen simülasyon (paper trading).
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import signals

# ── Telegram (Aşama 4'te aktif olacak). Ortam değişkeninden okunur. ──
try:
    from notify import send_telegram
except Exception:
    def send_telegram(msg):
        print("[TELEGRAM stub]", msg)


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
POS_PATH     = os.path.join(BASE_DIR, "position.json")

DEFAULT_CONFIG = {
    "ticker": "gc=f",
    "interval": "15m",
    "period": "5d",
    "lookback": 100,
    "k": 1.0,
    "capital": 1000.0,
    "cost_pct": 0.001,
}

INTRADAY_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "1h", "4h", "8h")


def load_json(path, default):
    if not os.path.exists(path):
        return dict(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(default)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch(ticker, period, interval):
    fetch_i = "1h" if interval in ("4h", "8h") else interval
    df = yf.download(ticker, period=period, interval=fetch_i, progress=False)
    df = signals.flatten_columns(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def fresh_position(ticker, capital):
    return {
        "ticker": ticker,
        "in_position": False,
        "entry_price": None,
        "entry_time": None,
        "qty": 0.0,
        "cash": capital,
        "capital": capital,
        "last_bar_time": None,   # işlenmiş son bar (çift işlem engeli)
        "history": [],           # [{action, price, time, pct, value}]
    }


def run():
    cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)
    ticker   = cfg.get("ticker", DEFAULT_CONFIG["ticker"])
    interval = cfg.get("interval", DEFAULT_CONFIG["interval"])
    period   = cfg.get("period", DEFAULT_CONFIG["period"])
    lookback = int(cfg.get("lookback", 100))
    k        = float(cfg.get("k", 1.0))
    capital  = float(cfg.get("capital", 1000.0))
    cost_pct = float(cfg.get("cost_pct", 0.001))
    is_intra = interval in INTRADAY_INTERVALS

    pos = load_json(POS_PATH, fresh_position(ticker, capital))
    # Ticker değiştiyse pozisyonu sıfırla (eldeki başka enstrümanı taşımayız).
    if pos.get("ticker") != ticker:
        if pos.get("in_position"):
            print(f"[UYARI] Ticker {pos.get('ticker')} -> {ticker} değişti, açık pozisyon vardı; sıfırlanıyor.")
        pos = fresh_position(ticker, capital)

    df = fetch(ticker, period, interval)
    if df is None or df.empty or len(df) < lookback + 5:
        print(f"[BİLGİ] Yetersiz veri ({0 if df is None else len(df)} bar). Çıkılıyor.")
        save_json(POS_PATH, pos)
        return

    enr   = signals.compute_indicators(df, is_intraday=is_intra)
    score = signals.compute_score(enr)
    dec_series, z_series = signals.zscore_signal(score, lookback=lookback, k=k)

    # Son KAPANMIŞ bar = sondan ikinci (canlı bar henüz kapanmadı).
    bar_time = str(enr.index[-2])
    decision = dec_series.iloc[-2]
    price    = float(enr["Close"].iloc[-2])
    z_val    = float(z_series.iloc[-2])

    # Bu bar daha önce işlendiyse tekrar işlem yapma.
    if pos.get("last_bar_time") == bar_time:
        print(f"[BİLGİ] {bar_time} zaten işlendi. Karar={decision}. Atlanıyor.")
        save_json(POS_PATH, pos)
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    acted = False

    if not pos["in_position"] and decision == "AL":
        qty = pos["cash"] / (price * (1 + cost_pct))
        pos["in_position"] = True
        pos["entry_price"] = price
        pos["entry_time"]  = bar_time
        pos["qty"]  = qty
        pos["cash"] = 0.0
        pos["history"].append({"action": "AL", "price": price, "time": bar_time, "value": pos["capital"]})
        acted = True
        send_telegram(
            f"🟢 ALDIM\n{ticker} @ {price:.4f}\n"
            f"Miktar: {qty:.4f}\nBar: {bar_time}\nz={z_val:+.2f}\n{now}")

    elif pos["in_position"] and decision == "SAT":
        gross = pos["qty"] * price * (1 - cost_pct)
        entry_val = pos["capital"]
        pct = (gross - entry_val) / entry_val * 100.0
        pos["in_position"] = False
        pos["cash"] = gross
        pos["capital"] = gross          # yeni sermaye = satış sonrası nakit
        sold_qty = pos["qty"]
        pos["qty"] = 0.0
        pos["entry_price"] = None
        pos["entry_time"]  = None
        pos["history"].append({"action": "SAT", "price": price, "time": bar_time,
                               "pct": round(pct, 2), "value": round(gross, 2)})
        acted = True
        emoji = "📈" if pct >= 0 else "📉"
        send_telegram(
            f"🔴 SATTIM {emoji}\n{ticker} @ {price:.4f}\n"
            f"Sonuç: %{pct:+.2f}\nYeni bakiye: {gross:.2f} TL\n"
            f"Bar: {bar_time}\nz={z_val:+.2f}\n{now}")

    pos["last_bar_time"] = bar_time
    save_json(POS_PATH, pos)

    state = "ELDE" if pos["in_position"] else "NAKİT"
    print(f"[{now}] {ticker} | karar={decision} z={z_val:+.2f} fiyat={price:.4f} "
          f"| durum={state} | işlem={'EVET' if acted else 'hayır'}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[HATA] {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot hatası: {e}")
        sys.exit(1)
