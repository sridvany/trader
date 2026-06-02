"""
trader.py — Çoklu-ticker paper-trade botu (tek seferlik çalışır, cron tetikler).

Akış:
  1) config.json oku    -> tickers listesi + interval, lookback, k
  2) her ticker için:
       veri çek -> kombine skor -> z-score -> AL/SAT/TUT
       o ticker'ın pozisyonuna göre paper al/sat
       işlem olduysa Telegram'a mesaj at
  3) tüm pozisyonları position.json'a yaz (ticker bazlı sözlük)

Her ticker BAĞIMSIZ kendi sermayesiyle işlem yapar.
Gerçek para YOK. Tamamen simülasyon (paper trading).
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import signals

try:
    from notify import send_telegram
except Exception:
    def send_telegram(msg):
        print("[TELEGRAM stub]", msg)


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
POS_PATH    = os.path.join(BASE_DIR, "position.json")

DEFAULT_CONFIG = {
    "tickers": ["gc=f", "NVDA", "ASELS.IS"],
    "interval": "15m",
    "period": "5d",
    "lookback": 32,
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


def get_tickers(cfg):
    """config'ten ticker listesini çıkarır. Eski tekil 'ticker' alanını da destekler."""
    if "tickers" in cfg and isinstance(cfg["tickers"], list) and cfg["tickers"]:
        return [str(t) for t in cfg["tickers"]]
    if "ticker" in cfg:
        return [str(cfg["ticker"])]
    return list(DEFAULT_CONFIG["tickers"])


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
        "last_bar_time": None,
        "history": [],
    }


def process_ticker(ticker, cfg, all_pos):
    """Tek bir ticker'ı işler, all_pos sözlüğünü günceller. Hata olursa yutar (diğerleri devam etsin)."""
    interval = cfg.get("interval", "15m")
    period   = cfg.get("period", "5d")
    lookback = int(cfg.get("lookback", 32))
    k        = float(cfg.get("k", 1.0))
    capital  = float(cfg.get("capital", 1000.0))
    cost_pct = float(cfg.get("cost_pct", 0.001))
    is_intra = interval in INTRADAY_INTERVALS

    pos = all_pos.get(ticker)
    if not pos or pos.get("ticker") != ticker:
        pos = fresh_position(ticker, capital)

    try:
        df = fetch(ticker, period, interval)
    except Exception as e:
        print(f"[{ticker}] veri hatası: {e} — atlanıyor.")
        all_pos[ticker] = pos
        return

    if df is None or df.empty or len(df) < lookback + 5:
        n = 0 if df is None else len(df)
        print(f"[{ticker}] yetersiz veri ({n} bar) — atlanıyor.")
        all_pos[ticker] = pos
        return

    enr   = signals.compute_indicators(df, is_intraday=is_intra)
    score = signals.compute_score(enr)
    dec_series, z_series = signals.zscore_signal(score, lookback=lookback, k=k)

    bar_time = str(enr.index[-2])      # son KAPANMIŞ bar
    decision = dec_series.iloc[-2]
    price    = float(enr["Close"].iloc[-2])
    z_val    = float(z_series.iloc[-2])

    if pos.get("last_bar_time") == bar_time:
        print(f"[{ticker}] {bar_time} zaten işlendi. Karar={decision}. Atlanıyor.")
        all_pos[ticker] = pos
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
        pos["capital"] = gross
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
    all_pos[ticker] = pos

    state = "ELDE" if pos["in_position"] else "NAKİT"
    print(f"[{now}] {ticker} | karar={decision} z={z_val:+.2f} fiyat={price:.4f} "
          f"| durum={state} | işlem={'EVET' if acted else 'hayır'}")


def run():
    cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)
    tickers = get_tickers(cfg)
    capital = float(cfg.get("capital", 1000.0))

    all_pos = load_json(POS_PATH, {})
    # Eski tek-ticker formatı (düz pozisyon sözlüğü) ise sözlük yapısına çevir.
    if isinstance(all_pos, dict) and "in_position" in all_pos:
        old_ticker = all_pos.get("ticker", "gc=f")
        all_pos = {old_ticker: all_pos}
    if not isinstance(all_pos, dict):
        all_pos = {}

    # config'te artık olmayan ticker'ları pozisyondan düşür (temizlik).
    for t in list(all_pos.keys()):
        if t not in tickers:
            print(f"[TEMİZLİK] {t} artık izlenmiyor, pozisyondan çıkarılıyor.")
            del all_pos[t]

    for t in tickers:
        process_ticker(t, cfg, all_pos)

    save_json(POS_PATH, all_pos)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[HATA] {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot hatası: {e}")
        sys.exit(1)
