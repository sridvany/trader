import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from itertools import product as iter_product
import requests
import json
import hashlib
import time
from datetime import datetime, timedelta, timezone
import signals  # çekirdek sinyal/skor motoru (panel + bot ortak)

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="tahmin.ai", layout="wide")

auto_refresh_on = st.sidebar.toggle("🔄 Canlı Yenileme", value=True)
if auto_refresh_on:
    st_autorefresh(interval=55 * 1000, key="terminal_refresh")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    div[data-testid="stCaption"] { margin-top: -0.5rem; margin-bottom: -0.5rem; }
    h1 { margin-bottom: 0 !important; padding-bottom: 0 !important; }

    /* Plotly legend scrollbar — ince ve diskret */
    .js-plotly-plot .scrollbox::-webkit-scrollbar,
    .js-plotly-plot .legend ::-webkit-scrollbar {
        width: 4px !important;
        height: 4px !important;
    }
    .js-plotly-plot .scrollbox::-webkit-scrollbar-thumb,
    .js-plotly-plot .legend ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 2px !important;
    }
    .js-plotly-plot .scrollbox::-webkit-scrollbar-track,
    .js-plotly-plot .legend ::-webkit-scrollbar-track {
        background: transparent !important;
    }
    /* Firefox için */
    .js-plotly-plot .scrollbox,
    .js-plotly-plot .legend {
        scrollbar-width: thin !important;
        scrollbar-color: rgba(255,255,255,0.2) transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 PİYASA TERMİNALİ")
st.caption("YATIRIM TAVSİYESİ İÇERMEZ. ARAŞTIRMA İÇİNDİR.")

# ============================================================
# SESSION STATE VARSAYILANLARI
# ============================================================
_defaults = {
    "sma_short":     20,
    "sma_long":      200,
    "rsi_period":    14,
    "rsi_lower":     30,
    "rsi_upper":     70,
    "rsi_trend_period": 200,
    "bb_period":     20,
    "bb_std":        2.0,
    "macd_fast":     12,
    "macd_slow":     26,
    "macd_signal":   9,
    "adx_period":    14,
    "adx_threshold": 25,
    "st_period":     10,
    "st_multiplier": 3.0,
    "lrc_period":    50,
    "lrc_std_mult":  2.0,
    "wt_n1":         10,
    "wt_n2":         21,
    "obv_short":     10,
    "obv_long":      30,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 🤖 LLM PROVIDER KONFİGÜRASYONU (Google Gemini)
# ============================================================
# API key almak için: https://aistudio.google.com/app/apikey
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

AI_DETAIL_LEVELS = {"Kısa": 1500, "Orta": 4000, "Detaylı": 8000}


def _parse_http_error(response, default_msg):
    """HTTP hata gövdesinden anlamlı mesaj çıkar."""
    try:
        body = response.text
    except Exception:
        return default_msg
    msg = body[:500]
    try:
        err_json = json.loads(body)
        if isinstance(err_json, dict) and "error" in err_json:
            err_detail = err_json["error"]
            if isinstance(err_detail, dict):
                msg = err_detail.get("message", msg)
            else:
                msg = str(err_detail)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return msg


def fetch_llm(api_key, system_prompt, user_prompt, max_tokens):
    """Google Gemini — (text, meta) döner."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     0.4,
            "topP":            0.95,
        },
    }
    r = requests.post(
        f"{GEMINI_ENDPOINT}?key={api_key}",
        headers=headers, json=payload, timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_parse_http_error(r, r.text[:500])}")

    data = r.json()
    candidates = data.get("candidates", []) or []
    text   = ""
    finish = None
    if candidates:
        cand   = candidates[0] or {}
        finish = cand.get("finishReason")
        parts  = (cand.get("content") or {}).get("parts", []) or []
        text   = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    usage = data.get("usageMetadata", {}) or {}
    meta = {
        "finish_reason":   finish,
        "prompt_tokens":   usage.get("promptTokenCount",     0),
        "output_tokens":   usage.get("candidatesTokenCount", 0),
        "thinking_tokens": usage.get("thoughtsTokenCount",   0),
        "total_tokens":    usage.get("totalTokenCount",      0),
    }
    return text, meta


def build_ai_prompt(*, detail, ticker, close, interval,
                    res_rows, swing_levels, fib_levels):
    """Yapılandırılmış system + user prompt üret.
    Sadece Algoritmik Detaylar tablosu + Seviye bilgileri kullanılır.
    """
    system = (
        "Sen deneyimli bir kurumsal teknik analiz uzmanısın. "
        "SADECE sana verilen 'Algoritmik Detaylar' tablosundaki bilgileri kullan. "
        "Hiçbir sayıyı uydurma, tahmin etme veya ek veri varsay. "
        "Tablodaki her indikatörün 'Durum/Sebep' sütununu dikkatle oku — "
        "içinde zengin bilgi var (değerler, ilişkiler, yönler, uyarılar).\n\n"
        "DİL VE ÜSLUP:\n"
        "- Türkçe yanıt ver, teknik jargon kullanabilirsin ama netlikten taviz verme\n"
        "- Markdown formatında, başlıklar altında organize et\n"
        "- Somut ve aksiyona dönüştürülebilir ol\n"
        "- 'Yatırım tavsiyesi' ibaresi kullanma\n\n"
        "KISA VADELİ BEKLENTİ KURALLARI:\n"
        "- Gelecek yönünü KEHANET olarak değil, 'göstergelerin ima ettiği eğilim' olarak sun\n"
        "- 'Muhtemelen', 'eğilim gösteriyor', 'olasılıkla' gibi ihtimal dili kullan\n"
        "- Güven seviyesini göstergelerin UYUMUNA göre belirle:\n"
        "  • 15+ gösterge aynı yönde → Yüksek güven\n"
        "  • 10-14 gösterge aynı yönde → Orta güven\n"
        "  • Dağınık / çelişkili → Düşük güven\n"
        "- Yön için somut TETİKLEYİCİ seviyeler ver (hangi fiyat kırılırsa ne olur)\n\n"
        "RİSK/ÖDÜL KURALI:\n"
        "- Stop-loss ve hedef verdikten sonra R/R = (hedef-giriş)/(giriş-stop) hesapla\n"
        "- R/R < 2:1 ise 'R/R uygun değil, pozisyonu yeniden değerlendirin' şeklinde AÇIKÇA uyar\n\n"
        "FORMATLAMA KURALLARI:\n"
        "- Her cümleyi tam bitir, asla yarıda bırakma\n"
        "- Her başlığı mutlaka tamamla\n"
        "- Son cümle noktayla bitmeli"
    )

    # Destek / Direnç seviyeleri
    sr_lines = []
    if swing_levels:
        below = sorted([s for s in swing_levels if s["price"] < close], key=lambda x: -x["price"])
        above = sorted([s for s in swing_levels if s["price"] > close], key=lambda x: x["price"])
        for i, b in enumerate(below[:2]):
            pct = abs(b["price"] - close) / close * 100
            sr_lines.append(f"Destek-{i+1}: {b['price']:.2f} (%{pct:.2f} altta, {b['touches']}x test)")
        for i, a in enumerate(above[:2]):
            pct = abs(a["price"] - close) / close * 100
            sr_lines.append(f"Direnç-{i+1}: {a['price']:.2f} (%{pct:.2f} üstte, {a['touches']}x test)")
    sr_text = ("\n  - " + "\n  - ".join(sr_lines)) if sr_lines else " Tespit edilmedi"

    # Fibonacci seviyeleri (en yakın 3)
    fib_text = "Hesaplanmadı"
    if fib_levels:
        sorted_fib = sorted(fib_levels.items(), key=lambda x: abs(x[1] - close))[:5]
        fib_text = ", ".join(f"{k} = {v:.2f}" for k, v in sorted_fib)

    # Algoritmik Detaylar tablosu (res_rows = [[karar, algoritma, durum], ...])
    detay_lines = []
    for row in res_rows:
        if len(row) >= 3:
            karar_c, algo_c, durum_c = row[0], row[1], row[2]
            detay_lines.append(f"| {karar_c} | **{algo_c}** | {durum_c} |")
    detay_table = (
        "| Karar | Algoritma | Durum / Sebep |\n"
        "|---|---|---|\n"
        + "\n".join(detay_lines)
    )

    # Çıktı şablonu
    if detail == "Kısa":
        output_req = (
            "\n## İstenen Çıktı (KISA)\n"
            "Şu başlıklarda kısa yorum yap:\n"
            "1. **🎯 Durum** — genel resim (2-3 cümle)\n"
            "2. **⚠️ Uyarı** — en kritik risk\n"
            "3. **📍 Aksiyon** — ne yapmalı (R/R ile)\n"
            "4. **🔮 Kısa Vadeli Beklenti** — muhtemel yön + tetikleyici seviye\n"
        )
    elif detail == "Orta":
        output_req = (
            "\n## İstenen Çıktı (ORTA)\n"
            "Şu başlıklarda orta uzunlukta yorum yap:\n"
            "1. **🎯 Genel Değerlendirme** — tablonun verdiği resim (3-4 cümle)\n"
            "2. **📊 Öne Çıkan Göstergeler** — tabloda en önemli 4-5 satır yorumu\n"
            "3. **⚠️ Ana Risk** — en kritik uyarı\n"
            "4. **📍 Giriş Senaryosu** — hangi seviyeler aksiyon için uygun\n"
            "5. **🛡️ Risk Yönetimi** — stop, hedef, R/R hesabı\n"
            "6. **🔮 Kısa Vadeli Beklenti** — muhtemel yön (olasılıklı) + tetikleyici seviyeler\n"
            "7. **👁️ Takip Listesi** — 3-4 kritik sinyal\n"
        )
    else:  # Detaylı
        output_req = (
            "\n## İstenen Çıktı (DETAYLI)\n"
            "Aşağıdaki başlıklarda GENİŞ ve DERİNLEMESİNE yorum yap. "
            "Tablodaki HER indikatörü kategorisine göre grupla ve yorumla — "
            "sadece değeri söyleme, ne anlama geldiğini açıkla.\n\n"
            "1. **🎯 Genel Değerlendirme** — tablonun verdiği bütüncül resim (3-4 cümle)\n\n"
            "2. **📊 İndikatör Bazlı Detaylı Analiz**\n"
            "   Alt başlıklar altında her indikatörü yorumla:\n\n"
            "   **🔹 Trend Göstergeleri** — SMA, EMA200, KAMA, SuperTrend, Ichimoku\n"
            "   (KAMA için ER değerini, SuperTrend için flip yakınlığı ve bar sayısını, "
            "   Ichimoku için bulut pozisyonu + rejim uyarısını dikkate al)\n\n"
            "   **🔹 Momentum Göstergeleri** — RSI, Stoch RSI, MACD, WaveTrend\n"
            "   (Stoch RSI için K/D ilişkisi ve teyit durumunu, MACD için histogram "
            "   rengi/yönü/zero line'ı, WaveTrend için histogram rengini dikkate al)\n\n"
            "   **🔹 Volatilite ve Kanallar** — Bollinger, ATR Filtre, LR Channel\n"
            "   (ATR için son 5 bar yönünü, LRC için slope yönü + bant genişliğini dikkate al)\n\n"
            "   **🔹 Hacim ve Seviye** — OBV, Swing S/R, Fibonacci, VWAP\n"
            "   (OBV için SMA farkı ve fark büyüklüğünü dikkate al)\n\n"
            "   **🔹 Uyarı Sinyalleri** — ADX (+DI/-DI), Divergence (RSI + MACD + OBV)\n\n"
            "3. **⚠️ Ana Risk Faktörleri** — en kritik 2-3 uyarı ve nedenleri\n\n"
            "4. **📍 Aksiyon Planı**\n"
            "   - Önerilen giriş seviyesi\n"
            "   - Stop-loss + hedef (somut sayılarla)\n"
            "   - Risk/Ödül hesabı: R/R = (hedef - giriş) / (giriş - stop)\n"
            "   - R/R < 2:1 ise AÇIKÇA uyar\n\n"
            "5. **🔮 Kısa Vadeli Beklenti (1-5 bar)**\n"
            "   - **Muhtemel Senaryo:** 🟢 Yükseliş eğilimi / 🔴 Düşüş eğilimi / ⚪ Yatay\n"
            "   - **Güven Seviyesi:** Düşük / Orta / Yüksek (gösterge uyumuna göre)\n"
            "   - **Olasılık tahmini:** yükseliş ~X% / düşüş ~Y% / yatay ~Z%\n"
            "   - **Gerekçe:** hangi göstergeler ne diyor\n"
            "   - **Tetikleyici seviyeler:**\n"
            "     ✅ Yükselişi teyit edecek: [somut seviye]\n"
            "     ❌ Düşüşe çevirecek: [somut seviye]\n\n"
            "6. **👁️ Takip Listesi** — durumu değiştirebilecek 5-6 kritik sinyal\n"
        )

    user = f"""## Analiz Edilecek Veri

**Enstrüman:** {ticker.upper()}
**Fiyat:** {close:.4f}
**Zaman Dilimi:** {interval}

## 📋 Algoritmik Detaylar Tablosu

{detay_table}

## 📍 Seviye Bilgileri

- **Destek / Direnç:**{sr_text}
- **En Yakın Fibonacci Seviyeleri:** {fib_text}

---
{output_req}
### Önemli Kurallar
- Yukarıdaki tabloda verilmeyen hiçbir sayıyı uydurma
- Her indikatörün "Durum/Sebep" sütununu dikkatle oku (zengin bilgi içerir)
- "Yatırım tavsiyesi" ibaresi kullanma
- Markdown formatında yaz (başlıklar, bold, listeler)
- Kısa Vadeli Beklenti'de kehanet dili değil, 'göstergelerin ima ettiği' dili kullan
"""
    return system, user


def ai_cache_key(ticker, interval, total_score, close, detail):
    """Analiz durumu için stabil cache anahtarı."""
    s = f"{ticker}|{interval}|{round(total_score, 2)}|{round(close, 4)}|{detail}"
    return "ai_report_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def clean_half_sentence(text):
    """Yanıt sonunda yarım kalmış cümleyi temizle.
    Gemini Flash streaming bazen cümle ortasında kesiyor — bunu tespit edip
    son tam cümleyi koru. (clean_text, was_cut) döner."""
    if not text or len(text) < 30:
        return text, False

    stripped = text.rstrip()
    if not stripped:
        return text, False

    # Zaten düzgün bir bitirici karakter ile bitiyor mu?
    # (nokta, ünlem, soru, iki nokta, parantez, yıldız, backtick, emoji/özel)
    safe_endings = ".!?:;)]}*`\"'›»"
    if stripped[-1] in safe_endings:
        return text, False

    # Cümle sonlarını bul — SADECE "nokta/ünlem/soru + boşluk veya satır sonu"
    # Bu decimal sayıları (örn. "78.4") yanlışlıkla cümle sonu saymaz.
    import re
    matches = list(re.finditer(r'[.!?](?=\s|$)', stripped))
    if not matches:
        return text, False

    last_end = matches[-1].end()  # son cümle bitişinin konumu

    # Son bitiş çok başlardaysa (metnin %40'ından az), olduğu gibi bırak —
    # muhtemelen kısa bir özetti, müdahale etmeyelim
    if last_end < len(stripped) * 0.4:
        return text, False

    cleaned = stripped[:last_end]
    return cleaned, True


# ============================================================
# 2. YAN PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Veri Ayarları")
    ticker = st.text_input("Ticker Sembolü:","gc=f")

    period = st.selectbox(
        "Toplam Veri Süresi (Period):",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=6,
    )

    if period in ["1d", "5d"]:
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 0
    elif period == "1mo":
        interval_options = ["2m", "5m", "15m", "30m", "60m", "1h", "4h", "1d"]
        default_int_idx = 6
    else:
        interval_options = ["1h", "4h", "8h", "1d", "1wk", "1mo"]
        default_int_idx = 3

    interval = st.selectbox(
        "Mum Aralığı (Interval):", options=interval_options, index=default_int_idx
    )

    # ──────────────────────────────────────────────────────────
    # 📅 PAPER-TRADE BACKTEST KONSOLU
    # Seçilen tarihte 1000 TL'lik alınmış say; z-score sinyalleriyle
    # bugüne kadar simüle et, % kazancı göster.
    # ──────────────────────────────────────────────────────────
    st.write("---")
    with st.expander("📅 Tarih & % Kazanç (Paper-Trade)", expanded=True):
        bt_date = st.date_input(
            "Başlangıç tarihi:",
            value=(datetime.now() - timedelta(days=7)).date(),
            help="Bu tarihte 1000 TL'lik pozisyon açılmış sayılır.",
        )
        bt_lookback = st.slider("Z-Score Penceresi (lookback):", 20, 300, 100, step=10)
        bt_k = st.slider("Eşik Katsayısı (k):", 0.5, 3.0, 1.0, step=0.1,
                         help="Skor ortalamadan kaç std uzaklaşınca sinyal üretilsin.")
        if st.button("💹 Hesapla", use_container_width=True):
            try:
                _is_intra = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "1h", "4h", "8h")
                _fetch_i = "1h" if interval in ("4h", "8h") else interval
                _bt_df = yf.download(ticker, period=period, interval=_fetch_i, progress=False)
                _bt_df = signals.flatten_columns(_bt_df)
                _bt_df = _bt_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                if _bt_df.empty:
                    st.error("Veri çekilemedi.")
                else:
                    _r = signals.backtest_from_date(
                        _bt_df, bt_date, lookback=bt_lookback, k=bt_k,
                        capital=1000.0, is_intraday=_is_intra)
                    if not _r.get("ok"):
                        st.warning(_r.get("msg", "Hesaplanamadı."))
                    else:
                        _pct = _r["pct_return"]
                        st.metric(
                            "💰 % Kazanç",
                            f"%{_pct:+.2f}",
                            delta=f"{_r['final_value']:.2f} TL (1000 TL → )",
                        )
                        _d = _r["last_decision"]
                        _emoji = {"AL": "🟢", "SAT": "🔴", "TUT": "⚪"}.get(_d, "⚪")
                        st.caption(
                            f"İşlem: {_r['n_trades']} · Kazanma: %{_r['win_rate']:.0f} · "
                            f"Max DD: %{_r['max_dd']:.1f}  \n"
                            f"Güncel sinyal: {_emoji} **{_d}** (z={_r['last_z']:+.2f})"
                        )
            except Exception as _e:
                st.error(f"Hata: {_e}")

        st.write("")
        if st.button("🤖 Bota Gönder", use_container_width=True,
                     help="Bu ticker + ayarları config.json'a yazar; bot bunu izler."):
            try:
                # Mevcut config'i oku, tickers listesini koru
                try:
                    with open("config.json", "r", encoding="utf-8") as _f:
                        _cfg = json.load(_f)
                except (FileNotFoundError, json.JSONDecodeError):
                    _cfg = {}

                _tickers = _cfg.get("tickers")
                if not isinstance(_tickers, list):
                    # Eski tekil format → listeye çevir
                    _tickers = [_cfg["ticker"]] if "ticker" in _cfg else []
                if ticker not in _tickers:
                    _tickers.append(ticker)

                _cfg.update({
                    "tickers":  _tickers,
                    "interval": interval,
                    "period":   period,
                    "lookback": int(bt_lookback),
                    "k":        float(bt_k),
                    "capital":  1000.0,
                    "cost_pct": 0.001,
                    # Paneldeki güncel indikatör parametreleri — bot birebir aynı
                    # sinyalleri üretsin diye compute_indicators'a geçirilir.
                    "params": {
                        "sma_short": int(sma_short),   "sma_long": int(sma_long),
                        "rsi_period": int(rsi_period), "rsi_lower": int(rsi_lower),
                        "rsi_upper": int(rsi_upper),   "rsi_ma_period": int(rsi_ma_period),
                        "rsi_trend_period": int(rsi_trend_period),
                        "bb_period": int(bb_period),   "bb_std": float(bb_std),
                        "macd_fast": int(macd_fast),   "macd_slow": int(macd_slow),
                        "macd_signal": int(macd_signal),
                        "obv_short": int(obv_short),   "obv_long": int(obv_long),
                        "adx_period": int(adx_period), "adx_threshold": int(adx_threshold),
                        "stoch_rsi_period": int(stoch_rsi_period),
                        "stoch_d_period": int(stoch_d_period),
                        "stoch_lower": int(stoch_lower), "stoch_upper": int(stoch_upper),
                        "ichi_tenkan": int(ichi_tenkan), "ichi_kijun": int(ichi_kijun),
                        "ichi_senkou_b": int(ichi_senkou_b),
                        "kama_period": int(kama_period), "kama_fast": int(kama_fast),
                        "kama_slow": int(kama_slow),
                        "st_period": int(st_period),   "st_multiplier": float(st_multiplier),
                        "lrc_period": int(lrc_period), "lrc_std_mult": float(lrc_std_mult),
                        "atr_period": int(atr_period),
                        "vwap_band_pct": float(vwap_band_pct),
                        "wt_n1": int(wt_n1), "wt_n2": int(wt_n2),
                        "wt_ob": int(wt_ob), "wt_os": int(wt_os),
                    },
                })
                _cfg.pop("ticker", None)  # eski tekil alanı temizle

                with open("config.json", "w", encoding="utf-8") as _f:
                    json.dump(_cfg, _f, ensure_ascii=False, indent=2)
                st.success(f"✅ Bot izleme listesi: **{', '.join(_tickers)}** ({interval}).")
            except Exception as _e:
                st.error(f"config.json yazılamadı: {_e}")

    st.write("---")
    chart_type = st.radio("📊 Grafik Tipi:", ["Mum", "Çizgi"], horizontal=True)
    show_vp = st.checkbox("Fiyat Hacimlerini Göster", value=True)

    # ──────────────────────────────────────────────────────────
    # 🤖 AI RAPOR YORUMCUSU — Google Gemini
    # API key kullanıcı tarafından konsola girilir
    # ──────────────────────────────────────────────────────────
    st.write("---")
    st.subheader("🤖 AI Rapor Yorumcusu")

    ai_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="api anahtarı buraya...",
        key="gemini_api_key",
        help="API key almak için: https://aistudio.google.com/app/apikey",
    )
    if ai_api_key:
        st.caption(f"Model: **{GEMINI_MODEL}** · Key: ✅ girildi")
    else:
        st.caption(f"Model: **{GEMINI_MODEL}** · Key: ❌ yok")
        st.markdown(
            "🔑 [API key al →](https://aistudio.google.com/app/apikey)",
            unsafe_allow_html=False,
        )

    ai_detail = st.select_slider(
        "Detay Seviyesi",
        options=list(AI_DETAIL_LEVELS.keys()),
        value="Detaylı",
        key="ai_detail_level",
    )
    st.caption(f"Max token: {AI_DETAIL_LEVELS[ai_detail]} · Sıcaklık: 0.4")

    st.write("---")
    st.subheader("Sabit Parametreler")
    ss = st.session_state
    sma_short        = st.slider("SMA Kısa Periyot:",        5,   50,  value=ss["sma_short"])
    sma_long         = st.slider("SMA Uzun Periyot:",        50,  300, value=ss["sma_long"])
    rsi_period       = st.slider("RSI Periyodu:",            7,   21,  value=ss["rsi_period"])
    rsi_lower        = st.slider("RSI Alt Eşik:",            20,  40,  value=ss["rsi_lower"])
    rsi_upper        = st.slider("RSI Üst Eşik:",            60,  80,  value=ss["rsi_upper"])
    rsi_trend_period = st.slider("RSI Trend Filtresi (SMA):", 50, 300, value=ss["rsi_trend_period"], step=10,
        help="AL yalnızca fiyat bu SMA üstündeyse, SAT yalnızca altındaysa geçerli. Falling-knife önlemi.")
    rsi_ma_period    = st.slider("RSI MA Periyodu:",         5,   50,  14)
    bb_period        = st.slider("BB Periyodu:",             10,  50,  value=ss["bb_period"])
    bb_std           = st.slider("BB Standart Sapma:",       1.0, 3.0, value=ss["bb_std"],        step=0.5)
    macd_fast        = st.slider("MACD Hızlı EMA:",          5,   20,  value=ss["macd_fast"])
    macd_slow        = st.slider("MACD Yavaş EMA:",          15,  40,  value=ss["macd_slow"])
    macd_signal      = st.slider("MACD Sinyal:",             5,   15,  value=ss["macd_signal"])
    obv_short        = st.slider("OBV Kısa SMA:",            5,   20,  value=ss["obv_short"])
    obv_long         = st.slider("OBV Uzun SMA:",            15,  50,  value=ss["obv_long"])
    adx_period       = st.slider("ADX Periyodu:",            7,   30,  value=ss["adx_period"])
    adx_threshold    = st.slider("ADX Trend Eşiği:",        15,  35,  value=ss["adx_threshold"])
    atr_period       = st.slider("ATR Periyodu:",            7,   30,  14)
    stoch_rsi_period = st.slider("Stoch RSI Periyodu:",      7,   21,  14)
    stoch_d_period   = st.slider("Stoch RSI %D Smoothing:",  2,   5,   3)
    stoch_lower      = st.slider("Stoch RSI Alt Eşik:",      5,   30,  20)
    stoch_upper      = st.slider("Stoch RSI Üst Eşik:",      70,  95,  80)
    ichi_tenkan      = st.slider("Tenkan-sen:",              5,   20,  9,
        help="⚠️ Ichimoku'da klasik değer 9'dur (Hosoda 1930'lar). Değiştirmek önerilmez — "
             "9-26-52 Schelling noktasıdır, dünya çapında bu değerler izlenir.")
    ichi_kijun       = st.slider("Kijun-sen:",               20,  40,  26,
        help="⚠️ Ichimoku'da klasik değer 26'dır. Değiştirmek önerilmez.")
    ichi_senkou_b    = st.slider("Senkou Span B:",           40,  65,  52,
        help="⚠️ Ichimoku'da klasik değer 52'dir. Değiştirmek önerilmez.")
    st_period        = st.slider("SuperTrend ATR Periyodu:", 5,   20,  value=ss["st_period"])
    st_multiplier    = st.slider("SuperTrend Çarpan:",       1.0, 5.0, value=ss["st_multiplier"], step=0.5)
    kama_period      = st.slider("KAMA Etkinlik Periyodu:",  5,   20,  10)
    kama_fast        = st.slider("KAMA Hızlı EMA:",          2,   5,   2)
    kama_slow        = st.slider("KAMA Yavaş EMA:",          20,  40,  30)
    lrc_period       = st.slider("LRC Periyodu:",            20,  100, value=ss["lrc_period"])
    lrc_std_mult     = st.slider("LRC Standart Sapma:",      1.0, 3.0, value=ss["lrc_std_mult"],  step=0.5)
    vwap_band_pct    = st.slider("VWAP Nötr Bant (%):",     0.0, 1.0, 0.1, step=0.05)

    st.write("---")
    st.subheader("📐 Fibonacci Ayarları")
    fib_lookback = st.slider("Fibonacci Lookback (bar):", 20, 300, 100)

    st.write("---")
    st.subheader("〰️ WaveTrend Ayarları")
    wt_n1 = st.slider("WaveTrend Kanal (n1):",    5,  20,  value=ss["wt_n1"])
    wt_n2 = st.slider("WaveTrend Ortalama (n2):", 10, 40,  value=ss["wt_n2"])
    wt_ob = st.slider("WaveTrend Aşırı Alım:",    40, 80,  60)
    wt_os = st.slider("WaveTrend Aşırı Satım:",  -80, -20, -60)

    st.write("---")
    st.subheader("🔀 Divergence Ayarları")
    div_window = st.slider("Divergence Pivot Pencere:", 3, 10, 5)

    # ── Destek/Direnç ve Trend Çizgisi Ayarları ───────────────
    st.write("---")
    st.subheader("📊 Destek / Direnç Ayarları")
    swing_window  = st.slider("S/R Pivot Pencere:",    3,  20, 10,
        help="Tepe/dip tespiti için her yönde bakılacak bar sayısı")
    swing_touches = st.slider("Min. Dokunuş Sayısı:", 1,   5,  1,
        help="1 = tek pivotlu seviyeler de gösterilir (daha fazla çizgi, zayıf güç)")
    swing_atr_k   = st.slider("ATR Tolerans Çarpanı:", 0.2, 2.0, 0.5, step=0.1,
        help="Seviye birleştirme toleransı = bu değer × ATR / fiyat. "
             "Volatil enstrümanlarda yükselt, sakin enstrümanlarda düşür.")
    swing_tol     = 0.003  # fallback (ATR yoksa kullanılır)

    st.write("---")
    st.subheader("📐 Trend Çizgisi Ayarları")
    tl_pivot_window = st.slider("TL Pivot Pencere:",       5,  20,  10,
        help="Trend çizgisi pivot tespiti için pencere genişliği")
    tl_max_lines    = st.slider("Max Çizgi Sayısı:",       1,   5,   3,
        help="Her yönde (destek/direnç) gösterilecek maksimum çizgi")
    tl_tolerance    = st.slider("TL Tolerans (%):",        0.3, 2.0, 1.2, step=0.1,
        help="Pivotun çizgiye dokundu sayılması için fiyat toleransı") / 100
    tl_show_channel = st.checkbox("Kanalları Göster", value=True,
        help="Paralel destek+direnç kanallarını dolgulu göster")
    # ──────────────────────────────────────────────────────────

    st.write("---")
    st.subheader("📊 Backtest Ayarları")
    commission_pct = st.slider("Komisyon (% / işlem):", 0.0, 1.0, 0.1, step=0.01)
    slippage_pct   = st.slider("Slippage (% / işlem):", 0.0, 0.5, 0.05, step=0.01)

    st.write("---")
    st.subheader("🔁 Walk-Forward Optimizasyon")
    n_windows = st.slider("Pencere Sayısı:", 2, 8, 3,
        help="Veri kaç eşit parçaya bölünsün? Train sabit boyutta kayar (sliding).")
    st.caption(f"{n_windows} pencere · sliding window (train sabit boyut, kaydırmalı)")

    st.write("---")
    run_opt = st.button("🚀 Algoritmaları Optimize Et", use_container_width=True, type="primary")
    st.info("İpucu: 1 dakikalık analizler için Periyot: 5d, Mum Aralığı: 1m seçiniz.")


# ============================================================
# 3. OPTİMİZASYON PARAMETRE GRİDLERİ
# ============================================================
PARAM_GRIDS = {
    "SMA Crossover":  {"sma_s":         [10, 20, 50],
                       "sma_l":         [100, 150, 200]},
    "RSI":            {"rsi_period":       [10, 14, 21],
                       "rsi_lower":        [25, 30, 35],
                       "rsi_upper":        [65, 70, 75]},
    "Bollinger Bands":{"bb_period":     [15, 20, 30],
                       "bb_std":        [1.5, 2.0, 2.5]},
    "MACD":           {"macd_fast":     [8, 12, 16],
                       "macd_slow":     [20, 26, 30],
                       "macd_signal":   [7, 9, 12]},
    "ADX":            {"adx_period":    [10, 14, 20],
                       "adx_threshold": [20, 25, 30]},
    "SuperTrend":     {"st_period":     [7, 10, 14],
                       "st_multiplier": [2.0, 2.5, 3.0, 3.5]},
    "LR Channel":     {"lrc_period":    [30, 50, 75],
                       "lrc_std_mult":  [1.5, 2.0, 2.5]},
    "WaveTrend":      {"wt_n1":         [8, 10, 14],
                       "wt_n2":         [15, 21, 28]},
    "OBV":            {"obv_short":     [5, 10, 15],
                       "obv_long":      [20, 30, 40]},
}


# ============================================================
# 4. YARDIMCI & SINYAL FONKSIYONLARI
# Tek kaynak: signals.py. Panel ve bot AYNI fonksiyonlari kullanir.
# (Eski duplike kopyalar kaldirildi.)
# ============================================================
from signals import (
    safe_scalar, flatten_columns,
    calc_adx, calc_kama, calc_supertrend,
    calc_linear_regression_channel, calc_vwap_daily,
    find_swing_levels, find_trendlines,
    calc_fibonacci, calc_wavetrend, detect_divergence,
    sig_sma, sig_rsi_fn, sig_bb, sig_macd, sig_obv, sig_adx_fn,
    sig_stochrsi, sig_ichimoku, sig_kama_fn, sig_supertrend_fn,
    sig_lrc, sig_vwap_fn, sig_wavetrend_fn,
    bars_per_year_from_interval, _strategy_bar_returns,
    permutation_pvalue, stationary_bootstrap_pvalue,
    deflated_sharpe_ratio, run_backtest,
)


def _score(stats, metric):
    if metric == "Sharpe":
        return stats["sharpe"]
    elif metric == "Getiri":
        return stats["total_ret"]
    else:
        dd = stats["max_dd"]
        return stats["total_ret"] / dd if dd > 0 else stats["total_ret"]


def optimize_algo(param_grid, signal_fn, close_arr, cost_pct,
                  n_windows=4, metric="Sharpe", min_trades=5,
                  bars_per_year=252, run_permutation=True, n_perm=200,
                  purge_bars=10, embargo_pct=0.01):
    """Gerçek walk-forward optimizasyon — Sliding Window (Purging & Embargo destekli):
      1) Her pencerede sabit boyutlu TRAIN dilimi üzerinde en iyi kombo seçilir
      2) Kazanan kombo TEST diliminde dokunulmamış OOS skoru alır
      3) En çok seçilen (ve en yüksek OOS skora sahip) kombo sistem tarafından döndürülür
      4) Tüm OOS test dilimleri birleştirilip permutation test ile p-değeri hesaplanır

    Sliding vs Expanding:
      - Sliding: train penceresi sabit boyutta kayar — eski veri düşer.
        Rejim değişimlerinde daha hızlı adapte olur; son döneme daha duyarlı.
      - Expanding: tüm geçmiş train'e dahil edilir (daha fazla veri ama daha yavaş adaptasyon).

    Purging & Embargo (López de Prado 2018):
      - purge_bars: Train sonundan kesilen bar sayısı. Train'de başlayıp test'e
        yayılabilecek trade'lerin label-leakage'ını engeller.
      - embargo_pct: Test başından atlanan bar oranı (veri uzunluğunun yüzdesi).
        Train sonundaki son trade'in test'in ilk barlarına sızmasını engeller.
    """
    keys    = list(param_grid.keys())
    combos  = list(iter_product(*param_grid.values()))
    n       = len(close_arr)
    default = {k: v[0] for k, v in param_grid.items()}

    # Embargo bar sayısı: veri uzunluğunun yüzdesi (López de Prado formülü)
    embargo_bars = max(0, int(n * embargo_pct))

    # ── SLIDING (ROLLING) WINDOW ──
    # Her pencerede train sabit boyutta kayar; eski veri düşer, yeni veri girer.
    # Eğitim dilimi [w*step → w*step + train_size], test dilimi [train_end → train_end + step].
    #   Adım 1: train=[0,       2*step), test=[2*step, 3*step)
    #   Adım 2: train=[step,    3*step), test=[3*step, 4*step)
    #   ...
    #   Adım n: train=[(n-1)*step, (n+1)*step), test=[(n+1)*step, end)
    # Her pencerede train boyutu sabittir → daha kararlı parametre seçimi.
    # Ek olarak purge (train sonu) ve embargo (test başı) uygulanır.
    n_steps = max(n_windows, 2)
    step_size = n // (n_steps + 2)  # sliding: train(2*step) + n_steps*step ≤ n
    if step_size < 15:
        return default, None

    windows = []
    train_size = 2 * step_size   # sabit train penceresi
    for w in range(n_steps):
        train_start = w * step_size
        train_end   = train_start + train_size
        test_start  = train_end
        test_end    = min(test_start + step_size, n) if w < n_steps - 1 else n
        if test_end > n:
            break

        # Purge: train sonundan purge_bars kes
        train_end_purged = train_end - purge_bars
        # Embargo: test başından embargo_bars atla
        test_start_embargoed = test_start + embargo_bars

        # Yetersiz veri kontrolü (purge/embargo sonrası)
        if (train_end_purged - train_start < 20 or
            test_end - test_start_embargoed < 10):
            continue
        windows.append((train_start, train_end_purged, test_start_embargoed, test_end))

    if not windows:
        return default, None

    # Kombolara göre OOS sonuçları
    combo_oos = {combo: [] for combo in combos}  # (test_stats, test_sig_slice, test_price_slice)

    # ── Sinyal cache (pencerelerden bağımsız, bir kez üretilir) ──
    # signal_fn(p) tüm fiyat dizisi için üretilir ve pencereye göre slice'lanır.
    # Bu yüzden aynı kombo için pencere başına yeniden hesaplamaya gerek yok.
    sigs_cache = {}
    for combo in combos:
        p = dict(zip(keys, combo))
        sig_full = signal_fn(p)
        if sig_full is None:
            continue
        sigs_cache[combo] = np.asarray(
            sig_full.values if hasattr(sig_full, "values") else sig_full
        )

    for (ts, te, ts_test, es) in windows:
        train_arr = close_arr[ts:te]
        test_arr  = close_arr[ts_test:es]

        # Adaptif min_trades: pencere kısaysa alt sınır 3'e iner,
        # uzun pencerelerde kullanıcının ayarladığı tavan geçerli olur
        train_bars = te - ts
        eff_min_trades = max(3, min(min_trades, train_bars // 30))

        # TRAIN: her kombo için skor, en iyiyi bul
        best_train_combo = None
        best_train_score = -np.inf
        for combo in combos:
            sig_vals = sigs_cache.get(combo)
            if sig_vals is None:
                continue
            train_sig = sig_vals[ts:te]
            train_stats = run_backtest(train_sig, train_arr, cost_pct, bars_per_year)
            if train_stats["n"] < eff_min_trades:
                continue
            sc = _score(train_stats, metric)
            if sc > best_train_score:
                best_train_score = sc
                best_train_combo = combo

        if best_train_combo is None:
            continue

        # TEST: yalnız train-kazananını out-of-sample test et (dokunulmamış + embargoed)
        test_sig  = sigs_cache[best_train_combo][ts_test:es]
        test_stats = run_backtest(test_sig, test_arr, cost_pct, bars_per_year)
        combo_oos[best_train_combo].append((test_stats, test_sig, test_arr))

    # En iyi kombo: en çok seçilen; eşitlikte en yüksek ortalama OOS skor
    winners = [(c, v) for c, v in combo_oos.items() if v]
    if not winners:
        return default, None

    def _rank_key(item):
        combo, results = item
        sel_count = len(results)
        avg_score = float(np.mean([_score(st, metric) for (st, _, _) in results]))
        return (sel_count, avg_score)

    best_combo, best_results = max(winners, key=_rank_key)
    best_p = dict(zip(keys, best_combo))

    # ── OOS aggregate stats (asla train verisi dahil değil) ──
    pooled_n  = sum(st["n"] for (st, _, _) in best_results)
    cumul = 1.0
    for (st, _, _) in best_results:
        cumul *= (1 + st["total_ret"] / 100)
    pooled_ret = (cumul - 1) * 100
    pooled_max_dd = max((st["max_dd"] for (st, _, _) in best_results), default=0.0)
    valid_stats = [st for (st, _, _) in best_results if st["n"] > 0]
    pooled_wr   = float(np.mean([st["win_rate"] for st in valid_stats])) if valid_stats else 0.0
    pooled_aw   = float(np.mean([st["avg_win"]  for st in valid_stats])) if valid_stats else 0.0
    pooled_al   = float(np.mean([st["avg_loss"] for st in valid_stats])) if valid_stats else 0.0
    finite_pfs  = [st["pf"] for st in valid_stats if st["pf"] != float("inf")]
    pooled_pf   = float(np.mean(finite_pfs)) if finite_pfs else float("inf")

    # ── Bar-bazlı OOS Sharpe: test dilimlerinin strateji getirileri concat ──
    all_strat_ret = []
    for (_, test_sig, test_price) in best_results:
        sr = _strategy_bar_returns(test_sig, test_price)
        if len(sr) > 0:
            # NaN/inf değerleri temizle
            sr = sr[np.isfinite(sr)]
            if len(sr) > 0:
                all_strat_ret.append(sr)
    if all_strat_ret:
        strat_ret_concat = np.concatenate(all_strat_ret)
        if len(strat_ret_concat) > 1 and strat_ret_concat.std() > 0:
            oos_sharpe = float(strat_ret_concat.mean() / strat_ret_concat.std() * np.sqrt(bars_per_year))
            if not np.isfinite(oos_sharpe):
                oos_sharpe = 0.0
        else:
            oos_sharpe = 0.0
    else:
        strat_ret_concat = np.array([])
        oos_sharpe = 0.0

    best_s = {
        "total_ret":     round(pooled_ret, 4),
        "sharpe":        round(oos_sharpe, 4),
        "n":             pooled_n,
        "win_rate":      round(pooled_wr, 2),
        "avg_win":       round(pooled_aw, 4),
        "avg_loss":      round(pooled_al, 4),
        "max_dd":        round(pooled_max_dd, 4),
        "pf":            round(pooled_pf, 4) if pooled_pf != float("inf") else float("inf"),
        "wf_windows":    len(windows),
        "wf_selections": len(best_results),
        "oos_only":      True,
    }

    # ── Stationary Bootstrap p-value (Politis & Romano 1994) ──
    # Basit permutation yerine zaman serisi yapısını koruyan blok bootstrap.
    if run_permutation and len(strat_ret_concat) > 20:
        # Ortalama blok uzunluğu: veri uzunluğunun ~küp köküne yakın (yaygın pratik)
        avg_block = max(5, int(len(strat_ret_concat) ** (1.0 / 3.0)))
        p_value = stationary_bootstrap_pvalue(
            strat_ret_concat, oos_sharpe, bars_per_year,
            n_boot=n_perm, avg_block_len=avg_block
        )
        best_s["p_value"] = round(p_value, 4)

    # ── Deflated Sharpe Ratio (Bailey & López de Prado 2014) ──
    # Multiple testing / data snooping cezası uygula.
    n_trials_grid = len(combos)  # bu algoritma için denenen kombo sayısı
    if n_trials_grid > 1 and len(strat_ret_concat) > 20:
        # Skewness ve kurtosis güvenli hesap (sıfır std, NaN ve inf koruması)
        try:
            std_ret = float(strat_ret_concat.std())
            if std_ret > 1e-12:
                demeaned = strat_ret_concat - strat_ret_concat.mean()
                sk_raw = float((demeaned ** 3).mean() / (std_ret ** 3))
                kt_raw = float((demeaned ** 4).mean() / (std_ret ** 4))
                # Aşırı değerleri kırp (DSR formülü aşırı kurtosis'e karşı hassas)
                sk = sk_raw if np.isfinite(sk_raw) else 0.0
                kt = kt_raw if np.isfinite(kt_raw) and kt_raw > 0 else 3.0
                # Güvenli sınırlar: makul finansal getiri dağılımı için
                sk = max(-5.0, min(5.0, sk))
                kt = max(1.0, min(30.0, kt))
            else:
                sk, kt = 0.0, 3.0

            dsr_val = deflated_sharpe_ratio(
                observed_sharpe=oos_sharpe,
                n_trials=n_trials_grid,
                n_obs=len(strat_ret_concat),
                skew=sk, kurt=kt,
            )
            if np.isfinite(dsr_val):
                best_s["dsr"] = round(float(dsr_val), 4)
                best_s["n_trials"] = n_trials_grid
            else:
                best_s["dsr"] = None
        except Exception as _dsr_err:
            best_s["dsr"] = None
            best_s["dsr_error"] = str(_dsr_err)[:100]

    return best_p, best_s


# ============================================================
# 7. VERİ ÇEKME
# ============================================================
@st.cache_data(ttl=55)
def fetch_live_data(symbol, p, i):
    try:
        fetch_i = "1h" if i in ("4h", "8h") else i
        data = yf.download(symbol, period=p, interval=fetch_i, progress=False)
        if data is None or data.empty:
            return pd.DataFrame()
        if i in ("4h", "8h"):
            if isinstance(data.columns, pd.MultiIndex):
                uniq = data.columns.get_level_values(1).unique()
                data.columns = (data.columns.get_level_values(0)
                                if len(uniq) <= 1
                                else [f"{c[1]}_{c[0]}" for c in data.columns])
            rule = "4h" if i == "4h" else "8h"
            data = (
                data.resample(rule)
                .agg({"Open": "first", "High": "max", "Low": "min",
                      "Close": "last", "Volume": "sum"})
                .dropna()
            )
        return data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()


PLOTLY_CONFIG = dict(scrollZoom=True, displayModeBar=True,
    modeBarButtonsToAdd=["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    modeBarButtonsToRemove=["lasso2d", "select2d"])


def sub_layout(height=250):
    return dict(template="plotly_dark", height=height, margin=dict(t=30, b=30), dragmode="pan")


# ============================================================
# 8. ANA MANTIK
# ============================================================
if ticker:
    df = fetch_live_data(ticker, period, interval)

    if not df.empty:
        df = flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
        if missing:
            st.error(f"Eksik sütunlar: {missing}.")
            st.stop()

        close     = df["Close"].squeeze()
        high      = df["High"].squeeze()
        low       = df["Low"].squeeze()
        volume    = df["Volume"].squeeze()
        close_arr = close.values
        n_bars    = len(close)

        indicator_min_reqs = {
            "SMA Crossover":    sma_long,
            "Bollinger Bands":  bb_period,
            "RSI":              rsi_period * 2,
            "MACD":             macd_slow + macd_signal,
            "OBV":              obv_long,
            "ADX":              adx_period * 3,
            "Stoch RSI":        rsi_period + stoch_rsi_period,
            "Ichimoku":         ichi_senkou_b + ichi_kijun,
            "KAMA":             kama_period + kama_slow,
            "SuperTrend":       st_period * 2,
            "LR Channel":       lrc_period,
            "WaveTrend":        wt_n1 + wt_n2,
            "Walk-Forward Opt": 150,
        }

        affected = [
            f"{name} (min {req} mum)"
            for name, req in indicator_min_reqs.items()
            if n_bars < req
        ]

        min_req = max(150, adx_period * 3, ichi_senkou_b)
        if n_bars < min_req:
            if affected:
                st.warning(
                    f"⚠️ Yeterli veri yok: **{n_bars} mum** mevcut, en az **{min_req}** gerekli.\n\n"
                    f"**Etkilenen indikatörler:** {', '.join(affected)}"
                )
            else:
                st.warning(f"Yeterli veri yok: {n_bars} mum, en az {min_req} gerekli.")

        cost_pct    = (commission_pct + slippage_pct) / 100
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]

        # ATR
        tr1        = high - low
        tr2        = (high - close.shift(1)).abs()
        tr3        = (low  - close.shift(1)).abs()
        tr         = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        atr_ma     = atr_series.rolling(atr_period, min_periods=atr_period).mean()
        atr_high   = (atr_series > atr_ma).values

        # ── YENİ: 200 EMA ─────────────────────────────────────────
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
        # ──────────────────────────────────────────────────────────

        # ── Swing Destek/Direnç (yatay) ───────────────────────────
        swing_levels = find_swing_levels(
            high, low, close,
            window=swing_window,
            min_touches=swing_touches,
            tolerance=swing_tol,
            atr_series=atr_series,
            atr_k=swing_atr_k,
        )

        # ── Diyagonal Trend Çizgileri ──────────────────────────────
        trendlines, tl_channels, tl_dates = find_trendlines(
            high, low, close,
            pivot_window=tl_pivot_window,
            max_lines=tl_max_lines,
            tolerance=tl_tolerance,
        )
        # ──────────────────────────────────────────────────────────

        # ============================================================
        # OPTİMİZASYON
        # ============================================================
        OPT_KEY = f"opt_v6_dsr_{ticker}_{period}_{interval}_{n_windows}"

        # ── Veri uzunluğu uyarısı ─────────────────────────────────────────────
        _n_bars = len(close)
        _max_trend = max(
            200,  # RSI trend filtresi sabit 200
            max(PARAM_GRIDS["SMA Crossover"].get("sma_l", [200])),
        )
        _min_recommended = _max_trend * 3  # train + test + warmup için güvenli alt sınır
        if _n_bars < _min_recommended:
            st.warning(
                f"⚠️ **Yetersiz veri:** {_n_bars} bar mevcut, "
                f"SMA{_max_trend} filtresi için en az **{_min_recommended} bar** önerilir. "
                f"Daha uzun periyot seçin (örn. 5 yıl) veya SMA200 içeren kombinasyonlar "
                f"optimize edilemeyebilir."
            )
        # ─────────────────────────────────────────────────────────────────────

        if run_opt or OPT_KEY not in st.session_state:
            opt_params = {}
            opt_stats  = {}
            prog       = st.progress(0, text="Optimizasyon başlatılıyor…")
            algo_list  = list(PARAM_GRIDS.keys())

            for idx, algo_name in enumerate(algo_list):
                prog.progress(idx / len(algo_list), text=f"Optimize ediliyor: {algo_name}")
                grid = PARAM_GRIDS[algo_name]

                if algo_name == "SMA Crossover":
                    def make_fn():
                        def fn(p):
                            if p["sma_s"] >= p["sma_l"]: return None
                            s, _, _ = sig_sma(close, p["sma_s"], p["sma_l"]); return s
                        return fn
                elif algo_name == "RSI":
                    def make_fn():
                        def fn(p):
                            if p["rsi_lower"] >= p["rsi_upper"]: return None
                            s, _ = sig_rsi_fn(close, p["rsi_period"], p["rsi_lower"], p["rsi_upper"]); return s
                        return fn
                elif algo_name == "Bollinger Bands":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_bb(close, p["bb_period"], p["bb_std"]); return s
                        return fn
                elif algo_name == "MACD":
                    def make_fn():
                        def fn(p):
                            if p["macd_fast"] >= p["macd_slow"]: return None
                            s, _, _ = sig_macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"]); return s
                        return fn
                elif algo_name == "ADX":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_adx_fn(high, low, close, p["adx_period"], p["adx_threshold"]); return s
                        return fn
                elif algo_name == "SuperTrend":
                    def make_fn():
                        def fn(p):
                            s, _, _, _, _ = sig_supertrend_fn(high, low, close, p["st_period"], p["st_multiplier"]); return s
                        return fn
                elif algo_name == "LR Channel":
                    def make_fn():
                        def fn(p):
                            s, _, _, _, _, _ = sig_lrc(close, p["lrc_period"], p["lrc_std_mult"]); return s
                        return fn
                elif algo_name == "WaveTrend":
                    # WaveTrend filtresi için baseline RSI(14) + RSI_MA(14) kullan
                    # (optimize sırasında rsi_period'tan bağımsız tutarlı filtre)
                    _wt_rsi_delta = close.diff()
                    _wt_rsi_gain  = _wt_rsi_delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                    _wt_rsi_loss  = (-_wt_rsi_delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                    _wt_rsi       = 100 - 100 / (1 + _wt_rsi_gain / _wt_rsi_loss.replace(0, np.nan))
                    _wt_rsi_ma    = _wt_rsi.rolling(14).mean()
                    def make_fn():
                        def fn(p):
                            s, _, _ = sig_wavetrend_fn(high, low, close, _wt_rsi, _wt_rsi_ma,
                                                       p["wt_n1"], p["wt_n2"], wt_ob, wt_os); return s
                        return fn
                elif algo_name == "OBV":
                    def make_fn():
                        def fn(p):
                            if p["obv_short"] >= p["obv_long"]: return None
                            s, _, _, _ = sig_obv(close, volume, p["obv_short"], p["obv_long"]); return s
                        return fn

                best_p, best_s = optimize_algo(
                    grid, make_fn(), close_arr, cost_pct,
                    n_windows=n_windows,
                    metric="Sharpe", min_trades=5,
                    bars_per_year=bars_per_year_from_interval(interval),
                    run_permutation=True, n_perm=200)
                opt_params[algo_name] = best_p
                opt_stats[algo_name]  = best_s if best_s else {"total_ret": 0.0, "sharpe": 0.0, "n": 0, "win_rate": 0.0}

            prog.progress(1.0, text="✅ Optimizasyon tamamlandı!")
            st.session_state[OPT_KEY] = {"params": opt_params, "stats": opt_stats}

            p = opt_params
            st.session_state["sma_short"]     = int(p["SMA Crossover"]["sma_s"])
            st.session_state["sma_long"]      = int(p["SMA Crossover"]["sma_l"])
            st.session_state["rsi_period"]    = int(p["RSI"]["rsi_period"])
            st.session_state["rsi_lower"]     = int(p["RSI"]["rsi_lower"])
            st.session_state["rsi_upper"]     = int(p["RSI"]["rsi_upper"])
            st.session_state["bb_period"]     = int(p["Bollinger Bands"]["bb_period"])
            st.session_state["bb_std"]        = float(p["Bollinger Bands"]["bb_std"])
            st.session_state["macd_fast"]     = int(p["MACD"]["macd_fast"])
            st.session_state["macd_slow"]     = int(p["MACD"]["macd_slow"])
            st.session_state["macd_signal"]   = int(p["MACD"]["macd_signal"])
            st.session_state["adx_period"]    = int(p["ADX"]["adx_period"])
            st.session_state["adx_threshold"] = int(p["ADX"]["adx_threshold"])
            st.session_state["st_period"]     = int(p["SuperTrend"]["st_period"])
            st.session_state["st_multiplier"] = float(p["SuperTrend"]["st_multiplier"])
            st.session_state["lrc_period"]    = int(p["LR Channel"]["lrc_period"])
            st.session_state["lrc_std_mult"]  = float(p["LR Channel"]["lrc_std_mult"])
            st.session_state["wt_n1"]         = int(p["WaveTrend"]["wt_n1"])
            st.session_state["wt_n2"]         = int(p["WaveTrend"]["wt_n2"])
            st.session_state["obv_short"]     = int(p["OBV"]["obv_short"])
            st.session_state["obv_long"]      = int(p["OBV"]["obv_long"])
            st.rerun()

        else:
            opt_params = st.session_state[OPT_KEY]["params"]
            opt_stats  = st.session_state[OPT_KEY]["stats"]

        p_sma  = {"sma_s": sma_short,   "sma_l": sma_long}
        p_rsi  = {"rsi_period": rsi_period, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper}
        p_bb   = {"bb_period": bb_period,   "bb_std": bb_std}
        p_macd = {"macd_fast": macd_fast,   "macd_slow": macd_slow, "macd_signal": macd_signal}
        p_adx  = {"adx_period": adx_period, "adx_threshold": adx_threshold}
        p_st   = {"st_period": st_period,   "st_multiplier": st_multiplier}
        p_lrc  = {"lrc_period": lrc_period, "lrc_std_mult": lrc_std_mult}
        p_wt   = {"wt_n1": wt_n1,           "wt_n2": wt_n2}

        df["Sig_SMA"], df["SMA_SHORT"], df["SMA_LONG"] = sig_sma(
            close, p_sma["sma_s"], p_sma["sma_l"])

        # SMA 200 (EMA 200 ile karşılaştırma için — daha yavaş, daha stabil)
        df["SMA200"] = close.rolling(200, min_periods=200).mean()

        df["Sig_RSI"], df["RSI"] = sig_rsi_fn(
            close, p_rsi["rsi_period"], p_rsi["rsi_lower"], p_rsi["rsi_upper"],
            trend_period=rsi_trend_period)
        df["RSI_MA"] = df["RSI"].rolling(rsi_ma_period).mean()

        df["Sig_BB"], df["Mid"], df["Up"], df["Low_BB"] = sig_bb(
            close, p_bb["bb_period"], p_bb["bb_std"])

        df["Sig_MACD"], df["MACD"], df["MACD_S"] = sig_macd(
            close, p_macd["macd_fast"], p_macd["macd_slow"], p_macd["macd_signal"])

        df["Sig_OBV"], df["OBV"], obv_sma_short, obv_sma_long = sig_obv(
            close, volume, obv_short, obv_long)

        df["Sig_ADX"], df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = sig_adx_fn(
            high, low, close, p_adx["adx_period"], p_adx["adx_threshold"])

        df["Sig_StochRSI"], df["StochRSI_K"], df["StochRSI_D"] = sig_stochrsi(
            close, df["RSI"], df["RSI_MA"], stoch_rsi_period, stoch_d_period, stoch_lower, stoch_upper)

        df["Sig_Ichimoku"], df["Tenkan"], df["Kijun"], df["Senkou_A"], df["Senkou_B"], df["Chikou"] = sig_ichimoku(
            high, low, close, ichi_tenkan, ichi_kijun, ichi_senkou_b)

        df["Sig_KAMA"], df["KAMA"], df["KAMA_ER"] = sig_kama_fn(
            close, kama_period, kama_fast, kama_slow)

        df["Sig_SuperTrend"], df["SuperTrend"], df["ST_Direction"], df["ST_Lower"], df["ST_Upper"] = sig_supertrend_fn(
            high, low, close, p_st["st_period"], p_st["st_multiplier"])

        df["Sig_LRC"], df["LRC_Mid"], df["LRC_Upper"], df["LRC_Lower"], df["LRC_Slope"], df["LRC_R2"] = sig_lrc(
            close, p_lrc["lrc_period"], p_lrc["lrc_std_mult"])

        df["ATR"]      = atr_series
        df["ATR_High"] = atr_high

        if is_intraday:
            df["Sig_VWAP"], df["VWAP"] = sig_vwap_fn(high, low, close, volume, vwap_band_pct)
        else:
            df["Sig_VWAP"] = 0
            df["VWAP"]     = np.nan

        df["Sig_WaveTrend"], df["WT1"], df["WT2"] = sig_wavetrend_fn(
            high, low, close, df["RSI"], df["RSI_MA"],
            p_wt["wt_n1"], p_wt["wt_n2"], wt_ob, wt_os)

        fib_levels, fib_high, fib_low, fib_direction = calc_fibonacci(
            high, low, close, lookback=fib_lookback)

        df["Div_RSI"]  = detect_divergence(close, df["RSI"],  window=div_window)
        df["Div_MACD"] = detect_divergence(close, df["MACD"], window=div_window)
        df["Div_OBV"]  = detect_divergence(close, df["OBV"],  window=div_window)

        # ============================================================
        # ANA GRAFİK + VRP
        # ============================================================
        from plotly.subplots import make_subplots

        bull_st = df["ST_Direction"] == 1
        bear_st = df["ST_Direction"] == -1

        st_dir_shifted = df["ST_Direction"].shift(1).fillna(0)
        st_buy_signal  = (df["ST_Direction"] == 1)  & (st_dir_shifted != 1)
        st_sell_signal = (df["ST_Direction"] == -1) & (st_dir_shifted != -1)

        lp = float(close.iloc[-1])
        pp = float(close.iloc[-2]) if len(close) > 1 else lp

        vrp_bins     = 40
        if show_vp:
            price_min    = float(low.min())
            price_max    = float(high.max())
            bin_edges    = np.linspace(price_min, price_max, vrp_bins + 1)
            bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
            vol_at_price = np.zeros(vrp_bins)
            for i in range(len(df)):
                lo_i  = float(low.iloc[i])
                hi_i  = float(high.iloc[i])
                vol_i = float(volume.iloc[i])
                if hi_i == lo_i:
                    idx = np.clip(np.searchsorted(bin_edges, lo_i, side="right") - 1, 0, vrp_bins - 1)
                    vol_at_price[idx] += vol_i
                else:
                    for b in range(vrp_bins):
                        overlap = min(hi_i, bin_edges[b+1]) - max(lo_i, bin_edges[b])
                        if overlap > 0:
                            vol_at_price[b] += vol_i * overlap / (hi_i - lo_i)

            poc_idx   = int(np.argmax(vol_at_price))
            poc_price = bin_centers[poc_idx]
            max_vol   = vol_at_price.max()
            bar_colors = [
                "rgba(255,165,0,1.0)" if b == poc_idx
                else f"rgba(100,{int(80 + 175*(v/max_vol)) if max_vol > 0 else 200},255,0.85)"
                for b, v in enumerate(vol_at_price)
            ]

        if show_vp:
            fig = make_subplots(
                rows=2, cols=2,
                row_heights=[0.20, 0.80],
                column_widths=[0.85, 0.15],
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0.0,
                vertical_spacing=0.02,
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.20, 0.80],
                shared_xaxes=True,
                vertical_spacing=0.02,
            )

        # ── ÜST MİNİ PANEL: WT_CROSS_LB (bilgi amaçlı, saf cross) ─────
        # df["WT1"], df["WT2"] line ~2109'da zaten hesaplandı; yeniden hesaplamıyoruz.
        # Buradaki cross noktaları BÖLGE FİLTRESİZ — LazyBear orijinal davranışı.
        _wt1 = df["WT1"]; _wt2 = df["WT2"]
        _wt_cu = (_wt1 > _wt2) & (_wt1.shift(1) <= _wt2.shift(1))
        _wt_cd = (_wt1 < _wt2) & (_wt1.shift(1) >= _wt2.shift(1))
        # OB/OS yatay çizgileri (referans)
        fig.add_hline(y=wt_ob, line=dict(color="rgba(255,80,80,0.35)", width=1, dash="dot"), row=1, col=1)
        fig.add_hline(y=wt_os, line=dict(color="rgba(80,255,80,0.35)", width=1, dash="dot"), row=1, col=1)
        fig.add_hline(y=0,     line=dict(color="rgba(150,150,150,0.25)", width=1), row=1, col=1)
        # WT çizgileri
        fig.add_trace(go.Scatter(x=df.index, y=_wt1, name="WT1",
            line=dict(color="#00e5ff", width=1.4), showlegend=False,
            hovertemplate="WT1: %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=_wt2, name="WT2",
            line=dict(color="#ff9800", width=1.2, dash="dot"), showlegend=False,
            hovertemplate="WT2: %{y:.2f}<extra></extra>"), row=1, col=1)
        # Cross noktaları (filtresiz — tüm kesişimler)
        if _wt_cu.any():
            fig.add_trace(go.Scatter(x=df.index[_wt_cu], y=_wt2[_wt_cu],
                mode="markers", name="WT bull cross",
                marker=dict(color="#00e676", size=6, line=dict(color="#003d20", width=0.5)),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
        if _wt_cd.any():
            fig.add_trace(go.Scatter(x=df.index[_wt_cd], y=_wt2[_wt_cd],
                mode="markers", name="WT bear cross",
                marker=dict(color="#ff5252", size=6, line=dict(color="#3d0000", width=0.5)),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
        # Panel başlığı (sol üst köşe, küçük)
        fig.add_annotation(
            xref="x domain", yref="y domain", x=0.005, y=0.92,
            text="<b>WT_CROSS_LB</b>", showarrow=False,
            font=dict(color="rgba(200,200,200,0.65)", size=9, family="monospace"),
            row=1, col=1,
        )
        # ──────────────────────────────────────────────────────────────

        if chart_type == "Mum":
            # ── Sinyal bazlı mum renklendirme ─────────────────────
            _rsi_mid    = (rsi_lower + rsi_upper) / 2
            cyan_raw   = (df["ST_Direction"] == 1) & (df["Sig_OBV"] == 1) & (df["RSI"] < rsi_upper)
            cyan_mask  = cyan_raw & ~cyan_raw.shift(1).fillna(False)
            yellow_mask = (~cyan_mask) & (df["ADX"] < adx_threshold) & (df["RSI"] >= _rsi_mid - 5) & (df["RSI"] <= _rsi_mid + 5)
            red_mask   = (~cyan_mask) & (~yellow_mask) & (df["Close"] < df["Open"]) & (df["MACD"] < df["MACD_S"])
            green_mask = ~cyan_mask & ~yellow_mask & ~red_mask

            _color_defs = [
                ("Cyan AL",  cyan_mask,   "#00ffff"),
                ("Yeşil",    green_mask,  "#00cc66"),
                ("Sarı",     yellow_mask, "#ffcc00"),
                ("Ayı",      red_mask,    "#ff4444"),
            ]
            for _lbl, _mask, _color in _color_defs:
                _rising  = _mask & (df["Close"] >= df["Open"])
                _falling = _mask & (df["Close"] <  df["Open"])
                for _m, _fill, _trace_lbl in [
                    (_rising,  _color,   _lbl + " ↑"),
                    (_falling, "#111111", _lbl + " ↓"),
                ]:
                    if _m.any():
                        fig.add_trace(go.Candlestick(
                            x=df.index[_m],
                            open=df["Open"][_m], high=df["High"][_m],
                            low=df["Low"][_m],   close=df["Close"][_m],
                            name=_trace_lbl,
                            increasing_fillcolor=_fill, increasing_line_color=_color,
                            decreasing_fillcolor=_fill, decreasing_line_color=_color,
                            showlegend=False,
                        ), row=2, col=1)

            # ── Divergence marker katmanı (ana grafik) ────────────
            bull_div = (df["Div_RSI"] == 1) | (df["Div_MACD"] == 1) | (df["Div_OBV"] == 1)
            bear_div = (df["Div_RSI"] == -1) | (df["Div_MACD"] == -1) | (df["Div_OBV"] == -1)
            if bull_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bull_div], y=df["Low"][bull_div] * 0.998,
                    mode="markers", name="Bullish Div 🔺",
                    marker=dict(symbol="triangle-up", color="lime", size=10),
                ), row=2, col=1)
            if bear_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bear_div], y=df["High"][bear_div] * 1.002,
                    mode="markers", name="Bearish Div 🔻",
                    marker=dict(symbol="triangle-down", color="red", size=16),
                ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="orange", width=1.5)), row=2, col=1)

        # ── Mum renk legend girişleri (dummy scatter) ─────────────
        if chart_type == "Mum":
            for _leg_name, _leg_color in [
                ("🔴 Ayı",         "#ff4444"),
                ("🟡 Kararsız",    "#ffcc00"),
                ("🟢 Boğa",        "#00cc66"),
                ("🔵 Güçlü Boğa",  "#00ffff"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=_leg_name,
                    marker=dict(symbol="square", size=24, color=_leg_color),
                    showlegend=True,
                ), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SHORT"],
            name=f"SMA {p_sma['sma_s']}",
            line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_LONG"],
            name=f"SMA {p_sma['sma_l']}",
            line=dict(color="cyan")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["KAMA"],
            name="KAMA", line=dict(color="violet", width=1.5),
            visible="legendonly"), row=2, col=1)

        # ── YENİ: 200 EMA trace ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"],
            name="EMA 200",
            line=dict(color="yellow", width=2, dash="dot"),
            visible="legendonly",
        ), row=2, col=1)
        # SMA 200 — daha stabil, EMA'ya göre yavaş, uzun vade referansı
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA200"],
            name="SMA 200",
            line=dict(color="gold", width=2, dash="solid"),
        ), row=2, col=1)
        # ──────────────────────────────────────────────────────────

        fig.add_trace(go.Scatter(
            x=df.index[bull_st], y=df["SuperTrend"][bull_st],
            name="SuperTrend (Boğa çizgi)", mode="lines",
            line=dict(color="rgba(0,255,100,0.5)", width=1.5),
            visible=False, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index[bear_st], y=df["SuperTrend"][bear_st],
            name="SuperTrend (Ayı çizgi)", mode="lines",
            line=dict(color="rgba(255,60,60,0.5)", width=1.5),
            visible=False, showlegend=False), row=2, col=1)

        if st_buy_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_buy_signal],
                y=df["SuperTrend"][st_buy_signal],
                name="SuperTrend AL",
                mode="markers+text",
                marker=dict(symbol="square", color="#00c853", size=18, line=dict(color="#00c853", width=0)),
                text="AL",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
                visible="legendonly",
            ), row=2, col=1)

        if st_sell_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_sell_signal],
                y=df["SuperTrend"][st_sell_signal],
                name="SuperTrend SAT",
                mode="markers+text",
                marker=dict(symbol="square", color="#d50000", size=18, line=dict(color="#d50000", width=0)),
                text="SAT",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
                visible="legendonly",
            ), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"],
            name="LRC Orta", visible=False, showlegend=False,
            line=dict(color="white", width=1, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"],
            name="LRC Üst", visible=False, showlegend=False,
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"],
            name="LRC Alt", visible=False, showlegend=False,
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=2, col=1)

        if is_intraday:
            fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"],
                name="VWAP", visible="legendonly",
                line=dict(color="yellow", dash="dash", width=1.5)), row=2, col=1)

        FIB_COLORS = {
            "0.0%":   "rgba(128,128,128,0.7)",
            "23.6%":  "rgba(255,165,0,0.8)",
            "38.2%":  "rgba(255,215,0,0.9)",
            "50.0%":  "rgba(255,255,255,0.9)",
            "61.8%":  "rgba(255,215,0,0.9)",
            "78.6%":  "rgba(255,165,0,0.8)",
            "100.0%": "rgba(128,128,128,0.7)",
        }
        for lvl_name, lvl_price in fib_levels.items():
            fig.add_hline(
                y=lvl_price,
                line_dash="dot",
                line_color=FIB_COLORS.get(lvl_name, "gray"),
                line_width=1,
                annotation_text=f"  Fib {lvl_name} {lvl_price:.2f}",
                annotation_font=dict(color=FIB_COLORS.get(lvl_name, "gray"), size=9, family="monospace"),
                annotation_position="top left",
                row=2, col=1,
            )

        # ── Yatay S/R çizgileri (legend toggle destekli, güce göre kalınlık) ──
        # Hepsi "Swing S/R" legend grubu altında — tek yerden aç/kapa
        x_start = df.index[0]
        x_end   = df.index[-1]
        _swing_first = True
        for lvl in swing_levels:
            is_support = lvl["type"] == "S"
            t          = lvl["touches"]
            broken     = lvl.get("broken", False)

            # Kalınlık: dokunuş sayısına göre
            width = 1 if t <= 1 else (2 if t == 2 else 3)
            # Çizgi stili
            dash  = "dash" if t <= 1 else ("dashdot" if t == 2 else "solid")
            # Opaklık
            alpha = min(0.40 + 0.15 * t, 0.80)

            if broken:
                color = f"rgba(160,160,160,{alpha*0.6:.2f})"
                status = " [kırık]"
            else:
                color = (f"rgba(0,255,100,{alpha:.2f})" if is_support
                         else f"rgba(255,80,80,{alpha:.2f})")
                status = ""

            sr_label = (f"{'🟢 Destek' if is_support else '🔴 Direnç'} "
                        f"{lvl['price']:.2f} (x{t}){status}")

            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[lvl["price"], lvl["price"]],
                mode="lines",
                name=sr_label,
                line=dict(color=color, width=width, dash=dash),
                visible=False,
                showlegend=False,
                legendgroup="swing_sr",
                legendgrouptitle_text="Swing S/R" if _swing_first else None,
                hovertemplate=f"{sr_label}<extra></extra>",
            ), row=2, col=1)
            _swing_first = False

        # ── Diyagonal Trend Çizgileri (legend toggle destekli) ────
        for tl in trendlines:
            is_sup  = tl["type"] == "support"
            color   = "rgba(0,255,120,0.9)" if is_sup else "rgba(255,80,80,0.9)"
            width   = 1 if tl["touches"] <= 2 else (2 if tl["touches"] <= 4 else 3)
            label   = f"{'↗ Destek' if is_sup else '↘ Direnç'} TL (x{tl['touches']})"
            x0_date = tl_dates[tl["x0"]]
            x1_date = tl_dates[tl["x1"]]
            fig.add_trace(go.Scatter(
                x=[x0_date, x1_date],
                y=[tl["y0"], tl["y1"]],
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash="solid"),
                visible="legendonly",
                legendgroup="trendlines",
                legendgrouptitle_text="Trend Çizgileri" if tl == trendlines[0] else None,
            ), row=2, col=1)

        # ── Kanal dolgusu (legend toggle destekli) ────────────────
        if tl_show_channel:
            for ci, ch in enumerate(tl_channels):
                sl   = ch["support"];  rl = ch["resistance"]
                xi0  = max(sl["x0"], rl["x0"])
                xi1  = sl["x1"]
                xs   = [tl_dates[xi0], tl_dates[xi1],
                        tl_dates[xi1], tl_dates[xi0], tl_dates[xi0]]
                y_s0 = sl["slope"] * xi0 + sl["intercept"]
                y_s1 = sl["slope"] * xi1 + sl["intercept"]
                y_r0 = rl["slope"] * xi0 + rl["intercept"]
                y_r1 = rl["slope"] * xi1 + rl["intercept"]
                ys   = [y_s0, y_s1, y_r1, y_r0, y_s0]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    fill="toself",
                    fillcolor="rgba(100,180,255,0.07)",
                    line=dict(width=0),
                    mode="lines",
                    name=f"Kanal {ci+1}",
                    visible="legendonly",
                    legendgroup="trendlines",
                    showlegend=True,
                ), row=2, col=1)
        # ──────────────────────────────────────────────────────────

        if show_vp:
            fig.add_trace(go.Bar(
                x=vol_at_price, y=bin_centers,
                orientation="h",
                marker_color=bar_colors,
                name="Hacim Profili",
                showlegend=False,
                hovertemplate="Fiyat: %{y:.2f}<br>Hacim: %{x:,.0f}<extra></extra>",
            ), row=2, col=2)

            fig.add_hline(y=poc_price, line_dash="dash", line_color="orange",
                annotation_text=f"POC {poc_price:.2f}",
                annotation_font=dict(color="orange", size=10, family="monospace"),
                annotation_bgcolor="rgba(255,165,0,0.15)",
                annotation_position="top right", row=2, col=2)
            fig.add_hline(y=lp, line_dash="dot", line_color="lime" if lp >= pp else "red",
                annotation_text=f"  {lp:.2f}",
                annotation_font=dict(color="lime" if lp >= pp else "red", size=12, family="monospace"),
                annotation_bgcolor="rgba(0,255,0,0.12)" if lp >= pp else "rgba(255,0,0,0.12)",
                annotation_position="bottom right", row=2, col=2)
        else:
            # VP kapalı → son fiyat etiketi ana grafiğin sağ kenarında
            fig.add_hline(y=lp, line_dash="dot", line_color="lime" if lp >= pp else "red",
                annotation_text=f" {lp:.2f}",
                annotation_font=dict(color="black", size=12, family="monospace"),
                annotation_bgcolor="lime" if lp >= pp else "red",
                annotation_bordercolor="rgba(0,0,0,0.6)",
                annotation_position="right", row=2, col=1)

        _layout_common = dict(
            template="plotly_dark", height=720,
            dragmode="pan",
            legend=dict(
                orientation="v",
                x=-0.02, y=1,
                xanchor="right", yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
                itemwidth=30,
                itemsizing="constant",
                tracegroupgap=4,
            ),
            margin=dict(l=110, r=10, t=30, b=30),
        )
        if show_vp:
            fig.update_layout(
                **_layout_common,
                # row=1, col=1 → WT mini panel
                xaxis=dict(showgrid=True, showticklabels=False, rangeslider_visible=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=9), zeroline=False),
                # row=1, col=2 → boş köşe
                xaxis2=dict(showgrid=False, showticklabels=False, visible=False),
                yaxis2=dict(showgrid=False, showticklabels=False, visible=False),
                # row=2, col=1 → ana grafik
                xaxis3=dict(rangeslider_visible=False),
                # row=2, col=2 → hacim profili
                xaxis4=dict(showgrid=False, showticklabels=False),
                yaxis4=dict(showticklabels=False),
            )
        else:
            fig.update_layout(
                **_layout_common,
                # row=1, col=1 → WT mini panel (skala sağda)
                xaxis=dict(showgrid=True, showticklabels=False, rangeslider_visible=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=9), zeroline=False, side="right"),
                # row=2, col=1 → ana grafik (fiyat skalası sağda)
                xaxis2=dict(rangeslider_visible=False),
                yaxis2=dict(side="right"),
            )

        _hdr_last_close = float(df["Close"].iloc[-1])
        _hdr_prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else _hdr_last_close
        _hdr_diff = _hdr_last_close - _hdr_prev_close
        _hdr_pct  = (_hdr_diff / _hdr_prev_close * 100) if _hdr_prev_close else 0.0
        if _hdr_diff > 0:
            _hdr_color, _hdr_arrow, _hdr_sign = "#00c853", "▲", "+"
        elif _hdr_diff < 0:
            _hdr_color, _hdr_arrow, _hdr_sign = "#ff4b4b", "▼", ""
        else:
            _hdr_color, _hdr_arrow, _hdr_sign = "#bbbbbb", "▬", ""
        st.markdown(
            f"## {ticker} &nbsp;·&nbsp; "
            f"<span style='color:{_hdr_color}'>{_hdr_last_close:.2f}</span> &nbsp;&nbsp; "
            f"<span style='color:{_hdr_color};font-size:0.7em'>"
            f"{_hdr_arrow} {_hdr_sign}{_hdr_diff:.2f} ({_hdr_sign}{_hdr_pct:.2f}%)</span>"
            f" &nbsp;&nbsp;&nbsp;&nbsp; "
            f"<span style='color:#888;font-size:0.55em;font-family:monospace'>"
            f"{period.upper()} · {interval.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # ============================================================
        # ANA GRAFİK REHBERİ (Expander)
        # ============================================================
        with st.expander("📖 Ana Grafikte Ne Ne Anlama Geliyor? (Detaylı Rehber)", expanded=False):
            st.markdown("""
### 🕯️ Mum Renkleri

Her mum 4 kategoriden birine atanır. Hiyerarşik sıralama: önce **Cyan** kontrol edilir, olmazsa **Sarı**, olmazsa **Kırmızı**, kalanlar **Yeşil**.

| Renk | Anlamı | Tetikleyici |
|---|---|---|
| 🔵 **Cyan (Güçlü Boğa)** | Taze AL sinyali | SuperTrend yukarı **VE** OBV birikim **VE** RSI aşırı alım değil **VE** önceki barda bu koşul yoktu |
| 🟢 **Yeşil (Boğa)** | Normal yükseliş bağlamı | Diğer üç kategoriye girmeyen mumlar (varsayılan) |
| 🟡 **Sarı (Kararsız)** | Yatay/düşük momentum | ADX zayıf **VE** RSI nötr bölgede (eşiklerin ortası ±5) |
| 🔴 **Kırmızı (Ayı)** | Momentumlu düşüş | Düşüş mumu **VE** MACD negatif |

**Gövde dolgu farkı:** Yükselen mumlar (Close ≥ Open) kategori renginde **dolu**. Düşen mumlar (Close < Open) **içi siyah**, kenarı kategori renginde. Böylece hem renk kategorisi hem yön tek bakışta görünür.

---

### 📈 Hareketli Ortalamalar & Trend

| Çizgi | Renk/Stil | Neyi Gösterir |
|---|---|---|
| **SMA Kısa** | Turuncu | Kısa vadeli trend ortalaması (varsayılan 20 bar). Fiyat altındaysa zayıflık, üstündeyse güç |
| **SMA Uzun** | Cyan | Orta vadeli ortalama (varsayılan 200 bar). Trend yönü anchor'ı |
| **KAMA** | Mor | Kaufman Adaptif MA — volatiliteye göre hız değiştirir. Yatayda düz, trend başlayınca hızlanır |
| **EMA 200** | Sarı, noktalı | Uzun vadeli trend filtresi. Fiyat üstündeyse "boğa piyasası", altındaysa "ayı piyasası" |
| **SuperTrend çizgisi** | Yeşil (boğa) / Kırmızı (ayı) | ATR tabanlı trend takip. Çizginin rengi mevcut rejimi söyler |
| **🔼 SuperTrend AL** | Yeşil kare, beyaz "AL" yazısı | ST rejimi AYI'dan BOĞA'ya geçti — trend değişim sinyali |
| **🔽 SuperTrend SAT** | Kırmızı kare, beyaz "SAT" yazısı | ST rejimi BOĞA'dan AYI'ya geçti |

💡 **İpucu:** SMA kısa > SMA uzun → "altın haç" (golden cross) bağlamı. EMA200 üstünde kalan bir fiyat, SMA ve KAMA'nın da yukarı eğimiyle birleşirse **çok katmanlı trend teyidi** vardır.

---

### 📊 Kanallar & Zarflar

| Element | Renk | Neyi Gösterir |
|---|---|---|
| **LRC Orta** | Beyaz kesikli | Linear Regression Channel — periyoda göre fiyatın istatistiksel orta çizgisi |
| **LRC Üst** | Gri noktalı | Orta + N standart sapma. Fiyat burada = kanalın üst sınırı, olası SAT bölgesi |
| **LRC Alt** | Gri noktalı | Orta - N standart sapma. Fiyat burada = kanalın alt sınırı, olası AL bölgesi |

---

### 📐 Fibonacci Seviyeleri

**Trend yönüne göre dinamik çizilir:**
- **Yükseliş trendinde** (📈 bull retracement): Son swing LOW pivotundan sonraki swing HIGH'a kadar çizilir.
  Seviyeler **destek** olarak görev yapar — fiyat geri çekilince bu seviyelerden tepki bekler.
- **Düşüş trendinde** (📉 bear retracement): Son swing HIGH pivotundan sonraki swing LOW'a kadar çizilir.
  Seviyeler **direnç** olarak görev yapar — fiyat tepki yaparken bu seviyelerden satıcı bekler.
- **Yatay/range piyasada**: Lookback range'inin global high-low'u kullanılır (geleneksel davranış).

Trend tespiti: Son `fib_lookback` barın ilk %25 ortalama fiyatı ile son %25 ortalaması karşılaştırılır.
%0.5'ten büyük fark varsa yön belirlenir.

**Yedi seviye:**

| Seviye | Renk (tipik) | Bull retracement (destek) | Bear retracement (direnç) |
|---|---|---|---|
| **0.0%** | Kırmızı | Swing dibi (pivot) | Swing dibi (hedef) |
| **23.6%** | Turuncu | Hafif geri çekilme — güçlü trendde tepki beklenir | Zayıf direnç |
| **38.2%** | Sarı | Normal correction seviyesi | Orta direnç |
| **50.0%** | Yeşil | Psikolojik seviye (Fib değil ama eklenmiştir) | Psikolojik direnç |
| **61.8%** | Mavi | Altın oran — en önemli destek | Altın oran — en önemli direnç |
| **78.6%** | Mor | Derin geri çekilme — trend zayıflıyor | Trend dönüş yakın |
| **100.0%** | Kırmızı | Swing tepesi | Swing tepesi (pivot) |

💡 Karar matrisinde Fibonacci satırında **trend yönü** (bull/bear/range) ve **en yakın seviye** gösterilir.

---

### 🎯 Yatay Destek / Direnç

Swing pivot tespiti + ATR tabanlı gruplama ile otomatik çiziliyor.

| Görünüm | Anlamı |
|---|---|
| **Yeşil yatay çizgi** | Aktif **destek** (fiyatın altında) |
| **Kırmızı yatay çizgi** | Aktif **direnç** (fiyatın üstünde) |
| **Gri yatay çizgi** | Kırılmış seviye — artık aktif değil, referans için duruyor |

**Kalınlık/stil dokunuş sayısını söyler:**
- **İnce, dash** (— —) → 1 dokunuş (zayıf)
- **Orta, dashdot** (—·—·) → 2 dokunuş (orta)
- **Kalın, solid** (———) → 3+ dokunuş (güçlü)

**🔄 Role-Reversal (Rol Değişimi):**
- Fiyat eski bir direnci kırıp yukarı geçerse → o seviye **destek** rolüne geçer (yeşile döner)
- Fiyat eski bir desteği kırıp aşağı inerse → o seviye **direnç** rolüne geçer (kırmızıya döner)
- Klasik teknik analiz prensibi: "eski direnç yeni destektir"

---

### 📏 Diyagonal Trend Çizgileri (Legend'dan aç/kapa)

Pivot high'ları birleştirince **direnç TL**, pivot low'ları birleştirince **destek TL** oluşur. Legend başlığı "Trend Çizgileri" altında:

| Görünüm | Anlamı |
|---|---|
| **↗ Destek TL (xN)** yeşil | Yükselen trend çizgisi, N dokunuşla doğrulanmış |
| **↘ Direnç TL (xN)** kırmızı | Düşen trend çizgisi, N dokunuşla doğrulanmış |
| **Mavimsi dolgu alan** | Paralel kanal — fiyatın içinde hareket etmesi beklenen koridor |

Dokunuş sayısı (xN) arttıkça çizgi daha kalın çizilir. 5+ dokunuşlu bir trend çizgisinin kırılması çok anlamlıdır.

---

### 🔻 Divergence İşaretleri

| Sembol | Renk | Anlamı |
|---|---|---|
| **🔺 Bullish Div** | Yeşil üçgen (mumun altında) | Fiyat daha düşük dip yaptı **ama** RSI veya MACD daha yüksek dip yaptı → gizli güç, dönüş sinyali |
| **🔻 Bearish Div** | Kırmızı üçgen (mumun üstünde) | Fiyat daha yüksek tepe yaptı **ama** RSI veya MACD daha düşük tepe yaptı → zayıflama, düşüş uyarısı |

💡 Divergence tek başına giriş sinyali değildir — başka teyitlerle birlikte değerlendir.

---

### 📦 Volume Profile (Sağ Panel) & POC

Grafiğin sağında yatay hacim çubukları var. Her çubuk, o fiyat seviyesinde geçmişte **ne kadar hacim** gerçekleştiğini gösterir.

| Element | Renk | Anlamı |
|---|---|---|
| **POC (Point of Control)** | Turuncu kesikli yatay çizgi + etiket | En yüksek hacimli fiyat seviyesi — piyasanın "adil değer"i kabul edilir |
| **Mavi-yeşil tonlu çubuklar** | Yoğunluğa göre renk | Hacim arttıkça daha doygun yeşile kayar |
| **Son fiyat etiketi** | Yeşil (POC üstünde) / Kırmızı (POC altında) | Fiyatın POC'a göre konumu |

**Nasıl yorumlanır?**
- Fiyat POC'un **altında** → piyasa ucuza düşmüş, alıcılar devreye girebilir
- Fiyat POC'un **üstünde** → değerinin üstünde, satış baskısı gelebilir
- **Boş hacim bölgeleri** (az çubuk) = fiyat hızlı geçiyor, güçlü hareket zonu
- **Dolu hacim bölgeleri** = konsolidasyon, güçlü destek/direnç

---

### 💡 Hepsini Birlikte Nasıl Okumalı?

Görsel bir **çoklu-teyit sistemi** olarak tasarlanmış. Tek bir sinyale değil, **birbiriyle örtüşen** sinyallere güvenin:

1. **Büyük resim:** Fiyat EMA200'ün neresinde? Trend mi yatay mı?
2. **Rejim:** SuperTrend ne diyor? Kısa MA uzun MA'nın neresinde?
3. **Seviye:** Fiyat hangi Fib / LRC / S/R seviyesinde?
4. **Momentum:** Mum rengi ne? Cyan/Yeşil mi, Sarı/Kırmızı mı?
5. **Uyarı:** Divergence var mı? Kırılmış seviyeler hangileri?
6. **Hacim:** POC'un neresinde? Volume profile dağılımı nasıl?

Üç veya daha fazla sinyal **aynı yönü gösteriyorsa** konfidans yüksektir. Çelişiyorsa → **bekle**.

> ⚠️ **Not:** Bu rehber sadece grafik elementlerini açıklar. Alt sekmelerdeki göstergelerin (RSI, MACD, ADX vb.) detaylı yorumu her sekmenin kendi "📖 Nasıl Okunur?" bölümündedir.
""")

        # ============================================================
        # ALT GRAFİKLER
        # ============================================================
        tab_bb, tab_adx, tab_ichi, tab_kama, tab_st, tab_stoch, tab_wt, tab_rsi, tab_macd, tab_obv, tab_div = st.tabs([
            "Bollinger Bands", "ADX", "Ichimoku", "KAMA & LRC", "SuperTrend",
            "Stoch RSI", "WaveTrend", "RSI", "MACD", "OBV", "Divergence"])

        # Eski tab1..tab11 değişken isimlerini koru (içerik bloklarını değiştirmemek için)
        tab1  = tab_rsi
        tab2  = tab_macd
        tab3  = tab_adx
        tab4  = tab_obv
        tab5  = tab_stoch
        tab6  = tab_ichi
        tab7  = tab_st
        tab8  = tab_kama
        tab10 = tab_wt
        tab11 = tab_div

        with tab1:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="rgba(0,200,100,0.9)", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,200,100,0.15)"))
            f.add_trace(go.Scatter(x=df.index, y=df["RSI_MA"],
                name=f"RSI MA({rsi_ma_period})", line=dict(color="yellow", width=1.5, dash="dot")))
            f.add_hline(y=p_rsi["rsi_lower"], line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({p_rsi['rsi_lower']})")
            f.add_hline(y=p_rsi["rsi_upper"], line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({p_rsi['rsi_upper']})")
            f.add_hline(y=50, line_dash="dot", line_color="gray")
            bull_div_rsi = df["Div_RSI"] == 1
            bear_div_rsi = df["Div_RSI"] == -1
            if bull_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_rsi], y=df["RSI"][bull_div_rsi],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_rsi], y=df["RSI"][bear_div_rsi],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 RSI Nasıl Okunur?"):
                st.markdown("""
**RSI (Relative Strength Index)** — 0–100 arasında salınan momentum göstergesidir.

| Bölge | Anlam |
|---|---|
| RSI < Aşırı Satım eşiği | 🟢 Aşırı satılmış → potansiyel AL sinyali |
| RSI > Aşırı Alım eşiği | 🔴 Aşırı alınmış → potansiyel SAT sinyali |
| RSI ~ 50 | ⚪ Nötr bölge |

- **RSI MA (sarı noktalı):** RSI'nın hareketli ortalaması. RSI bu çizgiyi yukarı keserse momentum güçleniyor demektir.
- **Bullish Divergence 🔺:** Fiyat düşük dip yaparken RSI yüksek dip yapıyor → güçlü dönüş sinyali.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe yaparken RSI alçak tepe yapıyor → zayıflama uyarısı.
                """)

        with tab2:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="cyan")))
            f.add_trace(go.Scatter(x=df.index, y=df["MACD_S"], name="Sinyal", line=dict(color="orange")))
            hist = df["MACD"] - df["MACD_S"]
            f.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in hist], opacity=0.5))
            bull_div_macd = df["Div_MACD"] == 1
            bear_div_macd = df["Div_MACD"] == -1
            if bull_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_macd], y=df["MACD"][bull_div_macd],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_macd], y=df["MACD"][bear_div_macd],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 MACD Nasıl Okunur?"):
                st.markdown("""
**MACD (Moving Average Convergence Divergence)** — trend yönü ve momentumu ölçer.

| Unsur | Anlam |
|---|---|
| MACD > Sinyal çizgisi | 🟢 Yukarı momentum → AL eğilimi |
| MACD < Sinyal çizgisi | 🔴 Aşağı momentum → SAT eğilimi |
| Histogram yeşil & büyüyor | 🟢 Momentum güçleniyor |
| Histogram kırmızı & büyüyor | 🔴 Momentum zayıflıyor |

- **Sıfır çizgisi geçişi:** MACD sıfırı yukarı kesiyor = güçlü boğa sinyali; aşağı kesiyor = ayı sinyali.
- **Bullish Divergence 🔺:** Fiyat düşük dip, MACD yüksek dip → trend dönüş öncüsü.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe, MACD alçak tepe → zirve uyarısı.
                """)

        with tab3:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["ADX"],      name="ADX", line=dict(color="yellow", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["PLUS_DI"],  name="+DI", line=dict(color="lime", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["MINUS_DI"], name="-DI", line=dict(color="red",  dash="dot")))
            f.add_hline(y=p_adx["adx_threshold"], line_dash="dash", line_color="white",
                annotation_text=f"Trend Eşiği ({p_adx['adx_threshold']})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 ADX Nasıl Okunur?"):
                st.markdown("""
**ADX (Average Directional Index)** — trendin gücünü ölçer (yön değil, sadece güç).

| ADX Değeri | Trend Gücü |
|---|---|
| < 20 | Zayıf / yatay piyasa |
| 20–25 | Trend oluşuyor |
| > 25 | Güçlü trend |
| > 40 | Çok güçlü trend |

- **+DI (yeşil):** Yukarı yönlü hareketin gücü.
- **-DI (kırmızı):** Aşağı yönlü hareketin gücü.
- **+DI > -DI ve ADX > eşik:** 🟢 Güçlü yükseliş trendi.
- **-DI > +DI ve ADX > eşik:** 🔴 Güçlü düşüş trendi.
- ADX düşükken verilen sinyaller güvenilmezdir.
                """)

        with tab4:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="dodgerblue")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_short,
                name=f"OBV SMA {obv_short}", line=dict(color="orange", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_long,
                name=f"OBV SMA {obv_long}", line=dict(color="cyan", dash="dot")))
            bull_div_obv = df["Div_OBV"] == 1
            bear_div_obv = df["Div_OBV"] == -1
            if bull_div_obv.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_obv], y=df["OBV"][bull_div_obv],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_obv.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_obv], y=df["OBV"][bear_div_obv],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 OBV Nasıl Okunur?"):
                st.markdown("""
**OBV (On-Balance Volume)** — hacim akışını kümülatif olarak izler; fiyat hareketini önceden haber verebilir.

| Durum | Anlam |
|---|---|
| OBV yükseliyor, fiyat yükseliyor | 🟢 Trend onaylanıyor |
| OBV yükseliyor, fiyat düşüyor | 🟢 Gizli birikim → potansiyel yukarı kırılım |
| OBV düşüyor, fiyat yükseliyor | 🔴 Dağıtım var → zayıflama uyarısı |
| OBV düşüyor, fiyat düşüyor | 🔴 Trend onaylanıyor |

- **Kısa SMA (turuncu) > Uzun SMA (cyan):** OBV momentumu pozitif → AL eğilimi.
- **Kısa SMA < Uzun SMA:** OBV momentumu negatif → SAT eğilimi.
- **Bullish Divergence 🔺:** Fiyat yeni dip yaparken OBV yapmıyor → satıcı tükenmesi, dönüş habercisi.
- **Bearish Divergence 🔻:** Fiyat yeni tepe yaparken OBV yapmıyor → alıcı yorgunluğu, zayıflama.
- OBV'nin mutlak değeri değil, eğimi önemlidir.
                """)

        with tab5:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"], name="%K", line=dict(color="magenta")))
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"], name="%D", line=dict(color="orange", dash="dot")))
            f.add_hline(y=stoch_lower, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({stoch_lower})")
            f.add_hline(y=stoch_upper, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({stoch_upper})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Stochastic RSI Nasıl Okunur?"):
                st.markdown("""
**Stochastic RSI** — RSI'ya uygulanan Stochastic göstergesidir. RSI'dan daha hassas ve hızlıdır.

| Bölge | Anlam |
|---|---|
| %K < Aşırı Satım eşiği | 🟢 Aşırı satılmış → AL bölgesi |
| %K > Aşırı Alım eşiği | 🔴 Aşırı alınmış → SAT bölgesi |

- **%K (mor):** Hızlı çizgi — anlık sinyal verir.
- **%D (turuncu noktalı):** %K'nın ortalaması — yavaş, daha güvenilir.
- **%K, %D'yi aşırı satım bölgesinde yukarı kesiyor:** 🟢 Güçlü AL sinyali.
- **%K, %D'yi aşırı alım bölgesinde aşağı kesiyor:** 🔴 Güçlü SAT sinyali.
- RSI aşırı bölgelerde değilken Stoch RSI sinyalleri daha az güvenilirdir.
                """)

        with tab6:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index, y=df["Tenkan"], name="Tenkan-sen", line=dict(color="cyan",  width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Kijun"],  name="Kijun-sen",  line=dict(color="red",   width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Chikou"], name="Chikou Span",
                line=dict(color="rgba(120,180,255,0.7)", width=1, dash="dash")))

            # Senkou A ve B çizgileri (görsel referans)
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_A"], name="Senkou A",
                line=dict(color="rgba(0,255,100,0.6)", width=0.5, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_B"], name="Senkou B",
                line=dict(color="rgba(255,80,80,0.6)", width=0.5, dash="dot")))

            # ── Koşullu renkli bulut (Kumo) ──
            # Senkou A > Senkou B → YEŞİL (bullish)
            # Senkou A < Senkou B → KIRMIZI (bearish)
            # Plotly'de koşullu fill için her noktada "max" ve "min" çizip maskelemek gerekiyor
            sa = df["Senkou_A"].values
            sb = df["Senkou_B"].values
            # Bullish maske (A > B)
            sa_bull = np.where(sa >= sb, sa, np.nan)
            sb_bull = np.where(sa >= sb, sb, np.nan)
            # Bearish maske (A < B)
            sa_bear = np.where(sa < sb,  sa, np.nan)
            sb_bear = np.where(sa < sb,  sb, np.nan)

            # Yeşil bulut (bullish)
            f.add_trace(go.Scatter(x=df.index, y=sb_bull, name="Yeşil Bulut (A>B)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            f.add_trace(go.Scatter(x=df.index, y=sa_bull, name="Yeşil Bulut 🟢",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(0,255,100,0.18)", hoverinfo="skip",
                legendgroup="kumo_bull"))
            # Kırmızı bulut (bearish)
            f.add_trace(go.Scatter(x=df.index, y=sb_bear, name="Kırmızı Bulut (A<B)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            f.add_trace(go.Scatter(x=df.index, y=sa_bear, name="Kırmızı Bulut 🔴",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(255,80,80,0.18)", hoverinfo="skip",
                legendgroup="kumo_bear"))

            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Ichimoku Nasıl Okunur?"):
                st.markdown("""
**Ichimoku Kinko Hyo** — trend yönü, destek/direnç ve momentum'u tek grafikte gösterir.
Goichi Hosoda'nın 1930'larda geliştirdiği klasik 5'li set kullanılır.

| Unsur | Renk | Anlam |
|---|---|---|
| Tenkan-sen | Cyan | Kısa vadeli denge çizgisi (9 bar) |
| Kijun-sen | Kırmızı | Orta vadeli denge çizgisi (26 bar) |
| Senkou Span A | Yeşil | Bulutun üst sınırı |
| Senkou Span B | Kırmızı | Bulutun alt sınırı |
| **Chikou Span** | **Mavi (kesikli)** | **Kapanışın 26 bar geriye kaydırılmış hali — trend teyit çizgisi** |

**Okuma Kuralları:**
- **Fiyat bulutun üstünde:** 🟢 Yükseliş trendi.
- **Fiyat bulutun altında:** 🔴 Düşüş trendi.
- **Fiyat bulut içinde:** ⚪ Konsolidasyon.
- **Tenkan > Kijun:** 🟢 Kısa vadeli momentum pozitif.
- **Yeşil bulut (Span A > Span B):** Boğa piyasası.
- **Kırmızı bulut (Span B > Span A):** Ayı piyasası.
- **Chikou geçmiş fiyatların üstünde:** 🟢 Trend teyitli — bugünkü kapanış 26 bar öncesinden yüksek.
- **Chikou geçmiş fiyatların altında:** 🔴 Trend teyitli — bugünkü kapanış 26 bar öncesinden düşük.

**Sinyal Mantığı (Üçlü Teyit — Hosoda klasiği):**
Sistem AL/SAT üretmek için **üç koşulun birden** sağlanmasını ister:
1. Tenkan-Kijun cross (kısa vade momentum)
2. Fiyat-Bulut pozisyonu (uzun vade trend)
3. Chikou onayı (geçmişle kıyas — Hosoda'nın klasik kullanımı)

Bu konservatif yapı sinyali nadir ama güvenilir kılar. Düşük volatilite dönemlerinde
sinyal yine üretilir ama karar matrisinde "düşük vol" uyarısı görürsünüz — kararı
kullanıcı bağlama göre değerlendirir.

⚠️ **Parametreler:** 9-26-52 değerleri Hosoda'nın orijinal ayarlarıdır ve dünya çapında izlenir.
Schelling noktası etkisiyle bu seviyelerde fiyat tepkisi oluşur. Optimize etmek değil, **sabitlemek** doğrudur.
                """)

        with tab7:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index[bull_st], y=df["SuperTrend"][bull_st],
                name="SuperTrend (Boğa)", mode="lines", line=dict(color="lime", width=2)))
            f.add_trace(go.Scatter(x=df.index[bear_st], y=df["SuperTrend"][bear_st],
                name="SuperTrend (Ayı)", mode="lines", line=dict(color="red", width=2)))
            if st_buy_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_buy_signal], y=df["SuperTrend"][st_buy_signal],
                    name="AL", mode="markers+text",
                    marker=dict(symbol="square", color="#00c853", size=18, line=dict(width=0)),
                    text="AL",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            if st_sell_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_sell_signal], y=df["SuperTrend"][st_sell_signal],
                    name="SAT", mode="markers+text",
                    marker=dict(symbol="square", color="#d50000", size=18, line=dict(width=0)),
                    text="SAT",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 SuperTrend Nasıl Okunur?"):
                st.markdown("""
**SuperTrend** (Olivier Seban, 2008) — ATR tabanlı dinamik destek/direnç + trailing stop göstergesidir.

**İki katmanlı bilgi sağlar:**

**1. Rejim (sürekli):** Çizgi rengi mevcut trendi gösterir, her bar günceldir.

| Durum | Anlam |
|---|---|
| Çizgi yeşil (fiyatın altında) | 🟢 Yükseliş rejimi — çizgi destek seviyesi (trailing stop) |
| Çizgi kırmızı (fiyatın üstünde) | 🔴 Düşüş rejimi — çizgi direnç seviyesi |

**2. Sinyal (event-based):** Yalnızca yön değişiminde (flip) üretilir.

| İşaret | Anlam |
|---|---|
| 🟩 AL kutusu | ⚡ Flip-up: ayıdan boğaya geçiş, **yeni** AL sinyali |
| 🟥 SAT kutusu | ⚡ Flip-down: boğadan ayıya geçiş, **yeni** SAT sinyali |

Trend devam ettiği sürece (flip yok) yeni sinyal üretilmez — bu **doğru** davranıştır.
SuperTrend'in özgün gücü "bant kırılımıyla yön değişir" mantığında saklıdır;
her bar AL/SAT üretmek bu güçten faydalanmaz.

**Trailing Stop kullanımı:**
- Boğa modunda → SuperTrend çizgisi = stop loss seviyesi
- Fiyat çizginin altına düşerse → otomatik flip + çıkış sinyali
- Karar matrisinde "Çizgi: X (fiyatın %Y altında)" şeklinde görürsünüz

**Parametreler:**
- **ATR Periyodu:** Volatilite hesabı penceresi (klasik 10).
- **ATR Çarpanı:** Band genişliği. Yüksek değer → az sinyal, az whipsaw, ama geç giriş/çıkış. (klasik 3.0)

**İpuçları:**
- "Flip yakın" uyarısı (çizgi-fiyat mesafesi <%1): pozisyon kapama hazırlığı yap.
- "Flip'ten X bar" göstergesi: trendin ne kadar olgunlaştığını söyler.
- ADX > eşik ile birlikte kullanım sinyal kalitesini artırır.
                """)

        with tab8:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(color="white", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["KAMA"], name="KAMA", line=dict(color="violet", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"], name="LRC Orta",
                line=dict(color="white", dash="dash", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"], name="LRC Üst",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"], name="LRC Alt",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot"),
                fill="tonexty", fillcolor="rgba(150,150,150,0.07)"))
            f.update_layout(**sub_layout(height=350))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 KAMA & LR Channel Nasıl Okunur?"):
                st.markdown("""
**KAMA (Kaufman Adaptive Moving Average)** — Perry Kaufman 1995. Piyasa koşullarına göre hızını adapte eden akıllı ortalama.

**Felsefe:**
KAMA klasik MA gibi **cross sinyali için değil, eğim için** tasarlanmıştır.
"Fiyat KAMA üstünde" SMA'larla zaten ölçülüyor — KAMA'nın değeri **kendi yönü ve ER kalitesidir**.

| Bileşen | Anlam |
|---|---|
| **KAMA eğimi** | Yukarı = trend yukarı, aşağı = trend aşağı, yatay = beklemede |
| **ER (Efficiency Ratio)** | 0–1 arası "yön etkinliği". 1 = mükemmel trend, 0 = tam gürültü |

**Sinyal Mantığı:**
- KAMA son 3 barda yukarı eğimli **VE** ER ≥ 0.30 → 🟢 AL
- KAMA son 3 barda aşağı eğimli **VE** ER ≥ 0.30 → 🔴 SAT
- ER < 0.30 → ⚠️ Yatay/gürültü, sinyal sıfırlanır

**ER yorumu:**
| ER | Anlam |
|---|---|
| > 0.50 | 🔥 Güçlü trend — sinyal çok güvenilir |
| 0.30 – 0.50 | ⚖️ Orta momentum — sinyal geçerli |
| < 0.30 | ⚠️ Yatay/gürültü — KAMA susar |

**Ek bilgi (bağlamsal):**
- Fiyat-KAMA arası yüzde uzaklık trend gücünü gösterir ama tek başına sinyal değildir.
- ATR filtresi yerine ER filtresi kullanılır: ATR mutlak volatiliteyi, ER yön kalitesini ölçer.
  Yüksek ATR + düşük ER = yatay zikzak (KAMA'nın **çıkması gereken** durum).

**LR Channel (Linear Regression Channel)** — Gilbert Raff 1996.

Son N barın kapanışına OLS regresyon uygulanır. Mid çizgisi regresyon tahmini, bantlar rezidüel std ile çizilir.
BB'den farkı: orta çizgi düz değil **eğimlidir** — kanal trendi takip eder.

**Sinyal Mantığı (slope-aware mean reversion):**
- Slope ≥ 0 (yükselen/yatay kanal) **VE** fiyat alt bantta → 🟢 AL (trende uyumlu dip)
- Slope < 0 (alçalan kanal) **VE** fiyat üst bantta → 🔴 SAT (trende uyumlu tepe)
- Trende **ters** mean reversion sinyalleri (yükselen kanalda üst bant dokunma → SAT) silinir.
  Bu whipsaw'ı önler — trend devam ediyorsa sapma sinyal değil, momentum'dur.

**Bağlamsal Bilgiler (karar matrisinde):**

| Bilgi | Anlam |
|---|---|
| **Slope** | Bar başına fiyat değişimi. + → yükselen, − → alçalan, 0 → yatay |
| **R²** | Regresyon kalitesi. Yüksekse veri doğrusal, sinyaller güvenilir |
| **Bant genişliği** | Lokal volatilite ölçüsü. Daralma = squeeze (patlama yakın), genişleme = trend olgunlaşıyor |

**R² yorumu:**
| R² | Anlam |
|---|---|
| > 0.70 | 🔥 Güçlü doğrusal trend — LRC sinyalleri çok güvenilir |
| 0.40 – 0.70 | ⚖️ Orta uyum — sinyaller geçerli |
| < 0.40 | ⚠️ Zayıf uyum — veri zikzaklı, kanal anlamsız |

**LRC vs BB:** LRC'nin orta çizgisi **eğimli** (regresyon), BB'ninki düz (SMA). Trendli piyasada LRC bantları trendle birlikte hareket ettiği için mean reversion sinyalleri **daha doğru** verir. Yatay piyasada ikisi yakınsar.
                """)

        with tab_bb:
            f = go.Figure()
            # Üst-alt bant arası şeffaf mavi dolgu (önce alt bandı ekleyip,
            # üst bandı "tonexty" ile ona doldurmak gerekiyor)
            f.add_trace(go.Scatter(x=df.index, y=df["Low_BB"], name="Alt Band",
                line=dict(color="lime", width=1, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["Up"], name="Üst Band",
                line=dict(color="red", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(80,140,255,0.08)"))
            f.add_trace(go.Scatter(x=df.index, y=df["Mid"], name="Orta (SMA)",
                line=dict(color="gold", width=1.5, dash="dash")))
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="white", width=1.5)))

            # Üst/alt kırmalar
            bb_break_up   = close > df["Up"]
            bb_break_down = close < df["Low_BB"]
            if bb_break_up.any():
                f.add_trace(go.Scatter(x=df.index[bb_break_up], y=close[bb_break_up],
                    name="Aşırı Alım", mode="markers",
                    marker=dict(color="red", size=7, symbol="circle")))
            if bb_break_down.any():
                f.add_trace(go.Scatter(x=df.index[bb_break_down], y=close[bb_break_down],
                    name="Aşırı Satım", mode="markers",
                    marker=dict(color="lime", size=7, symbol="circle")))

            # Squeeze: bant genişliği son 60 barın p25'inin altındaysa
            bb_width = (df["Up"] - df["Low_BB"]) / df["Mid"]
            if len(bb_width.dropna()) >= 60:
                _wnd = bb_width.rolling(60, min_periods=20)
                _p25 = _wnd.quantile(0.25)
                squeeze_mask = (bb_width <= _p25) & bb_width.notna()
                if squeeze_mask.any():
                    f.add_trace(go.Scatter(
                        x=df.index[squeeze_mask], y=df["Mid"][squeeze_mask],
                        name="Squeeze (sıkışma)", mode="markers",
                        marker=dict(color="orange", size=4, symbol="diamond"),
                        opacity=0.7))

            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)

            # Anlık değerler (alt yazı)
            _bb_last_w   = float(bb_width.iloc[-1]) if not bb_width.empty else float("nan")
            _bb_w_med    = float(bb_width.tail(60).median()) if len(bb_width.dropna()) >= 20 else float("nan")
            _bb_pos_text = ""
            _lc = float(close.iloc[-1])
            _lu = float(df["Up"].iloc[-1]) if not df["Up"].empty else float("nan")
            _ll = float(df["Low_BB"].iloc[-1]) if not df["Low_BB"].empty else float("nan")
            _lm = float(df["Mid"].iloc[-1]) if not df["Mid"].empty else float("nan")
            if not (np.isnan(_lu) or np.isnan(_ll) or np.isnan(_lm)):
                if _lc > _lu:
                    _bb_pos_text = f"🔴 Üst bandın **üstünde** ({_lc:.2f} > {_lu:.2f}) — aşırı alım"
                elif _lc < _ll:
                    _bb_pos_text = f"🟢 Alt bandın **altında** ({_lc:.2f} < {_ll:.2f}) — aşırı satım"
                else:
                    _pct_b = (_lc - _ll) / (_lu - _ll) * 100 if (_lu - _ll) > 0 else 50.0
                    _bb_pos_text = f"⚪ Bant **içinde** (%{_pct_b:.0f} pozisyon, orta: {_lm:.2f})"

            _bb_w_pct = (_bb_last_w / _bb_w_med * 100 - 100) if _bb_w_med else 0.0
            _bb_w_label = (
                f"Bant Genişliği: %{_bb_last_w*100:.2f} "
                f"({'+' if _bb_w_pct >= 0 else ''}{_bb_w_pct:.1f}% medyana göre)"
            )
            st.caption(f"{_bb_pos_text} · {_bb_w_label}")

            with st.expander("📖 Bollinger Bands Nasıl Okunur?"):
                st.markdown(f"""
**Bollinger Bands** — fiyatın etrafına çizilen istatistiksel zarftır. Orta çizgi {bb_period} bar SMA, üst/alt bantlar bu ortalamadan **±{bb_std}σ** uzaklıkta. Volatilite ölçer; aynı zamanda mean-reversion ve breakout sinyali verir.

**Bantların yapısı**

| Unsur | Anlam |
|---|---|
| 🟡 Orta çizgi (sarı kesikli) | {bb_period} barlık SMA — fiyatın "denge" noktası |
| 🔴 Üst band (kırmızı kesikli) | SMA + {bb_std}σ — istatistiksel olarak yüksek seviye |
| 🟢 Alt band (yeşil kesikli) | SMA − {bb_std}σ — istatistiksel olarak düşük seviye |
| 🔵 Mavi dolgu | Bantlar arası alan — "normal" hareket aralığı (~%{int((1 - 2*(1 - 0.9772)) * 100) if bb_std == 2.0 else 95}) |

**Sinyal okuma**

| Durum | Yorum |
|---|---|
| 🔴 Fiyat üst bandın üstünde | **Aşırı alım** — istatistiksel olarak nadir bölge. Range piyasasında SAT sinyali; trend piyasasında "trend güçlü" anlamına gelir, hemen satma |
| 🟢 Fiyat alt bandın altında | **Aşırı satım** — Range piyasasında AL sinyali; düşüş trendinde "trend güçlü" |
| ⚪ Fiyat bant içinde | Normal seyir — orta çizgiye yakınlık denge halini gösterir |
| 🟠 Squeeze (turuncu elmaslar) | **Sıkışma** — bant genişliği son 60 barın en düşük %25'inde. Volatilite çökmüş, **breakout yakın** olabilir (yön belirsiz) |

**İki kullanım modu**

- **Mean Reversion (geri dönüş):** Range piyasada üst/alt band kırmaları ters yöne dönüş sinyali. Çoğu zaman geçerli.
- **Squeeze + Breakout:** Bant daralırken bir kırılım gelirse, **trend başlangıcı**. Squeeze sonrası ilk büyük bant dışı hareket güçlü sinyaldir.

⚠️ **Not:** Bollinger tek başına yön söylemez. ADX (trend gücü) ve hacim ile birlikte yorumlanmalı. Güçlü trendde fiyat üst banda yapışıp ilerleyebilir — bu durumda "aşırı alım" yanıltıcıdır.
                """)

        with tab10:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["WT1"], name="WT1",
                line=dict(color="cyan", width=1.5)))
            f.add_trace(go.Scatter(x=df.index, y=df["WT2"], name="WT2",
                line=dict(color="orange", width=1.5, dash="dot")))
            wt_hist = df["WT1"] - df["WT2"]
            f.add_trace(go.Bar(x=df.index, y=wt_hist, name="WT Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in wt_hist], opacity=0.4))
            f.add_hline(y=wt_ob, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({wt_ob})")
            f.add_hline(y=wt_os, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({wt_os})")
            f.add_hline(y=0, line_dash="dot", line_color="gray")
            wt_buy  = df["Sig_WaveTrend"] == 1
            wt_sell = df["Sig_WaveTrend"] == -1
            if wt_buy.any():
                f.add_trace(go.Scatter(x=df.index[wt_buy], y=df["WT1"][wt_buy],
                    name="AL", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if wt_sell.any():
                f.add_trace(go.Scatter(x=df.index[wt_sell], y=df["WT1"][wt_sell],
                    name="SAT", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout(height=300))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 WaveTrend Nasıl Okunur?"):
                st.markdown("""
**WaveTrend (WT_CROSS_LB)** — momentum ve aşırı bölge tespiti için kullanılan osilatördür.

| Unsur | Anlam |
|---|---|
| WT1 (cyan) | Hızlı sinyal çizgisi |
| WT2 (turuncu noktalı) | Yavaş sinyal çizgisi |

- **WT1, WT2'yi aşırı satım bölgesinde yukarı kesiyor 🔺:** Güçlü AL sinyali.
- **WT1, WT2'yi aşırı alım bölgesinde aşağı kesiyor 🔻:** Güçlü SAT sinyali.
                """)

        with tab11:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="red", width=1.5)))
            bull_div_r = df["Div_RSI"]  == 1
            bear_div_r = df["Div_RSI"]  == -1
            bull_div_m = df["Div_MACD"] == 1
            bear_div_m = df["Div_MACD"] == -1
            bull_div_o = df["Div_OBV"]  == 1
            bear_div_o = df["Div_OBV"]  == -1
            if bull_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_r], y=close[bull_div_r],
                    name="RSI Bullish Div", mode="markers",
                    marker=dict(color="lime", size=12, symbol="triangle-up")))
            if bear_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_r], y=close[bear_div_r],
                    name="RSI Bearish Div", mode="markers",
                    marker=dict(color="red", size=12, symbol="triangle-down")))
            if bull_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_m], y=close[bull_div_m],
                    name="MACD Bullish Div", mode="markers",
                    marker=dict(color="aquamarine", size=10, symbol="diamond")))
            if bear_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_m], y=close[bear_div_m],
                    name="MACD Bearish Div", mode="markers",
                    marker=dict(color="salmon", size=10, symbol="diamond")))
            if bull_div_o.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_o], y=close[bull_div_o],
                    name="OBV Bullish Div", mode="markers",
                    marker=dict(color="gold", size=10, symbol="star")))
            if bear_div_o.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_o], y=close[bear_div_o],
                    name="OBV Bearish Div", mode="markers",
                    marker=dict(color="orange", size=10, symbol="star")))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False,
                title_text="Divergence Noktaları (Fiyat Grafiği Üzerinde)")
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Divergence Nasıl Okunur?"):
                st.markdown("""
**Divergence (Uyumsuzluk)** — fiyat hareketi ile indikatör arasındaki zıtlık; trend dönüşünün erken habercisidir.

| Tür | Fiyat | İndikatör | Anlam |
|---|---|---|---|
| Bullish Div 🔺 | Düşük dip | Yüksek dip | 🟢 Satış baskısı azalıyor → yukarı dönüş olabilir |
| Bearish Div 🔻 | Yüksek tepe | Düşük tepe | 🔴 Alış gücü zayıflıyor → aşağı dönüş olabilir |
                """)

        # ============================================================
        # KARAR TABLOSU
        # ============================================================
        last       = df.iloc[-1]
        last_close = safe_scalar(last["Close"])
        last_ath   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False

        # ── Son bar indikatör değerleri (hem Kombine Skor hem Teknik Rapor kullanır) ──
        r_close    = safe_scalar(last["Close"])
        r_kama     = safe_scalar(last["KAMA"])
        r_adx      = safe_scalar(last["ADX"])
        r_pdi      = safe_scalar(last["PLUS_DI"])
        r_mdi      = safe_scalar(last["MINUS_DI"])
        r_macd     = safe_scalar(last["MACD"])
        r_macds    = safe_scalar(last["MACD_S"])
        r_rsi      = safe_scalar(last["RSI"])
        r_stk      = safe_scalar(last["StochRSI_K"])
        r_std      = safe_scalar(last["ST_Direction"])
        r_lrc_sig  = safe_scalar(last["Sig_LRC"])
        r_lrc_mid  = safe_scalar(last["LRC_Mid"])
        r_lrc_up   = safe_scalar(last["LRC_Upper"])
        r_lrc_lo   = safe_scalar(last["LRC_Lower"])
        r_vwap     = safe_scalar(last["VWAP"])     if is_intraday else np.nan
        r_vwap_sig = safe_scalar(last["Sig_VWAP"]) if is_intraday else 0
        r_obv_sig  = safe_scalar(last["Sig_OBV"])
        r_div_rsi  = safe_scalar(last["Div_RSI"])
        r_div_mac  = safe_scalar(last["Div_MACD"])
        r_div_obv  = safe_scalar(last["Div_OBV"])
        r_ichi     = safe_scalar(last["Sig_Ichimoku"])
        r_wt1      = safe_scalar(last["WT1"])
        r_atr_hi   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False
        r_ema200   = safe_scalar(last["EMA200"])

        # ── ADAPTİF ADX EŞİĞİ ──────────────────────────────────────
        # Volatiliteye göre ADX eşiğini otomatik ayarla.
        # Yüksek volatilitede (gürültülü) trend için daha yüksek eşik iste;
        # düşük volatilitede ise daha düşük eşik yeterli.
        # Kullanıcının manuel eşiği baz alınır, üzerine volatilite düzeltmesi uygulanır.
        r_atr      = safe_scalar(last["ATR"])
        r_atr_ma   = safe_scalar(atr_ma.iloc[-1]) if len(atr_ma) else np.nan
        if not (np.isnan(r_atr) or np.isnan(r_atr_ma)) and r_atr_ma > 0:
            atr_ratio = r_atr / r_atr_ma
        else:
            atr_ratio = 1.0

        # Volatilite düzeltmesi: ATR oranı ±%20'yi aşarsa ±5 puan oynat
        if atr_ratio > 1.2:
            adx_threshold_adaptive = min(adx_threshold + 5, 40)
            adx_regime_note = f"Yüksek vol. (ATR×{atr_ratio:.2f}) → eşik +5"
        elif atr_ratio < 0.8:
            adx_threshold_adaptive = max(adx_threshold - 5, 15)
            adx_regime_note = f"Düşük vol. (ATR×{atr_ratio:.2f}) → eşik -5"
        else:
            adx_threshold_adaptive = adx_threshold
            adx_regime_note = f"Normal vol. (ATR×{atr_ratio:.2f}) → eşik değişmedi"

        if fib_levels:
            r_fib_closest = min(fib_levels.items(), key=lambda x: abs(x[1] - r_close))
        else:
            r_fib_closest = ("N/A", r_close)

        res        = []

        def trend_dec(raw_dec, atr_ok):
            # E sürümü: ATR artık kararı override etmiyor. Düşük volatilite
            # bağlamsal bilgi olarak ayrı (ATR satırında ve Ichimoku gibi
            # bazı satırlarda) gösteriliyor. Karar olduğu gibi geçer.
            return raw_dec

        # ── Hiyerarşi satırı: SMA/EMA/KAMA/Fiyat sıralaması ──
        # Kullanıcıya "her şey nerede" tek bakışta göstersin (bullish/bearish hizalama)
        # A: yön okları, B: golden/death cross uyarısı, D: ADX bağlam, G: yakınlık uyarısı
        hiyerarsi_items = []  # (name, value, slope_arrow, near_price_warn)
        lss_h = safe_scalar(last["SMA_SHORT"])
        lsl_h = safe_scalar(last["SMA_LONG"])
        lk_h  = safe_scalar(last["KAMA"])
        le_h  = safe_scalar(last["EMA200"])
        ls200 = safe_scalar(last["SMA200"]) if "SMA200" in df.columns else np.nan

        # A) Yön oku helper: son 3 barlık fark işareti (↑ ↓ →)
        def _slope_arrow(series_name):
            if series_name not in df.columns or len(df) < 4:
                return ""
            s = df[series_name]
            cur  = safe_scalar(s.iloc[-1])
            prev = safe_scalar(s.iloc[-4])
            if np.isnan(cur) or np.isnan(prev) or prev == 0:
                return ""
            change_pct = (cur - prev) / prev * 100
            if change_pct > 0.05:    return "↑"
            if change_pct < -0.05:   return "↓"
            return "→"

        # G) Fiyat-ortalama yakınlık eşiği: %0.5
        near_price_pct = 0.005
        def _near_price(val):
            return abs(val - last_close) / last_close < near_price_pct if last_close > 0 else False

        if not np.isnan(lss_h):
            hiyerarsi_items.append((f"SMA{p_sma['sma_s']}",  lss_h, _slope_arrow("SMA_SHORT"), _near_price(lss_h)))
        hiyerarsi_items.append(("Fiyat", last_close, "", False))
        if not np.isnan(lsl_h):
            hiyerarsi_items.append((f"SMA{p_sma['sma_l']}",  lsl_h, _slope_arrow("SMA_LONG"),  _near_price(lsl_h)))
        if not np.isnan(lk_h):
            hiyerarsi_items.append(("KAMA",                  lk_h,  _slope_arrow("KAMA"),      _near_price(lk_h)))
        if not np.isnan(ls200):
            hiyerarsi_items.append(("SMA200",                ls200, _slope_arrow("SMA200"),    _near_price(ls200)))
        if not np.isnan(le_h):
            hiyerarsi_items.append(("EMA200",                le_h,  _slope_arrow("EMA200"),    _near_price(le_h)))

        # Değere göre büyükten küçüğe sırala
        hiyerarsi_items.sort(key=lambda x: x[1], reverse=True)

        # Format: "Fiyat (4639) > SMA10 (4625) ↑ ⚠️ > KAMA (4598) ↑ > ..."
        def _fmt(item):
            name, val, arrow, near = item
            txt = f"**{name}** ({val:.2f})" if name == "Fiyat" else f"{name} ({val:.2f})"
            if arrow:  txt += f" {arrow}"
            if near:   txt += " ⚠️"
            return txt
        hiyerarsi_str = " > ".join(_fmt(it) for it in hiyerarsi_items)

        # Tüm ortalamalar fiyatın altında → bullish hizalama (trend yukarı)
        # Tüm ortalamalar fiyatın üstünde → bearish hizalama (trend aşağı)
        fiyat_idx = next((i for i, it in enumerate(hiyerarsi_items) if it[0] == "Fiyat"), -1)
        total     = len(hiyerarsi_items) - 1  # fiyat hariç
        if fiyat_idx == 0:
            hiz_desc = "🟢 Güçlü Bullish hizalama"
        elif fiyat_idx == len(hiyerarsi_items) - 1:
            hiz_desc = "🔴 Güçlü Bearish hizalama"
        elif fiyat_idx <= total / 3:
            hiz_desc = "🟢 Zayıf Bullish hizalama"
        elif fiyat_idx >= 2 * total / 3:
            hiz_desc = "🔴 Zayıf Bearish hizalama"
        else:
            hiz_desc = "⚪ Karışık / geçiş"

        # D) ADX bağlam — hizalama gerçek mi yoksa yatay piyasada tesadüf mü?
        adx_val = safe_scalar(last["ADX"])
        if not np.isnan(adx_val):
            if adx_val > adx_threshold:
                adx_note = f"ADX: {adx_val:.0f} (trend güçlü ✅)"
            elif adx_val < max(adx_threshold - 5, 15):
                adx_note = f"ADX: {adx_val:.0f} (trend zayıf — hizalama yanıltıcı olabilir ⚠️)"
            else:
                adx_note = f"ADX: {adx_val:.0f} (geçiş rejimi)"
            hiz_desc += f" | {adx_note}"

        # G) En yakın ortalama uyarısı (kırılım riski) — desc satırına eklenir
        near_warns = [it for it in hiyerarsi_items if it[3] and it[0] != "Fiyat"]
        if near_warns:
            closest = min(near_warns, key=lambda it: abs(it[1] - last_close))
            dist_pct = abs(closest[1] - last_close) / last_close * 100
            hiz_desc += f" | ⚠️ Fiyat-{closest[0]} mesafesi %{dist_pct:.2f} (kırılım riski)"

        # B) Golden/Death Cross yakın mı? Hareketli ortalamalar arasında %0.5 altı mesafe
        # ve aralarında uygun eğim ilişkisi varsa cross yakındır
        cross_threshold = 0.005
        cross_alerts = []
        ma_pairs = []
        if not np.isnan(lss_h) and not np.isnan(lsl_h):
            ma_pairs.append((f"SMA{p_sma['sma_s']}", lss_h, "SMA_SHORT",
                             f"SMA{p_sma['sma_l']}", lsl_h, "SMA_LONG"))
        if not np.isnan(le_h) and not np.isnan(ls200):
            ma_pairs.append(("EMA200", le_h, "EMA200", "SMA200", ls200, "SMA200"))
        if not np.isnan(lk_h) and not np.isnan(lsl_h):
            ma_pairs.append(("KAMA", lk_h, "KAMA",
                             f"SMA{p_sma['sma_l']}", lsl_h, "SMA_LONG"))

        for short_name, short_val, short_col, long_name, long_val, long_col in ma_pairs:
            if long_val == 0:
                continue
            dist = abs(short_val - long_val) / long_val
            if dist > cross_threshold:
                continue
            # Eğimleri al, yaklaşma yönünü tespit et
            short_arrow = _slope_arrow(short_col)
            long_arrow  = _slope_arrow(long_col)
            # Golden cross: kısa MA, uzun MA'nın altında ama yukarı eğimli
            if short_val < long_val and short_arrow == "↑":
                cross_alerts.append(
                    f"🎯 Golden Cross yaklaşıyor: {short_name} ↔ {long_name} "
                    f"mesafesi %{dist*100:.2f}")
            # Death cross: kısa MA, uzun MA'nın üstünde ama aşağı eğimli
            elif short_val > long_val and short_arrow == "↓":
                cross_alerts.append(
                    f"💀 Death Cross yaklaşıyor: {short_name} ↔ {long_name} "
                    f"mesafesi %{dist*100:.2f}")

        # Hiyerarşi tablo yerine başlık altında markdown olarak gösterilecek
        _hiyerarsi_md = hiyerarsi_str
        _hiz_desc_md  = hiz_desc
        _cross_alert_md = "  \n".join(cross_alerts) if cross_alerts else ""
        # ──────────────────────────────────────────────────────────

        lss = safe_scalar(last["SMA_SHORT"])
        lsl = safe_scalar(last["SMA_LONG"])
        if not (np.isnan(lss) or np.isnan(lsl) or np.isnan(last_close)):
            if lss > lsl and last_close > lss:
                _dec, _why = "AL", "Hiyerarşi: Fiyat > SMA_kısa > SMA_uzun."
            elif lss < lsl and last_close < lss:
                _dec, _why = "SAT", "Hiyerarşi: Fiyat < SMA_kısa < SMA_uzun."
            else:
                _dec, _why = "TUT", "Hiyerarşi çelişkili — fiyat kısa MA'nın yanlış tarafında."
            res.append([trend_dec(_dec, last_ath),
                        f"SMA ({p_sma['sma_s']}/{p_sma['sma_l']})", _why])
        else:
            res.append(["N/A", "SMA Crossover", "Yetersiz veri."])

        lr = safe_scalar(last["RSI"])
        if not np.isnan(lr):
            dec = "AL" if lr < p_rsi["rsi_lower"] else ("SAT" if lr > p_rsi["rsi_upper"] else "TUT")
            res.append([dec, f"RSI ({p_rsi['rsi_period']}) [{p_rsi['rsi_lower']}/{p_rsi['rsi_upper']}]", f"Seviye: {lr:.1f}"])
        else:
            res.append(["N/A", "RSI", "Yetersiz veri."])

        lup = safe_scalar(last["Up"])
        llb = safe_scalar(last["Low_BB"])
        if not any(np.isnan(v) for v in [last_close, llb, lup]):
            dec = "AL" if last_close < llb else ("SAT" if last_close > lup else "TUT")
            res.append([dec, f"Bollinger Bands (σ={p_bb['bb_std']})", "Fiyatın kanaldaki yeri."])
        else:
            res.append(["N/A", "Bollinger Bands", "Yetersiz veri."])

        lm  = safe_scalar(last["MACD"])
        lms = safe_scalar(last["MACD_S"])
        if not (np.isnan(lm) or np.isnan(lms)):
            macd_hist = lm - lms
            hist_color = "🟢 Yeşil" if macd_hist > 0 else ("🔴 Kırmızı" if macd_hist < 0 else "⚪ Sıfır")
            relation   = "MACD > Signal" if lm > lms else ("MACD < Signal" if lm < lms else "MACD = Signal")
            macd_desc  = f"{relation} | Histogram: {macd_hist:+.4f} ({hist_color})"
            res.append([trend_dec("AL" if lm > lms else "SAT", last_ath),
                        f"MACD ({p_macd['macd_fast']},{p_macd['macd_slow']},{macd_signal})", macd_desc])
        else:
            res.append(["N/A", "MACD", "Yetersiz veri."])

        lo = safe_scalar(last["Sig_OBV"])
        if lo != 0 and not np.isnan(lo):
            # Son bar OBV SMA değerleri
            obv_s_last = safe_scalar(obv_sma_short.iloc[-1]) if len(obv_sma_short) else np.nan
            obv_l_last = safe_scalar(obv_sma_long.iloc[-1])  if len(obv_sma_long)  else np.nan

            if not (np.isnan(obv_s_last) or np.isnan(obv_l_last)):
                diff     = obv_s_last - obv_l_last
                # Sayıyı okunabilir formata çevir (milyon/milyar)
                def _fmt_vol(v):
                    av = abs(v)
                    if av >= 1e9:  return f"{v/1e9:+.2f}B"
                    if av >= 1e6:  return f"{v/1e6:+.2f}M"
                    if av >= 1e3:  return f"{v/1e3:+.2f}K"
                    return f"{v:+.2f}"
                relation = "Kısa SMA > Uzun SMA" if diff > 0 else "Kısa SMA < Uzun SMA"
                status   = "Birikim ✅" if lo > 0 else "Dağıtım ❌"
                obv_desc = f"{relation} | Fark: {_fmt_vol(diff)} ({status})"
            else:
                obv_desc = "Birikim ✅" if lo > 0 else "Dağıtım ❌"
            res.append(["AL" if lo > 0 else "SAT", f"OBV ({obv_short}/{obv_long})", obv_desc])
        else:
            res.append(["N/A", f"OBV ({obv_short}/{obv_long})", "Yetersiz veri."])

        la   = safe_scalar(last["ADX"])
        lpd  = safe_scalar(last["PLUS_DI"])
        lmd2 = safe_scalar(last["MINUS_DI"])
        if not np.isnan(la):
            # Adaptif eşiği kullan (volatiliteye göre düzeltilmiş)
            adx_eff_thresh = adx_threshold_adaptive
            # DI+/DI- farkı trend yönünün gücünü gösterir
            if not (np.isnan(lpd) or np.isnan(lmd2)):
                di_diff  = lpd - lmd2
                di_info  = f"| +DI: {lpd:.1f} / -DI: {lmd2:.1f} ({'↑' if di_diff > 0 else '↓'} fark: {abs(di_diff):.1f})"
            else:
                di_info = ""
            strength = "Güçlü" if la > adx_eff_thresh else "Zayıf"
            thresh_info = f"eşik: {adx_eff_thresh}"
            if adx_eff_thresh != adx_threshold:
                thresh_info += f" (kullanıcı: {adx_threshold}, adaptif: {adx_eff_thresh})"
            macd_desc = f"ADX: {la:.1f} ({strength}, {thresh_info}) {di_info}"
            if la > adx_eff_thresh:
                res.append([trend_dec("AL" if lpd > lmd2 else "SAT", last_ath), "ADX", macd_desc])
            else:
                res.append(["TUT", "ADX", macd_desc])
        else:
            res.append(["N/A", "ADX", "Yetersiz veri."])

        if is_intraday:
            lv  = safe_scalar(last["VWAP"])
            lvs = safe_scalar(last["Sig_VWAP"])
            if not np.isnan(lv):
                dec = "AL" if lvs == 1 else ("SAT" if lvs == -1 else "TUT")
                res.append([dec, "VWAP", f"VWAP: {lv:.2f} | bant: ±%{vwap_band_pct:.2f}"])
            else:
                res.append(["N/A", "VWAP", "Yetersiz veri."])
        else:
            res.append(["N/A", "VWAP", "Günlük+ periyotta devre dışı."])

        lsk = float(df["StochRSI_K"].iloc[-1])
        lsd = float(df["StochRSI_D"].iloc[-1]) if "StochRSI_D" in df.columns else np.nan
        lss = safe_scalar(last["Sig_StochRSI"])
        if not np.isnan(lsk):
            # Bölge tespiti
            if   lsk < stoch_lower:  bolge = f"Aşırı Satım 🟢 (<{stoch_lower})"
            elif lsk > stoch_upper:  bolge = f"Aşırı Alım 🔴 (>{stoch_upper})"
            else:                    bolge = f"Nötr ⚪ ({stoch_lower}-{stoch_upper})"

            # K/D ilişkisi
            if not np.isnan(lsd):
                if lsk > lsd:   kd_rel = f"K > D ↑ (K:{lsk:.1f} / D:{lsd:.1f})"
                elif lsk < lsd: kd_rel = f"K < D ↓ (K:{lsk:.1f} / D:{lsd:.1f})"
                else:           kd_rel = f"K = D (K:{lsk:.1f} / D:{lsd:.1f})"
            else:
                kd_rel = f"%K: {lsk:.1f}"

            # Teyit durumu: sinyal sadece bölge + K/D uyumluysa oluşur
            if lss == 1:
                teyit = "✅ AL teyidi (aşırı satım + yukarı dönüş)"
                dec = "AL"
            elif lss == -1:
                teyit = "✅ SAT teyidi (aşırı alım + aşağı dönüş)"
                dec = "SAT"
            else:
                # Bölgede ama kesişim teyidi yok
                if lsk < stoch_lower and not np.isnan(lsd) and lsk < lsd:
                    teyit = "⏸ Aşırı satımda ama K < D (dönüş teyidi bekle)"
                elif lsk > stoch_upper and not np.isnan(lsd) and lsk > lsd:
                    teyit = "⏸ Aşırı alımda ama K > D (dönüş teyidi bekle)"
                else:
                    teyit = "Nötr bölgede"
                dec = "TUT"

            stoch_desc = f"{bolge} | {kd_rel} | {teyit}"
            res.append([dec, f"Stoch RSI ({stoch_rsi_period})", stoch_desc])
        else:
            res.append(["N/A", "Stoch RSI", "Yetersiz veri."])

        # ───────── Ichimoku zenginleştirilmiş satır ─────────
        lis = safe_scalar(last["Sig_Ichimoku"])
        l_tenkan = safe_scalar(last["Tenkan"])
        l_kijun  = safe_scalar(last["Kijun"])
        l_seka   = safe_scalar(last["Senkou_A"])
        l_sekb   = safe_scalar(last["Senkou_B"])

        if any(np.isnan([l_tenkan, l_kijun, l_seka, l_sekb])):
            # Senkou'lar 26 bar ileri kaydırıldığı için başlarda NaN olabilir
            res.append(["N/A", "Ichimoku", "Yetersiz veri (Senkou henüz hesaplanmadı)."])
        else:
            # 1) Tenkan-Kijun ilişkisi
            if l_tenkan > l_kijun:
                tk_rel = f"T:{l_tenkan:.1f} > K:{l_kijun:.1f} ↑"
            elif l_tenkan < l_kijun:
                tk_rel = f"T:{l_tenkan:.1f} < K:{l_kijun:.1f} ↓"
            else:
                tk_rel = f"T:{l_tenkan:.1f} = K:{l_kijun:.1f}"

            # 2) Fiyat - Bulut pozisyonu
            cloud_top    = max(l_seka, l_sekb)
            cloud_bottom = min(l_seka, l_sekb)
            if last_close > cloud_top:
                cloud_pos = "Bulut ÜSTÜNDE ✅"
            elif last_close < cloud_bottom:
                cloud_pos = "Bulut ALTINDA ❌"
            else:
                cloud_pos = "Bulut İÇİNDE ⚪"

            # 3) Bulut rengi (Senkou A vs B)
            cloud_color = "Yeşil 🟢" if l_seka > l_sekb else ("Kırmızı 🔴" if l_seka < l_sekb else "Eşit ⚪")

            # 4) Chikou teyidi: bugünün kapanışı, ik bar önceki kapanışla kıyaslama
            #    (Hosoda: Chikou geçmiş fiyatların üstünde → boğa onayı)
            chikou_ref_idx = -ichi_kijun - 1
            if len(close) >= ichi_kijun + 1:
                close_ref = safe_scalar(close.iloc[chikou_ref_idx])
                if not np.isnan(close_ref):
                    if last_close > close_ref:
                        chikou_note = f"Chikou ↑ (kapanış {ichi_kijun} bar öncesinden yüksek)"
                    elif last_close < close_ref:
                        chikou_note = f"Chikou ↓ (kapanış {ichi_kijun} bar öncesinden düşük)"
                    else:
                        chikou_note = "Chikou ="
                else:
                    chikou_note = "Chikou: yetersiz veri"
            else:
                chikou_note = "Chikou: yetersiz veri"

            # 5) Rejim bazlı dinamik uyarı (ADX'e göre)
            #    Adaptif eşik kullanıyoruz — tutarlılık için
            if not np.isnan(la):
                if la > adx_threshold_adaptive:
                    regime_note = f"✅ Trend piyasa — sinyal güvenilir (ADX: {la:.1f})"
                elif la < max(adx_threshold_adaptive - 5, 15):
                    regime_note = f"⚠️ Yatay piyasada aldatıcı (ADX: {la:.1f})"
                else:
                    regime_note = f"⏸ Geçiş rejimi (ADX: {la:.1f})"
            else:
                regime_note = ""

            # Karar + açıklama birleşimi
            desc = f"{tk_rel} | {cloud_pos} | {cloud_color} | {chikou_note}"
            if regime_note:
                desc += f" | {regime_note}"

            if lis == 1:
                # Düşük volatilite uyarısı (artık otomatik silmiyor, sadece bağlamsal not)
                if not last_ath:
                    res.append(["AL", "Ichimoku", desc + " | ⚠️ Düşük vol — sinyal güvenilirliği azalmış olabilir."])
                else:
                    res.append(["AL", "Ichimoku", desc])
            elif lis == -1:
                if not last_ath:
                    res.append(["SAT", "Ichimoku", desc + " | ⚠️ Düşük vol — sinyal güvenilirliği azalmış olabilir."])
                else:
                    res.append(["SAT", "Ichimoku", desc])
            else:
                res.append(["TUT", "Ichimoku", desc])

        lk = safe_scalar(last["KAMA"])
        if not np.isnan(lk):
            # 1) Fiyat-KAMA ilişkisi + yüzde uzaklık (bağlam bilgisi)
            dist_pct_k = (last_close - lk) / lk * 100
            if last_close > lk:
                rel_k = f"Fiyat {last_close:.2f} > KAMA {lk:.2f} (+%{dist_pct_k:.2f})"
            elif last_close < lk:
                rel_k = f"Fiyat {last_close:.2f} < KAMA {lk:.2f} ({dist_pct_k:+.2f}%)"
            else:
                rel_k = f"Fiyat = KAMA ({lk:.2f})"

            # 2) KAMA eğimi (son 3 barlık fark) — ASIL sinyal kaynağı
            slope_window = 3
            if len(df["KAMA"]) >= slope_window + 1:
                kama_slope = lk - safe_scalar(df["KAMA"].iloc[-slope_window - 1])
                if not np.isnan(kama_slope):
                    if kama_slope > 0:
                        slope_desc = f"KAMA ↑ (+{kama_slope:.2f}, {slope_window} bar)"
                    elif kama_slope < 0:
                        slope_desc = f"KAMA ↓ ({kama_slope:.2f}, {slope_window} bar)"
                    else:
                        slope_desc = "KAMA yatay"
                else:
                    slope_desc = "Eğim: yetersiz veri"
            else:
                slope_desc = "Eğim: yetersiz veri"

            # 3) Efficiency Ratio — sinyal kalite filtresi (df'ten direkt)
            er = safe_scalar(last["KAMA_ER"])
            if not np.isnan(er):
                if er > 0.5:
                    er_desc = f"ER: {er:.2f} (güçlü trend 🔥)"
                elif er > 0.30:
                    er_desc = f"ER: {er:.2f} (orta momentum)"
                else:
                    er_desc = f"ER: {er:.2f} (yatay/gürültü ⚠️ sinyal yok)"
            else:
                er_desc = "ER: yetersiz veri"

            kama_desc = f"{slope_desc} | {er_desc} | {rel_k}"

            # Karar: artık eğim + ER tabanlı (Sig_KAMA)
            lks = safe_scalar(last["Sig_KAMA"])
            if lks == 1:
                kama_dec = "AL"
            elif lks == -1:
                kama_dec = "SAT"
            else:
                kama_dec = "TUT"

            res.append([trend_dec(kama_dec, last_ath),
                        f"KAMA ({kama_period},{kama_fast},{kama_slow})", kama_desc])
        else:
            res.append(["N/A", "KAMA", "Yetersiz veri."])

        lst  = safe_scalar(last["SuperTrend"])
        lstd = safe_scalar(last["ST_Direction"])
        if not np.isnan(lst) and not np.isnan(lstd):
            # 1) Yön
            yon = "YUKARI ↑" if lstd == 1 else "AŞAĞI ↓"

            # 2) Çizgi seviyesi ve fiyata uzaklık
            if last_close > 0:
                dist_pct = abs(lst - last_close) / last_close * 100
                if lstd == 1:
                    # Trend yukarı → çizgi altta (destek)
                    uzak_str = f"fiyatın %{dist_pct:.2f} altında (destek)"
                else:
                    # Trend aşağı → çizgi üstte (direnç)
                    uzak_str = f"fiyatın %{dist_pct:.2f} üstünde (direnç)"
                # Flip yakınlığı uyarısı
                if dist_pct < 1.0:
                    uzak_str += " ⚠️ flip yakın"
            else:
                uzak_str = ""

            # 3) Güncel ATR (volatilite bağlamı)
            r_atr_st = safe_scalar(last["ATR"])
            atr_str  = f"ATR: {r_atr_st:.2f}" if not np.isnan(r_atr_st) else ""

            # 4) Flip'ten bu yana bar sayısı (sinyal olgunluğu)
            st_dir_series = df["ST_Direction"].values
            bars_since_flip = 0
            for i in range(len(st_dir_series) - 1, 0, -1):
                if st_dir_series[i] != st_dir_series[i-1]:
                    break
                bars_since_flip += 1
            if bars_since_flip == 0:
                flip_str = "🆕 Yeni flip!"
            elif bars_since_flip < 3:
                flip_str = f"Flip'ten {bars_since_flip} bar (yeni sinyal)"
            else:
                flip_str = f"Flip'ten {bars_since_flip} bar"

            # Birleştir
            parts = [f"Yön: {yon}", f"Çizgi: {lst:.2f} ({uzak_str})"]
            if atr_str:   parts.append(atr_str)
            parts.append(flip_str)
            st_desc = " | ".join(parts)

            # Karar: flip event-bazlı (Sig_SuperTrend)
            # +1 = flip-up (yeni AL), -1 = flip-down (yeni SAT), 0 = trend devam
            lsts = safe_scalar(last["Sig_SuperTrend"])
            if lsts == 1:
                st_dec = "AL"
            elif lsts == -1:
                st_dec = "SAT"
            else:
                # Flip yok — yön bilgisi mevcut (lstd) ama yeni event yok → TUT
                st_dec = "TUT"

            res.append([trend_dec(st_dec, last_ath),
                        f"SuperTrend ({p_st['st_period']}, x{p_st['st_multiplier']})", st_desc])
        else:
            res.append(["N/A", "SuperTrend", "Yetersiz veri."])

        llrc = safe_scalar(last["Sig_LRC"])
        llm  = safe_scalar(last["LRC_Mid"])
        llu  = safe_scalar(last["LRC_Upper"])
        lll  = safe_scalar(last["LRC_Lower"])
        if not np.isnan(llm) and not np.isnan(llu) and not np.isnan(lll):
            # 1) Kanal içi pozisyon
            if last_close > llu:
                pos_lrc = f"Fiyat {last_close:.2f} ÜST kanal üstünde ({llu:.2f}) ❌ aşırı alım"
            elif last_close < lll:
                pos_lrc = f"Fiyat {last_close:.2f} ALT kanal altında ({lll:.2f}) ✅ aşırı satım"
            else:
                # Kanal içinde — orta çizgiye yakınlık
                if last_close > llm:
                    pct_mid = (last_close - llm) / llm * 100
                    pos_lrc = f"Fiyat {last_close:.2f} kanal içinde (orta üstü, +%{pct_mid:.2f})"
                else:
                    pct_mid = (llm - last_close) / llm * 100
                    pos_lrc = f"Fiyat {last_close:.2f} kanal içinde (orta altı, -%{pct_mid:.2f})"

            # 2) Slope yönü (df'ten direkt — sig_lrc içinde hesaplandı)
            slope = safe_scalar(last["LRC_Slope"])
            if not np.isnan(slope):
                slope_pct = slope / llm * 100 if llm > 0 else 0.0
                if slope > 0:
                    slope_desc = f"Slope: +{slope:.3f} ↗ (yükselen, bar başı +%{slope_pct:.3f})"
                elif slope < 0:
                    slope_desc = f"Slope: {slope:.3f} ↘ (alçalan, bar başı %{slope_pct:.3f})"
                else:
                    slope_desc = "Slope: 0 → (yatay)"
            else:
                slope_desc = ""

            # 3) R² — regresyon kalitesi (LRC sinyallerinin güvenilirliği)
            r2 = safe_scalar(last["LRC_R2"])
            if not np.isnan(r2):
                if r2 > 0.7:
                    r2_desc = f"R²: {r2:.2f} (güçlü doğrusal trend 🔥)"
                elif r2 > 0.4:
                    r2_desc = f"R²: {r2:.2f} (orta uyum)"
                else:
                    r2_desc = f"R²: {r2:.2f} (zayıf uyum ⚠️ kanal anlamsız)"
            else:
                r2_desc = ""

            # 4) Bant genişliği (lokal volatilite — normalize)
            bant_width = llu - lll
            if llm > 0:
                bant_pct = bant_width / llm * 100
                bant_desc = f"Bant: ±{bant_width/2:.2f} (kanal genişliği %{bant_pct:.2f})"
            else:
                bant_desc = f"Bant: ±{bant_width/2:.2f}"

            # Birleştir
            parts = [pos_lrc]
            if slope_desc: parts.append(slope_desc)
            if r2_desc:    parts.append(r2_desc)
            parts.append(bant_desc)
            lrc_desc = " | ".join(parts)

            dec = "AL" if llrc == 1 else ("SAT" if llrc == -1 else "TUT")
            res.append([dec, f"LR Channel (σ={p_lrc['lrc_std_mult']})", lrc_desc])
        else:
            res.append(["N/A", "LR Channel", "Yetersiz veri."])

        la2 = safe_scalar(last["ATR"])
        lam = safe_scalar(atr_ma.iloc[-1])
        if not np.isnan(la2) and not np.isnan(lam):
            # 1) Yüzde fark (MA'ya göre)
            if lam > 0:
                pct_diff = (la2 - lam) / lam * 100
                if last_ath:
                    pct_str = f"Yüksek ↑ (%{abs(pct_diff):.1f} üstü MA'dan)"
                else:
                    pct_str = f"Düşük ↓ (%{abs(pct_diff):.1f} altı MA'dan)"
            else:
                pct_str = "Yüksek ↑" if last_ath else "Düşük ↓"

            # 2) Son 5 bar volatilite yönü (artıyor mu azalıyor mu)
            atr_vals = atr_series.values
            if len(atr_vals) >= 6:
                recent       = atr_vals[-5:]
                older        = atr_vals[-6:-1]
                avg_recent   = float(np.nanmean(recent))
                avg_older    = float(np.nanmean(older))
                if np.isfinite(avg_recent) and np.isfinite(avg_older) and avg_older > 0:
                    change_pct = (avg_recent - avg_older) / avg_older * 100
                    if change_pct > 2:
                        trend_str = "Son 5 bar: yükseliyor ↗ (patlama yakın olabilir)"
                    elif change_pct < -2:
                        trend_str = "Son 5 bar: düşüyor ↘ (sıkışma derinleşiyor)"
                    else:
                        trend_str = "Son 5 bar: stabil →"
                else:
                    trend_str = ""
            else:
                trend_str = ""

            parts = [f"Volatilite: {pct_str}", f"ATR: {la2:.2f}", f"MA: {lam:.2f}"]
            if trend_str:
                parts.append(trend_str)
            atr_desc = " | ".join(parts)
            res.append(["BİLGİ", "ATR Filtre", atr_desc])
        else:
            res.append(["N/A", "ATR Filtre", "Yetersiz veri."])

        lwt1    = safe_scalar(last["WT1"])
        lwt2    = safe_scalar(last["WT2"])
        lwt_sig = safe_scalar(last["Sig_WaveTrend"])
        if not np.isnan(lwt1):
            # 1) Bölge tespiti (eşik değerlerini de göster)
            if lwt1 > wt_ob:
                wt_zone = f"Aşırı Alım 🔴 (>{wt_ob})"
            elif lwt1 < wt_os:
                wt_zone = f"Aşırı Satım 🟢 (<{wt_os})"
            else:
                wt_zone = f"Nötr Bölge ({wt_os}/+{wt_ob})"

            # 2) WT1 / WT2 değerleri + ilişki
            if not np.isnan(lwt2):
                if lwt1 > lwt2:
                    kd_rel = f"WT1: {lwt1:.1f} > WT2: {lwt2:.1f} ↑"
                elif lwt1 < lwt2:
                    kd_rel = f"WT1: {lwt1:.1f} < WT2: {lwt2:.1f} ↓"
                else:
                    kd_rel = f"WT1 = WT2 ({lwt1:.1f})"

                # 3) Histogram (WT1 - WT2) + renk
                wt_hist = lwt1 - lwt2
                hist_color = "🟢 Yeşil" if wt_hist > 0 else ("🔴 Kırmızı" if wt_hist < 0 else "⚪ Sıfır")
                hist_str = f"Histogram: {wt_hist:+.2f} ({hist_color})"

                parts = [kd_rel, wt_zone, hist_str]
            else:
                parts = [f"WT1: {lwt1:.1f}", wt_zone]

            wt_desc = " | ".join(parts)
            wt_dec = "AL" if lwt_sig == 1 else ("SAT" if lwt_sig == -1 else "TUT")
            res.append([wt_dec, f"WaveTrend ({p_wt['wt_n1']}/{p_wt['wt_n2']})", wt_desc])
        else:
            res.append(["N/A", "WaveTrend", "Yetersiz veri."])

        # ── YENİ: EMA200 karar satırı ─────────────────────────────
        lema200 = safe_scalar(last["EMA200"])
        if not np.isnan(lema200):
            ema_dec = trend_dec("AL" if last_close > lema200 else "SAT", last_ath)
            res.append([ema_dec, "EMA 200", f"EMA200: {lema200:.2f} | Fiyat {'üstünde ✅' if last_close > lema200 else 'altında ❌'}"])
        else:
            res.append(["N/A", "EMA 200", "Yetersiz veri (min 200 bar gerekli)."])

        # ── YENİ: Fibonacci + Swing S/R Confluence ────────────────
        # Bağımsız iki teknik (swing pivot + Fib retracement) aynı seviyeye
        # işaret ediyorsa "güçlü destek/direnç bandı" — trader için kritik bilgi.
        # Eşik: %0.5 fiyat mesafesi (çok yakın değil, çok uzak değil)
        if swing_levels and fib_levels and last_close > 0:
            confluence_threshold = 0.005   # %0.5
            confluences = []
            for sw in swing_levels:
                if sw.get("broken"):       # kırılmış seviyeler hariç
                    continue
                sw_price = sw["price"]
                for fib_name, fib_price in fib_levels.items():
                    if fib_name in ("0.0%", "100.0%"):   # uçlar zaten swing
                        continue
                    dist = abs(sw_price - fib_price) / last_close
                    if dist <= confluence_threshold:
                        # Ortalama band: iki seviyenin orta noktası
                        band_mid = (sw_price + fib_price) / 2
                        confluences.append({
                            "type":      sw["type"],
                            "swing":     sw_price,
                            "touches":   sw["touches"],
                            "fib_name":  fib_name,
                            "fib_price": fib_price,
                            "band_mid":  band_mid,
                            "dist_to_price": abs(band_mid - last_close) / last_close,
                        })
            # Fiyata yakınlık sırası, en fazla 3 confluence göster
            confluences.sort(key=lambda x: x["dist_to_price"])
            for c in confluences[:3]:
                role = "Güçlü Destek" if c["type"] == "S" else "Güçlü Direnç"
                lo, hi = sorted([c["swing"], c["fib_price"]])
                desc = (f"{lo:.2f}–{hi:.2f} "
                        f"(Swing {c['type']} [{c['touches']}x dokunuş] + Fib {c['fib_name']})")
                res.append(["🎯 Confluence", role, desc])

        # ── YENİ: En yakın S/R seviyesi karar satırı ──────────────
        if swing_levels:
            closest_sr = min(swing_levels, key=lambda x: abs(x["price"] - last_close))
            dist_pct   = abs(closest_sr["price"] - last_close) / last_close * 100
            sr_label   = "Destek" if closest_sr["type"] == "S" else "Direnç"
            res.append(["BİLGİ", "Swing S/R",
                f"En yakın {sr_label}: {closest_sr['price']:.2f} "
                f"(%{dist_pct:.1f} uzakta, {closest_sr['touches']}x dokunuş)"])
        # ──────────────────────────────────────────────────────────

        last_div_rsi  = safe_scalar(last["Div_RSI"])
        last_div_macd = safe_scalar(last["Div_MACD"])
        last_div_obv  = safe_scalar(last["Div_OBV"])
        if last_div_rsi == 1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔺 Bullish Divergence — güçlü dip sinyali olabilir"])
        elif last_div_rsi == -1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔻 Bearish Divergence — zayıflayan momentum"])
        else:
            res.append(["BİLGİ", "Divergence (RSI)", "Aktif divergence yok"])
        if last_div_macd == 1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔺 Bullish Divergence"])
        elif last_div_macd == -1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔻 Bearish Divergence"])
        else:
            res.append(["BİLGİ", "Divergence (MACD)", "Aktif divergence yok"])
        if last_div_obv == 1:
            res.append(["BİLGİ", "Divergence (OBV)", "🔺 Bullish Divergence — fiyat dip yapıyor, hacim desteği zayıflıyor (alıcı tükenmesi)"])
        elif last_div_obv == -1:
            res.append(["BİLGİ", "Divergence (OBV)", "🔻 Bearish Divergence — fiyat tepe yapıyor, hacim desteklemiyor (satıcı tükenmesi)"])
        else:
            res.append(["BİLGİ", "Divergence (OBV)", "Aktif divergence yok"])

        if fib_levels:
            closest_lvl = min(fib_levels.items(), key=lambda x: abs(x[1] - last_close))
            if fib_direction == "up":
                dir_str = "📈 Bull retracement (destek arıyor)"
            elif fib_direction == "down":
                dir_str = "📉 Bear retracement (direnç test ediyor)"
            else:
                dir_str = "↔️ Yatay (range — yön belirsiz)"
            res.append(["BİLGİ", f"Fibonacci ({fib_lookback} bar)",
                        f"En yakın seviye: {closest_lvl[0]} ({closest_lvl[1]:.2f}) | "
                        f"Swing: {fib_low:.2f} — {fib_high:.2f} | {dir_str}"])

        # ============================================================
        # (Kombine Sinyal Skoru kaldırıldı)

        st.subheader("🔍 Algoritmik Detaylar")
        # Hiyerarşi — tablonun üstünde markdown olarak (bold çalışır, tek satır)
        _hier_block = f"**📊 Hiyerarşi:** {_hiyerarsi_md}  \n{_hiz_desc_md}"
        if _cross_alert_md:
            _hier_block += f"  \n{_cross_alert_md}"
        st.markdown(_hier_block)
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])

        def color_map(val):
            if val == "AL":    return "color: #00ff00; font-weight: bold"
            if val == "SAT":   return "color: #ff4b4b; font-weight: bold"
            if val == "N/A":   return "color: #ffaa00; font-weight: bold"
            if val == "BİLGİ": return "color: #00bfff; font-weight: bold"
            if "düşük vol." in str(val): return "color: #808495; font-style: italic"
            return "color: #808495; font-weight: bold"

        st.table(res_df.style.map(color_map, subset=["Karar"]))

        # ============================================================
        # OPTİMİZASYON ÖZET TABLOSU
        # ============================================================
        st.write("---")
        st.subheader("🧬 Walk-Forward Optimizasyon Sonuçları")
        st.caption(f"{n_windows} pencere · expanding window · kriter: Sharpe (yıllıklandırılmış, **out-of-sample**)")

        def opt_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if not np.isfinite(v):   return "color: #888888"
            if v > 0:  return "color: #00ff00"
            if v < 0:  return "color: #ff4b4b"
            return "color: #888888"  # sıfır için gri — koyu arka planda görünür

        def pval_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if not np.isfinite(v): return "color: #888888"
            if v < 0.05:  return "color: #00ff00; font-weight: bold"   # anlamlı
            if v < 0.10:  return "color: #ffcc00"                       # sınırda
            return "color: #aaaaaa"                                     # anlamsız

        def _safe_round(x, nd=2, default=0.0):
            try:
                v = float(x)
                if not np.isfinite(v):
                    return default
                return round(v, nd)
            except (ValueError, TypeError):
                return default

        opt_rows  = []
        for algo_name, grid in PARAM_GRIDS.items():
            p = opt_params.get(algo_name, {})
            s = opt_stats.get(algo_name, {})
            row = {"Algoritma": algo_name}
            param_str            = "  |  ".join(f"{k} = {v}" for k, v in p.items())
            row["Parametreler"]  = param_str
            row["Getiri (%)"]    = _safe_round(s.get("total_ret", 0), 2)
            row["Sharpe (OOS)"]  = _safe_round(s.get("sharpe",    0), 2)
            row["DSR"]           = _safe_round(s.get("dsr", np.nan), 2, default=np.nan)
            row["Trade"]         = int(s.get("n", 0) or 0)
            row["Win Rate (%)"]  = _safe_round(s.get("win_rate",  0), 1)
            sel = s.get("wf_selections", 0); wins = s.get("wf_windows", 0)
            row["Seçim"]         = f"{sel}/{wins}" if wins else "—"
            row["p-değeri"]      = _safe_round(s.get("p_value", np.nan), 4, default=np.nan)
            opt_rows.append(row)

        opt_df     = pd.DataFrame(opt_rows)
        color_cols = [c for c in ["Getiri (%)", "Sharpe (OOS)", "DSR"] if c in opt_df.columns]
        fmt        = {"Getiri (%)": "{:.2f}", "Sharpe (OOS)": "{:.2f}", "DSR": "{:.2f}",
                      "Win Rate (%)": "{:.1f}", "p-değeri": "{:.3f}"}
        fmt        = {k: v for k, v in fmt.items() if k in opt_df.columns}
        styled = opt_df.style.format(fmt, na_rep="—").map(opt_color, subset=color_cols)
        if "p-değeri" in opt_df.columns:
            styled = styled.map(pval_color, subset=["p-değeri"])
        # Seçim k/n: k == n olanları (tüm pencerelerde seçilenler) koyu gri yap
        if "Seçim" in opt_df.columns:
            def _sel_highlight(v):
                try:
                    k, n_ = str(v).split("/")
                    if int(k) == int(n_) and int(n_) > 0:
                        return "background-color: #2a2a2a; font-weight: bold;"
                except Exception:
                    pass
                return ""
            styled = styled.map(_sel_highlight, subset=["Seçim"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "💡 **Sharpe (OOS)**: Yalnız out-of-sample test dilimlerinden yıllıklandırılmış risk ayarlı getiri. "
            "**DSR** (Deflated Sharpe Ratio — Bailey & López de Prado 2014): Multiple testing "
            "cezası çıkarılmış. **DSR > 0** → gerçekten rastgeleden iyi; **DSR ≤ 0** → yüksek Sharpe "
            "muhtemelen şans eseri. **p-değeri** (Stationary Bootstrap — Politis & Romano 1994): "
            "< 0.05 → sinyal istatistiksel olarak anlamlı. **Seçim k/n** → kombonun n expanding "
            "adımında k tanesinde train-kazananı olduğu."
        )

        # ============================================================
        # 📅 EKONOMİK TAKVİM (TradingView)
        # ============================================================
        st.write("---")
        with st.expander("📅 Ekonomik Takvim", expanded=False):
            TV_CAL_URL = "https://economic-calendar.tradingview.com/events"
            TV_CAL_HEADERS = {
                "accept":     "application/json",
                "origin":     "https://www.tradingview.com",
                "referer":    "https://www.tradingview.com/",
                "user-agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/147.0.0.0 Safari/537.36"
                ),
            }
            CAL_COUNTRIES = {
                "TR": "🇹🇷 Türkiye",
                "US": "🇺🇸 ABD",
                "EU": "🇪🇺 Euro Bölgesi",
                "GB": "🇬🇧 İngiltere",
                "DE": "🇩🇪 Almanya",
                "JP": "🇯🇵 Japonya",
                "CN": "🇨🇳 Çin",
            }

            # Başlık çevirileri — sözlükte olmayanlar orijinal İngilizce kalır.
            TR_EVENT_TITLES = {
                # — Türkiye özel —
                "Economic Confidence Index":                       "Ekonomik Güven Endeksi",
                "Unemployment Rate":                               "İşsizlik Oranı",
                "Participation Rate":                              "İşgücüne Katılım Oranı",
                "Tourism Revenues":                                "Turizm Gelirleri",
                "Tourist Arrivals YoY":                            "Turist Sayısı (Y/Y)",
                "MPC Meeting Summary":                             "PPK Toplantı Özeti",
                "MPC Meeting Minutes":                             "PPK Toplantı Tutanakları",
                "Foreign Exchange Reserves":                       "Döviz Rezervleri",
                "Capacity Utilization":                            "Kapasite Kullanım Oranı",
                "Istanbul Chamber of Industry Manufacturing PMI":  "İSO İmalat PMI",
                "TCMB Interest Rate Decision":                     "TCMB Faiz Kararı",
                "Real Sector Confidence":                          "Reel Kesim Güveni",
                "Manufacturing Confidence":                        "İmalat Güveni",
                "Services Confidence":                             "Hizmet Güveni",
                "Retail Confidence":                               "Perakende Güveni",
                "Construction Confidence":                         "İnşaat Güveni",
                # — Resmi tatiller —
                "Labor and Solidarity Day":                        "Emek ve Dayanışma Günü",
                "Republic Day":                                    "Cumhuriyet Bayramı",
                "Victory Day":                                     "Zafer Bayramı",
                "Democracy and National Unity Day":                "Demokrasi ve Milli Birlik Günü",
                "Commemoration of Atatürk, Youth and Sports Day":  "Atatürk'ü Anma, Gençlik ve Spor Bayramı",
                "National Sovereignty and Children's Day":         "Ulusal Egemenlik ve Çocuk Bayramı",
                "Christmas Day":                                   "Noel",
                "New Year's Day":                                  "Yılbaşı",
                # — Genel makro (tüm ülkeler) —
                "Balance of Trade":          "Dış Ticaret Dengesi",
                "Balance of Trade Final":    "Dış Ticaret Dengesi (Nihai)",
                "Balance of Trade Prel":     "Dış Ticaret Dengesi (Öncü)",
                "Imports":                   "İthalat",
                "Imports Final":             "İthalat (Nihai)",
                "Imports Prel":              "İthalat (Öncü)",
                "Exports":                   "İhracat",
                "Exports Final":             "İhracat (Nihai)",
                "Exports Prel":              "İhracat (Öncü)",
                "Trade Balance":             "Ticaret Dengesi",
                "Current Account":           "Cari İşlemler Dengesi",
                "Inflation Rate YoY":        "Enflasyon Oranı (Y/Y)",
                "Inflation Rate MoM":        "Enflasyon Oranı (A/A)",
                "Core Inflation Rate YoY":   "Çekirdek Enflasyon (Y/Y)",
                "Core Inflation Rate MoM":   "Çekirdek Enflasyon (A/A)",
                "CPI":                       "TÜFE",
                "CPI YoY":                   "TÜFE (Y/Y)",
                "CPI MoM":                   "TÜFE (A/A)",
                "Core CPI YoY":              "Çekirdek TÜFE (Y/Y)",
                "Core CPI MoM":              "Çekirdek TÜFE (A/A)",
                "PPI YoY":                   "ÜFE (Y/Y)",
                "PPI MoM":                   "ÜFE (A/A)",
                "GDP Growth Rate YoY":       "GSYH Büyüme (Y/Y)",
                "GDP Growth Rate QoQ":       "GSYH Büyüme (Ç/Ç)",
                "GDP Growth Rate":           "GSYH Büyüme",
                "GDP YoY":                   "GSYH (Y/Y)",
                "Industrial Production YoY": "Sanayi Üretimi (Y/Y)",
                "Industrial Production MoM": "Sanayi Üretimi (A/A)",
                "Retail Sales YoY":          "Perakende Satışlar (Y/Y)",
                "Retail Sales MoM":          "Perakende Satışlar (A/A)",
                "Manufacturing PMI":         "İmalat PMI",
                "Services PMI":              "Hizmet PMI",
                "Composite PMI":             "Bileşik PMI",
                "Consumer Confidence":       "Tüketici Güveni",
                "Business Confidence":       "İş Dünyası Güveni",
                "Budget Balance":            "Bütçe Dengesi",
                "Government Debt to GDP":    "Devlet Borcu/GSYH",
                "House Price Index YoY":     "Konut Fiyat Endeksi (Y/Y)",
                "House Price Index MoM":     "Konut Fiyat Endeksi (A/A)",
                "Interest Rate Decision":    "Faiz Kararı",
                "Deposit Facility Rate":     "Mevduat Faizi",
                # — ABD —
                "Fed Interest Rate Decision":          "Fed Faiz Kararı",
                "FOMC Minutes":                        "FOMC Toplantı Tutanakları",
                "Fed Chair Powell Speech":             "Fed Başkanı Powell Konuşması",
                "Non Farm Payrolls":                   "Tarım Dışı İstihdam",
                "Initial Jobless Claims":              "Haftalık İşsizlik Başvuruları",
                "Continuing Jobless Claims":           "Devam Eden İşsizlik Başvuruları",
                "Average Hourly Earnings MoM":         "Saatlik Kazançlar (A/A)",
                "Average Hourly Earnings YoY":         "Saatlik Kazançlar (Y/Y)",
                "ADP Employment Change":               "ADP İstihdam Değişimi",
                "JOLTs Job Openings":                  "JOLTS Açık İş Sayısı",
                "ISM Manufacturing PMI":               "ISM İmalat PMI",
                "ISM Services PMI":                    "ISM Hizmet PMI",
                "Durable Goods Orders MoM":            "Dayanıklı Mal Siparişleri (A/A)",
                "Factory Orders MoM":                  "Fabrika Siparişleri (A/A)",
                "Building Permits":                    "Yapı İzinleri",
                "Housing Starts":                      "Konut Başlangıçları",
                "Existing Home Sales":                 "Mevcut Konut Satışları",
                "New Home Sales":                      "Yeni Konut Satışları",
                "Pending Home Sales MoM":              "Bekleyen Konut Satışları (A/A)",
                "Crude Oil Inventories":               "Ham Petrol Stokları",
                "PCE Price Index YoY":                 "PCE Fiyat Endeksi (Y/Y)",
                "PCE Price Index MoM":                 "PCE Fiyat Endeksi (A/A)",
                "Core PCE Price Index YoY":            "Çekirdek PCE (Y/Y)",
                "Core PCE Price Index MoM":            "Çekirdek PCE (A/A)",
                "Personal Income MoM":                 "Kişisel Gelir (A/A)",
                "Personal Spending MoM":               "Kişisel Harcama (A/A)",
                "Michigan Consumer Sentiment":         "Michigan Tüketici Güveni",
                "CB Consumer Confidence":              "CB Tüketici Güveni",
                "Chicago PMI":                         "Chicago PMI",
                "Philadelphia Fed Manufacturing Index":"Philadelphia Fed İmalat Endeksi",
                "NY Empire State Manufacturing Index": "NY Empire State İmalat Endeksi",
                # — ECB / BoE / BoJ / PBoC —
                "ECB Interest Rate Decision":  "ECB Faiz Kararı",
                "BoE Interest Rate Decision":  "BoE Faiz Kararı",
                "BoJ Interest Rate Decision":  "BoJ Faiz Kararı",
                "PBoC Loan Prime Rate 1Y":     "PBoC 1Y LPR",
                "PBoC Loan Prime Rate 5Y":     "PBoC 5Y LPR",
            }

            @st.cache_data(ttl=1800, show_spinner=False)
            def _fetch_economic_calendar(country, days, past_days=30):
                """TradingView ekonomik takvimi → (events_list, error_msg).
                past_days: geriye kaç günlük açıklanmış veri çekilsin."""
                _now = datetime.now(timezone.utc)
                _params = {
                    "from":      (_now - timedelta(days=past_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "to":        (_now + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "countries": country,
                }
                try:
                    _r = requests.get(TV_CAL_URL, headers=TV_CAL_HEADERS,
                                      params=_params, timeout=15)
                    _r.raise_for_status()
                    _js = _r.json()
                    if _js.get("status") != "ok":
                        return None, f"API status != ok ({_js.get('status')})"
                    return _js.get("result", []), None
                except requests.exceptions.Timeout:
                    return None, "Zaman aşımı — TradingView yanıt vermiyor."
                except requests.exceptions.ConnectionError as _e:
                    return None, f"Bağlantı hatası: {str(_e)[:200]}"
                except requests.exceptions.HTTPError as _e:
                    return None, f"HTTP {_e.response.status_code}"
                except (ValueError, KeyError) as _e:
                    return None, f"Yanıt ayrıştırılamadı ({type(_e).__name__})"
                except Exception as _e:
                    return None, f"{type(_e).__name__}: {str(_e)[:200]}"

            _cc1, _cc2 = st.columns([1, 1])
            with _cc1:
                _cal_country = st.selectbox(
                    "Ülke",
                    options=list(CAL_COUNTRIES.keys()),
                    format_func=lambda k: CAL_COUNTRIES[k],
                    index=0, key="cal_country",
                )
            with _cc2:
                _cal_days = st.slider(
                    "Önümüzdeki gün sayısı",
                    min_value=1, max_value=30, value=7, step=1,
                    key="cal_days",
                )

            _cal_events, _cal_err = _fetch_economic_calendar(_cal_country, _cal_days)

            if _cal_err:
                st.error(f"❌ Takvim çekilemedi — {_cal_err}")
            elif not _cal_events:
                st.info("Bu aralıkta olay bulunamadı.")
            else:
                _TRT = timezone(timedelta(hours=3))
                _IMP_MAP = {1: "YÜK", 0: "ORT", -1: "DÜŞ"}

                # Geçmiş (actual dolu, son 5) + gelecek olarak ayır
                _now_utc = datetime.now(timezone.utc)
                _past, _future = [], []
                for _ev in _cal_events:
                    _d_raw = _ev.get("date", "")
                    try:
                        _edt = datetime.fromisoformat(_d_raw.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                    if _edt < _now_utc:
                        if _ev.get("actual") is not None:
                            _past.append(_ev)
                    else:
                        _future.append(_ev)

                _past.sort(key=lambda x: x.get("date", ""), reverse=True)
                _past = list(reversed(_past[:5]))  # son 5, kronolojik
                _cal_events_view = _past + _future

                _cal_rows = []
                for _e in _cal_events_view:
                    _d_str = _e.get("date", "")
                    try:
                        _dt = datetime.fromisoformat(_d_str.replace("Z", "+00:00"))
                        _dt_str = _dt.astimezone(_TRT).strftime("%Y-%m-%d %H:%M")
                    except (ValueError, TypeError):
                        _dt_str = _d_str[:16].replace("T", " ")

                    _unit = _e.get("unit") or ""
                    def _f(v, u=_unit):
                        return f"{v}{u}" if v is not None else "—"

                    _cal_rows.append({
                        "Tarih (TRT)": _dt_str,
                        "Önem":        _IMP_MAP.get(_e.get("importance"), "—"),
                        "Önceki":      _f(_e.get("previous")),
                        "Beklenti":    _f(_e.get("forecast")),
                        "Açıklanan":   _f(_e.get("actual")),
                        "Başlık":      TR_EVENT_TITLES.get(_e.get("title", ""), _e.get("title", "")),
                    })

                _cal_df = pd.DataFrame(_cal_rows)

                def _cal_imp_color(v):
                    if v == "YÜK": return "color: #ff4b4b; font-weight: bold"
                    if v == "ORT": return "color: #ffcc00"
                    if v == "DÜŞ": return "color: #888888"
                    return ""

                _cal_styled = _cal_df.style.map(_cal_imp_color, subset=["Önem"])
                st.caption(
                    f"{CAL_COUNTRIES[_cal_country]} · son {len(_past)} açıklanan + "
                    f"önümüzdeki {_cal_days} gün ({len(_future)} olay) · cache 30dk"
                )
                st.dataframe(_cal_styled, use_container_width=True, hide_index=True)

        # ============================================================
        # 🤖 AI RAPOR YORUMU (Manuel tetikleme + cache + streaming)
        # ============================================================
        st.write("---")
        st.subheader("🤖 AI Rapor Yorumu")

        if not ai_api_key:
            st.info(
                "💡 **Gemini API key girilmedi.** Sol kenar çubuğundan API key girin. "
                "Key almak için: https://aistudio.google.com/app/apikey"
            )
        else:
            st.caption(
                f"Model: **{GEMINI_MODEL}** · "
                f"Detay: **{ai_detail}** (max {AI_DETAIL_LEVELS[ai_detail]} token)"
            )

            _cache_key = ai_cache_key(
                ticker, interval, 0.0, r_close, ai_detail
            )
            _cached = st.session_state.get(_cache_key)

            if auto_refresh_on:
                st.warning(
                    "⚠️ **Canlı Yenileme açık.** Yorum üretimi sürerken sayfa yenilenirse "
                    "yanıt yarıda kesilir. Sol kenar çubuğundan 🔄 **Canlı Yenileme**'yi kapatın."
                )

            _bc1, _bc2, _ = st.columns([1.2, 1.4, 3])
            with _bc1:
                _gen_btn = st.button(
                    "📝 Yorum Al", type="primary",
                    use_container_width=True, key="ai_gen_btn"
                )
            with _bc2:
                _regen_btn = st.button(
                    "🔄 Yeniden Üret",
                    use_container_width=True,
                    disabled=(_cached is None),
                    key="ai_regen_btn",
                )

            if _gen_btn or _regen_btn:
                if _regen_btn:
                    st.session_state.pop(_cache_key, None)

                _sys_p, _usr_p = build_ai_prompt(
                    detail=ai_detail, ticker=ticker, close=r_close,
                    interval=interval, res_rows=res,
                    swing_levels=swing_levels, fib_levels=fib_levels,
                )

                try:
                    _t0 = time.time()

                    with st.spinner(f"🤖 {GEMINI_MODEL} yanıt üretiyor..."):
                        _full_text, _meta = fetch_llm(
                            ai_api_key, _sys_p, _usr_p,
                            AI_DETAIL_LEVELS[ai_detail]
                        )

                    _dt = time.time() - _t0

                    # Yarım cümle güvenlik ağı (safety net)
                    _cleaned, _was_cut = clean_half_sentence(_full_text)
                    _final = _cleaned

                    # Kesilme uyarıları
                    _finish = (_meta or {}).get("finish_reason", "")
                    _finish_lower = str(_finish).lower() if _finish else ""
                    if _finish_lower in ("max_tokens", "length"):
                        _final += (
                            f"\n\n---\n⚠️ **Yanıt token limitine takıldı** "
                            f"(`{_finish}`). Detay seviyesini yükseltin."
                        )
                    elif _finish_lower in ("safety", "recitation", "blocklist", "content_filter"):
                        _final += f"\n\n---\n⚠️ **Yanıt güvenlik filtresi nedeniyle kesildi** (`{_finish}`)."
                    elif _was_cut:
                        _final += (
                            "\n\n---\n⚠️ *Model yanıtı yarıda bıraktı; son yarım cümle otomatik kaldırıldı. "
                            "Yeniden üretmek için 🔄 tuşuna basabilirsiniz.*"
                        )

                    # Token kullanım satırı
                    if _meta:
                        _prompt_t   = _meta.get("prompt_tokens",   0)
                        _output_t   = _meta.get("output_tokens",   0)
                        _thought_t  = _meta.get("thinking_tokens", 0)
                        _total_t    = _meta.get("total_tokens",    0) or (_prompt_t + _output_t + _thought_t)
                        _final += (
                            f"\n\n📊 Token Kullanımı — Prompt: {_prompt_t} · "
                            f"Cevap: {_output_t} · Thinking: {_thought_t} · Toplam: {_total_t}"
                        )

                    if _cleaned:
                        st.markdown(_final)
                        st.session_state[_cache_key] = _final
                        st.caption(f"✅ Tamamlandı · {_dt:.1f}s · ~{len(_cleaned.split())} kelime")
                    else:
                        st.warning("⚠️ Boş yanıt alındı. Farklı bir model veya detay seviyesi deneyin.")
                except RuntimeError as e:
                    st.error(f"❌ API Hatası: {str(e)}")
                except requests.exceptions.Timeout:
                    st.error("❌ Zaman aşımı — sunucu yanıt vermiyor. Tekrar deneyin.")
                except requests.exceptions.ConnectionError as e:
                    st.error(f"❌ Bağlantı hatası: {str(e)[:300]}")
                except Exception as e:
                    st.error(f"❌ {type(e).__name__}: {str(e)[:400]}")

            elif _cached:
                st.markdown(_cached)
                st.caption(
                    "💾 Cache'den gösteriliyor — fiyat/skor değişince anahtar değişir ve "
                    "yeni yorum gerekir. Manuel yenileme için 🔄 tuşuna basın."
                )

    else:
        st.error("Veri çekilemedi. Ticker veya internet bağlantısını kontrol edin.")
