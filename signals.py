"""
signals.py — Çekirdek sinyal/indikatör motoru (Streamlit'ten bağımsız).

app.py (panel) ve trader.py (paper bot) bu modülü ortak kullanır.
Tek kaynak: indikatör hesapları, sig_* sinyal fonksiyonları, backtest,
kombine skor ve z-score karar mantığı buradadır.
"""
import numpy as np
import pandas as pd
from itertools import product as iter_product


def safe_scalar(value):
    if isinstance(value, (pd.Series, np.ndarray)):
        return float(value.iloc[0]) if len(value) > 0 else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        unique_tickers = df.columns.get_level_values(1).unique()
        if len(unique_tickers) <= 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    return df


def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm   = pd.Series(plus_dm,  index=high.index, dtype=float)
    minus_dm  = pd.Series(minus_dm, index=high.index, dtype=float)
    alpha     = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sp        = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm        = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di   = 100 * (sp / atr_s.replace(0, np.nan))
    minus_di  = 100 * (sm / atr_s.replace(0, np.nan))
    dx        = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx       = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average — KAMA + Efficiency Ratio.
    ER (0..1): yön etkinliği. 1=mükemmel trend, 0=tam gürültü.
    Returns (kama_series, er_series).
    """
    ca   = close.values.astype(float)
    kama = np.full(len(ca), np.nan)
    er_a = np.full(len(ca), np.nan)
    kama[period - 1] = ca[period - 1]
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)
    for i in range(period, len(ca)):
        direction  = abs(ca[i] - ca[i - period])
        volatility = np.sum(np.abs(np.diff(ca[i - period:i + 1])))
        er  = 0.0 if volatility == 0 else direction / volatility
        er_a[i] = er
        sc  = (er * (fsc - ssc) + ssc) ** 2
        kama[i] = kama[i - 1] + sc * (ca[i] - kama[i - 1])
    return pd.Series(kama, index=close.index), pd.Series(er_a, index=close.index)


def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    hl2 = (high + low) / 2
    ub  = (hl2 + multiplier * atr).values.astype(float)
    lb  = (hl2 - multiplier * atr).values.astype(float)
    ca  = close.values.astype(float)
    ubf = ub.copy()
    lbf = lb.copy()
    direction  = np.ones(len(ca), dtype=float)
    supertrend = np.full(len(ca), np.nan)
    for i in range(1, len(ca)):
        if np.isnan(ubf[i-1]) or np.isnan(lbf[i-1]):
            ubf[i] = ub[i]
            lbf[i] = lb[i]
        else:
            ubf[i] = ub[i] if (ub[i] < ubf[i-1] or ca[i-1] > ubf[i-1]) else ubf[i-1]
            lbf[i] = lb[i] if (lb[i] > lbf[i-1] or ca[i-1] < lbf[i-1]) else lbf[i-1]
        if   ca[i] > ubf[i-1]: direction[i] = 1
        elif ca[i] < lbf[i-1]: direction[i] = -1
        else:                   direction[i] = direction[i-1]
        supertrend[i] = lbf[i] if direction[i] == 1 else ubf[i]
    return (pd.Series(supertrend, index=close.index), pd.Series(direction, index=close.index),
            pd.Series(lbf, index=close.index),        pd.Series(ubf, index=close.index))


def calc_linear_regression_channel(close, period=50, std_mult=2.0):
    """Linear Regression Channel — Raff 1996.

    Her bar için son `period` kapanışına OLS regresyon uygulanır.
    Returns (mid, upper, lower, slope, r2):
      - mid    : son tahmin (regresyon çizgisinin son noktası)
      - upper  : mid + std_mult × rezidüel_std
      - lower  : mid - std_mult × rezidüel_std
      - slope  : regresyon eğimi (birim: fiyat/bar)
      - r2     : R² (0..1) — regresyonun veriye uyum kalitesi
    """
    n = len(close)
    mid   = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    slope_a = np.full(n, np.nan)
    r2_a    = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = close.values[i - period + 1:i + 1].astype(float)
        x = np.arange(period)
        sl, ic = np.polyfit(x, y, 1)
        yp  = sl * x + ic
        resid = y - yp
        std = np.std(resid)
        mid[i]   = yp[-1]
        upper[i] = yp[-1] + std_mult * std
        lower[i] = yp[-1] - std_mult * std
        slope_a[i] = sl
        # R² = 1 - SS_res / SS_tot
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_a[i] = 1.0 - (np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return (pd.Series(mid, index=close.index),  pd.Series(upper, index=close.index),
            pd.Series(lower, index=close.index), pd.Series(slope_a, index=close.index),
            pd.Series(r2_a, index=close.index))


def calc_vwap_daily(high, low, close, volume):
    tp = (high + low + close) / 3
    dk = pd.Series(close.index.date, index=close.index)
    return (tp * volume).groupby(dk).cumsum() / volume.groupby(dk).cumsum().replace(0, np.nan)


# ── YENİ: Swing Destek/Direnç ─────────────────────────────────────────────────
def find_swing_levels(high, low, close, window=10, min_touches=2, tolerance=0.003,
                      atr_series=None, atr_k=0.5):
    """
    Swing High/Low bazlı otomatik destek/direnç tespiti.
    - atr_series verilirse tolerans = atr_k * ATR / fiyat (dinamik, volatiliteye uyumlu)
    - Aksi halde sabit 'tolerance' yüzdesi kullanılır (geriye uyumluluk)
    - Her seviyenin 'broken' alanı vardır: son kapanış seviyeyi kırmışsa True
    """
    n      = len(close)
    levels = []

    for i in range(window, n - window):
        if high.iloc[i] == high.iloc[i - window: i + window + 1].max():
            levels.append(("R", float(high.iloc[i]), i))
        if low.iloc[i] == low.iloc[i - window: i + window + 1].min():
            levels.append(("S", float(low.iloc[i]), i))

    # Dinamik tolerans: her pivot için kendi ATR'sine göre yüzde tolerans
    def _tol_for(price, bar_idx):
        if atr_series is not None and bar_idx < len(atr_series):
            atr_val = float(atr_series.iloc[bar_idx]) if hasattr(atr_series, "iloc") else float(atr_series[bar_idx])
            if not np.isnan(atr_val) and price > 0:
                return max(atr_k * atr_val / price, 0.0005)  # minimum %0.05 taban
        return tolerance

    merged = []
    used   = set()
    for idx, (typ, price, bar) in enumerate(levels):
        if idx in used:
            continue
        tol         = _tol_for(price, bar)
        touches     = [price]
        touch_bars  = [bar]
        for jdx, (typ2, price2, bar2) in enumerate(levels):
            if jdx != idx and jdx not in used:
                if abs(price2 - price) / price < tol:
                    touches.append(price2)
                    touch_bars.append(bar2)
                    used.add(jdx)
        used.add(idx)
        avg_price  = float(np.mean(touches))
        last_touch = max(touch_bars)

        # ── Break detection & role reversal ──
        # Fiyat bir direnci kırıp yukarı geçerse o seviye artık "destek"
        # Fiyat bir desteği kırıp aşağı inerse o seviye artık "direnç"
        last_close = float(close.iloc[-1])
        tol_now = _tol_for(avg_price, n - 1)
        if typ == "R":
            if last_close > avg_price * (1 + tol_now):
                typ = "S"           # direnç kırıldı, destek oldu
                broken = False      # yeni rolüyle aktif
            else:
                broken = False
        else:  # "S"
            if last_close < avg_price * (1 - tol_now):
                typ = "R"           # destek kırıldı, direnç oldu
                broken = False
            else:
                broken = False

        # ── Recency: son dokunuşun yakınlığı (0-1, yeni olan yüksek) ──
        recency = last_touch / max(n - 1, 1)

        # ── Güç skoru: dokunuş sayısı × recency ağırlığı ──
        strength = len(touches) * (0.5 + 0.5 * recency)

        merged.append({
            "type":       typ,
            "price":      avg_price,
            "touches":    len(touches),
            "last_touch": last_touch,
            "broken":     broken,
            "strength":   strength,
        })

    merged = [m for m in merged if m["touches"] >= min_touches]
    merged = sorted(merged, key=lambda x: -x["strength"])[:10]
    return merged
# ──────────────────────────────────────────────────────────────────────────────


# ── YENİ: Diyagonal Trend Çizgileri ───────────────────────────────────────────
def find_trendlines(high, low, close, pivot_window=10, max_lines=3, tolerance=0.012):
    """
    Gelişmiş otomatik trend çizgisi tespiti.
    - Swing high/low pivotları tespit edilir
    - Her ikili kombinasyon için çizgi skoru hesaplanır
       (dokunuş sayısı + yenilik + ihlal cezası)
    - Benzer eğimli çizgiler tekilleştirilir
    - Paralel destek+direnç çiftleri kanal olarak işaretlenir
    Döndürür: (lines, channels)
      lines   : list of dict  {type, x0,y0,x1,y1,slope,touches,last_touch}
      channels: list of dict  {support, resistance}
    """
    n     = len(close)
    dates = close.index

    # Pivot tespiti
    pivot_highs, pivot_lows = [], []
    for i in range(pivot_window, n - pivot_window):
        if high.iloc[i] == high.iloc[i - pivot_window: i + pivot_window + 1].max():
            pivot_highs.append((i, float(high.iloc[i])))
        if low.iloc[i] == low.iloc[i - pivot_window: i + pivot_window + 1].min():
            pivot_lows.append((i, float(low.iloc[i])))

    def _score_line(p1, p2, pivots, violation_series):
        x1, y1 = p1;  x2, y2 = p2
        if x2 == x1: return 0, []
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        touches   = []
        violations = 0
        for xi in range(min(x1, x2), n):
            y_line = slope * xi + intercept
            y_act  = float(violation_series.iloc[xi])
            rel    = (y_act - y_line) / (abs(y_line) + 1e-9)
            # Dokunuş: pivot bu çizgiye yeterince yakın mı?
            for (px, py) in pivots:
                if px == xi and abs(py - y_line) / (abs(y_line) + 1e-9) < tolerance:
                    touches.append((xi, py))
            # İhlal: fiyat destek/direnç çizgisini kırdı mı?
            if slope >= 0 and rel < -tolerance * 3:   violations += 1
            if slope < 0  and rel >  tolerance * 3:   violations += 1
        score = len(touches) - violations * 0.5
        return score, touches

    def _best_lines(pivots, violation_series, line_type):
        if len(pivots) < 2:
            return []
        candidates = []
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                p1, p2 = pivots[i], pivots[j]
                score, touches = _score_line(p1, p2, pivots, violation_series)
                if score < 1.5 or len(touches) < 2:
                    continue
                x1, y1 = p1;  x2, y2 = p2
                slope     = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                y_end     = slope * (n - 1) + intercept
                last_bar  = max(t[0] for t in touches)
                candidates.append({
                    "type":       line_type,
                    "x0":         x1,         "y0": y1,
                    "x1":         n - 1,      "y1": y_end,
                    "slope":      slope,
                    "intercept":  intercept,
                    "touches":    len(touches),
                    "last_touch": last_bar,
                    "score":      score,
                })
        # Sırala: skor desc, yenilik desc
        candidates.sort(key=lambda c: (-c["score"], -c["last_touch"]))
        # Benzer eğimli çizgileri tekilleştir
        unique = []
        for c in candidates:
            dup = any(
                abs(c["slope"] - u["slope"]) / (abs(u["slope"]) + 1e-9) < 0.08
                for u in unique
            )
            if not dup:
                unique.append(c)
            if len(unique) >= max_lines:
                break
        return unique

    support_lines    = _best_lines(pivot_lows,  low,  "support")
    resistance_lines = _best_lines(pivot_highs, high, "resistance")

    # Kanal tespiti: yaklaşık paralel destek + direnç çiftleri
    channels = []
    for sl in support_lines:
        for rl in resistance_lines:
            sdiff = abs(sl["slope"] - rl["slope"]) / (abs(sl["slope"]) + 1e-9)
            if sdiff < 0.12:
                channels.append({"support": sl, "resistance": rl})

    return support_lines + resistance_lines, channels, dates
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================
# FİBONACCİ, WAVETREND, DIVERGENCE
# ============================================================
def calc_fibonacci(high, low, close, lookback=100, swing_window=5):
    """Fibonacci Retracement — trend yönü ve swing-tabanlı pivot ile.

    Klasik Fibonacci kullanımı:
    - Yükseliş trendinde: swing LOW → swing HIGH yönünde çizilir.
      Seviyeler retracement (geri çekilme) seviyeleri olur — destek olarak görev yapar.
    - Düşüş trendinde:    swing HIGH → swing LOW yönünde çizilir.
      Seviyeler tepki seviyeleri olur — direnç olarak görev yapar.

    Algoritma:
    1. Trend yönü: son lookback barın ilk %25'i ile son %25'inin ortalama
       fiyatları kıyaslanır. Son ortalama yüksekse trend yukarı.
    2. Swing pivotu: lookback içindeki gerçek swing high/low (fractal pivot)
       seçilir; trend yönüne göre **en derin/en yüksek** pivot kullanılır
       (major swing yakalama — kısa vade gürültüsü yerine asıl trend hareketi):
       - Yukarı trend: en derin swing LOW (lookback içindeki en düşük pivot)
                       + ardından gelen en yüksek HIGH
       - Aşağı trend: en yüksek swing HIGH (lookback içindeki en yüksek pivot)
                      + ardından gelen en düşük LOW
    3. Seviyeler statik kalır (mevcut major swing range içinde sabit).

    Returns: (levels_dict, swing_high, swing_low, direction)
      direction: "up" / "down" / "none"
    """
    if len(close) < lookback:
        lookback = len(close)
    if lookback < swing_window * 4:
        # Yetersiz veri için basit min-max'a düş
        recent_high = float(high.iloc[-lookback:].max())
        recent_low  = float(low.iloc[-lookback:].min())
        diff = recent_high - recent_low
        if diff == 0:
            return {}, recent_high, recent_low, "none"
        levels = {
            "0.0%":   recent_low,   "23.6%":  recent_low + 0.236 * diff,
            "38.2%":  recent_low + 0.382 * diff, "50.0%":  recent_low + 0.500 * diff,
            "61.8%":  recent_low + 0.618 * diff, "78.6%":  recent_low + 0.786 * diff,
            "100.0%": recent_high,
        }
        return levels, recent_high, recent_low, "none"

    # 1) Trend yönü tespiti
    seg = close.iloc[-lookback:]
    q   = max(lookback // 4, 5)
    avg_first = float(seg.iloc[:q].mean())
    avg_last  = float(seg.iloc[-q:].mean())
    if avg_last > avg_first * 1.005:    # %0.5 üstü → yukarı trend
        direction = "up"
    elif avg_last < avg_first * 0.995:  # %0.5 altı → aşağı trend
        direction = "down"
    else:
        direction = "none"

    # 2) Swing pivotları (lookback penceresi içinde fractal high/low)
    h_seg = high.iloc[-lookback:].reset_index(drop=True)
    l_seg = low.iloc[-lookback:].reset_index(drop=True)
    swing_highs = []  # (index_in_seg, price)
    swing_lows  = []
    n_seg = len(h_seg)
    for i in range(swing_window, n_seg - swing_window):
        if h_seg.iloc[i] == h_seg.iloc[i - swing_window:i + swing_window + 1].max():
            swing_highs.append((i, float(h_seg.iloc[i])))
        if l_seg.iloc[i] == l_seg.iloc[i - swing_window:i + swing_window + 1].min():
            swing_lows.append((i, float(l_seg.iloc[i])))

    swing_high = swing_low = None

    if direction == "up" and swing_lows:
        # Trend yukarı: lookback içindeki EN DÜŞÜK swing LOW (anlamlı major dip)
        # ve sonrasında oluşan EN YÜKSEK HIGH
        deepest = min(swing_lows, key=lambda x: x[1])
        deepest_idx, deepest_price = deepest
        after = h_seg.iloc[deepest_idx:]
        swing_high = float(after.max())
        swing_low  = deepest_price
    elif direction == "down" and swing_highs:
        # Trend aşağı: lookback içindeki EN YÜKSEK swing HIGH (anlamlı major tepe)
        # ve sonrasında oluşan EN DÜŞÜK LOW
        highest = max(swing_highs, key=lambda x: x[1])
        highest_idx, highest_price = highest
        after = l_seg.iloc[highest_idx:]
        swing_low  = float(after.min())
        swing_high = highest_price
    else:
        # Yön belirsiz: lookback range'inin global high/low'u
        swing_high = float(h_seg.max())
        swing_low  = float(l_seg.min())

    diff = swing_high - swing_low
    if diff == 0:
        return {}, swing_high, swing_low, direction

    # 3) Seviyeler — retracement her iki yön için aynı oranlarda hesaplanır
    # Görsel/yorum yön bilgisiyle yapılır (dokümantasyonda)
    levels = {
        "0.0%":   swing_low,
        "23.6%":  swing_low + 0.236 * diff,
        "38.2%":  swing_low + 0.382 * diff,
        "50.0%":  swing_low + 0.500 * diff,
        "61.8%":  swing_low + 0.618 * diff,
        "78.6%":  swing_low + 0.786 * diff,
        "100.0%": swing_high,
    }
    return levels, swing_high, swing_low, direction


def calc_wavetrend(high, low, close, n1=10, n2=21):
    ap  = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def detect_divergence(price, indicator, window=5):
    n      = len(price)
    result = np.zeros(n)
    pv     = price.values.astype(float)
    iv     = indicator.values.astype(float)
    for i in range(window * 2, n):
        seg_p = pv[max(0, i - window * 4):i + 1]
        seg_i = iv[max(0, i - window * 4):i + 1]
        m     = len(seg_p)
        lows_p = []; lows_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.min(seg_p[j - window:j + window + 1]):
                lows_p.append(seg_p[j])
                lows_i.append(seg_i[j])
        if len(lows_p) >= 2:
            if lows_p[-1] < lows_p[-2] and lows_i[-1] > lows_i[-2]:
                result[i] = 1
        highs_p = []; highs_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.max(seg_p[j - window:j + window + 1]):
                highs_p.append(seg_p[j])
                highs_i.append(seg_i[j])
        if len(highs_p) >= 2:
            if highs_p[-1] > highs_p[-2] and highs_i[-1] < highs_i[-2]:
                result[i] = -1
    return pd.Series(result, index=price.index)


# ============================================================
# 5. SİNYAL FONKSİYONLARI
# ============================================================
def sig_sma(close, sma_s=20, sma_l=100):
    """SMA Crossover — hiyerarşi onaylı.
    AL  : SMA_short > SMA_long  VE  Fiyat > SMA_short
    SAT : SMA_short < SMA_long  VE  Fiyat < SMA_short
    Diğer tüm durumlar (fiyat kısa MA'nın yanlış tarafında) → NÖTR.
    Bu, kısa MA'nın altına/üstüne sarkan ama crossover henüz dönmemiş
    çelişkili durumlarda whipsaw'ı azaltır.
    """
    sh  = close.rolling(sma_s, min_periods=sma_s).mean()
    sl  = close.rolling(sma_l, min_periods=sma_l).mean()
    buy  = (sh > sl) & (close > sh)
    sell = (sh < sl) & (close < sh)
    sig = np.where(buy, 1, np.where(sell, -1, 0))
    sig = np.where(sh.isna() | sl.isna(), 0, sig)
    return pd.Series(sig, index=close.index), sh, sl


def sig_rsi_fn(close, rsi_period, rsi_lower=30, rsi_upper=70, trend_period=200):
    """RSI sinyali — Wilder EWM + SMA trend filtreli + RSI 50 çıkış.

    Hesaplama:
    - Wilder RSI: EWM(alpha=1/period) — TradingView/Bloomberg ile tutarlı.
    - SMA(trend_period) trend filtresi: catching a falling knife önlemi.
      AL yalnızca fiyat SMA üzerinde, SAT yalnızca fiyat SMA altında geçerli.
    - Giriş: RSI < rsi_lower → AL, RSI > rsi_upper → SAT.
    - Çıkış: Long için RSI 50'yi yukarı geçince kapat (Connors standardı).
             Short için RSI 50'yi aşağı geçince kapat.
    """
    d     = close.diff()
    gain  = d.where(d > 0, 0.0)
    loss  = (-d.where(d < 0, 0.0))
    # Wilder smoothing: ilk değer SMA, sonrası EWM (adjust=False, alpha=1/period)
    alpha = 1.0 / rsi_period
    avg_g = gain.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    avg_l = loss.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    rsi   = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))

    # Giriş sinyalleri
    rsi_v   = rsi.values
    entry   = np.where(rsi_v < rsi_lower, 1, np.where(rsi_v > rsi_upper, -1, 0))

    # RSI 50 çıkış: long pozisyon RSI 50 yukarı geçince SAT,
    #               short pozisyon RSI 50 aşağı geçince AL
    cross_above_50 = (rsi_v >= 50) & (np.concatenate(([50], rsi_v[:-1])) < 50)
    cross_below_50 = (rsi_v <= 50) & (np.concatenate(([50], rsi_v[:-1])) > 50)

    sig      = np.zeros(len(rsi_v), dtype=float)
    position = 0
    for i in range(len(rsi_v)):
        if position == 0:
            if entry[i] == 1:  position = 1;  sig[i] = 1
            elif entry[i] == -1: position = -1; sig[i] = -1
        elif position == 1:
            if cross_above_50[i]: position = 0; sig[i] = -1  # long kapat
            else: sig[i] = 1
        elif position == -1:
            if cross_below_50[i]: position = 0; sig[i] = 1   # short kapat
            else: sig[i] = -1

    # Trend filtresi: SMA(trend_period)
    trend_sma = close.rolling(trend_period, min_periods=trend_period).mean()
    above = (close > trend_sma).values
    below = (close < trend_sma).values
    valid = trend_sma.notna().values
    sig = np.where(valid & (sig == 1)  & above, 1,
          np.where(valid & (sig == -1) & below, -1,
          np.where(~valid, sig, 0)))
    return pd.Series(sig, index=close.index), rsi


def sig_bb(close, bb_period, bb_std_val=2.0, trend_period=200):
    """Bollinger Bands sinyali — SMA trend filtreli mean reversion.

    Hesaplama:
    - Orta çizgi: SMA(bb_period)
    - Üst/alt bantlar: orta ± bb_std_val * std
    - Giriş: fiyat alt bandın altında → AL, üst bandın üstünde → SAT
    - Trend filtresi: SMA(trend_period)
      AL yalnızca fiyat trend SMA üstündeyse geçerli (yükselen trendde dip alımı).
      SAT yalnızca fiyat trend SMA altındaysa geçerli (düşen trendde tepe satışı).
      Bu, BB mean reversion'un trendli piyasada whipsaw yapmasını önler.
    """
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    up  = mid + bb_std_val * std
    lo  = mid - bb_std_val * std
    sig = np.where(close < lo, 1, np.where(close > up, -1, 0))

    # Trend filtresi: SMA(trend_period)
    trend_sma = close.rolling(trend_period, min_periods=trend_period).mean()
    above = (close > trend_sma).values
    below = (close < trend_sma).values
    valid = trend_sma.notna().values
    sig = np.where(valid & (sig == 1)  & above, 1,
          np.where(valid & (sig == -1) & below, -1,
          np.where(~valid, sig, 0)))
    return pd.Series(sig, index=close.index), mid, up, lo


def sig_macd(close, macd_fast=12, macd_slow=26, macd_sig_p=9):
    ef   = close.ewm(span=macd_fast, adjust=False).mean()
    es   = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ef - es
    ms   = macd.ewm(span=macd_sig_p, adjust=False).mean()
    sig  = np.where(macd > ms, 1, -1)
    sig  = np.where(macd.isna() | ms.isna(), 0, sig)
    return pd.Series(sig, index=close.index), macd, ms


def sig_obv(close, volume, obv_short, obv_long):
    obv = (volume * np.sign(close.diff()).fillna(0)).cumsum()
    s   = obv.rolling(obv_short, min_periods=obv_short).mean()
    l   = obv.rolling(obv_long,  min_periods=obv_long).mean()
    sig = np.where(s > l, 1, -1)
    sig = np.where(s.isna() | l.isna(), 0, sig)
    return pd.Series(sig, index=close.index), obv, s, l


def sig_adx_fn(high, low, close, adx_period, adx_threshold=25):
    adx_v, pdi, mdi = calc_adx(high, low, close, period=adx_period)
    sig = np.where(adx_v > adx_threshold, np.where(pdi > mdi, 1, -1), 0)
    return pd.Series(sig, index=close.index), adx_v, pdi, mdi


def sig_stochrsi(close, rsi_series, rsi_ma_series, srsi_period, sd_period, sl, su):
    """Stoch RSI — bölge + %K/%D kesişim + RSI MA momentum filtreli sinyal.
    - Aşırı satım (K < sl) VE yukarı dönüş (K > D) VE RSI > RSI_MA → AL (+1)
    - Aşırı alım  (K > su) VE aşağı dönüş (K < D) VE RSI < RSI_MA → SAT (-1)
    - Aksi halde nötr (0)

    Çift teyit:
    1. Bölgede olmak yetmez — K/D kesişimi dönüş teyidi şarttır.
    2. RSI > RSI_MA: kısa-vadeli momentum yukarı eğimli → AL'lar geçerli.
       RSI < RSI_MA: kısa-vadeli momentum aşağı eğimli → SAT'lar geçerli.
       Bu filtre RSI ekosistemi ile tutarlılık sağlar; trend dönüşü henüz
       teyitlenmemişken erken sinyalleri eler.
    """
    rmin = rsi_series.rolling(srsi_period, min_periods=srsi_period).min()
    rmax = rsi_series.rolling(srsi_period, min_periods=srsi_period).max()
    k    = ((rsi_series - rmin) / (rmax - rmin).replace(0, np.nan) * 100).fillna(50).clip(0, 100)
    d    = k.rolling(sd_period).mean()

    # RSI MA momentum filtresi
    rsi_ma_v = rsi_ma_series.values
    rsi_v    = rsi_series.values
    momentum_up   = rsi_v > rsi_ma_v
    momentum_down = rsi_v < rsi_ma_v
    valid_ma      = ~np.isnan(rsi_ma_v)

    # Kesişim teyitli + momentum filtreli sinyal
    bull = (k < sl) & (k > d)   # Aşırı satımda yukarı dönüş
    bear = (k > su) & (k < d)   # Aşırı alımda aşağı dönüş
    sig  = np.where(valid_ma & bull & momentum_up,   1,
           np.where(valid_ma & bear & momentum_down, -1, 0))
    return pd.Series(sig, index=close.index), k, d


def sig_ichimoku(high, low, close, it, ik, isb):
    """Ichimoku Kinko Hyo — klasik 5'li set, üçlü teyitli sinyal (Hosoda).

    Bileşenler:
    - Tenkan-sen   : Kısa vade denge çizgisi (it bar)
    - Kijun-sen    : Orta vade denge çizgisi (ik bar)
    - Senkou A/B   : Bulut sınırları (ik bar İLERİ kaydırılır)
    - Chikou Span  : Kapanışın ik bar GERİ kaydırılmış hali (trend teyidi)

    Sinyal — üç koşul birden gerekli (Hosoda klasiği):
    1. TK Cross           : Tenkan > Kijun (AL) / Tenkan < Kijun (SAT)
    2. Fiyat-Bulut        : close > cloud_top (AL) / close < cloud_bottom (SAT)
    3. Chikou onayı       : close > close.shift(ik) (AL) / close < close.shift(ik) (SAT)
    """
    tenkan   = (high.rolling(it).max()  + low.rolling(it).min())  / 2
    kijun    = (high.rolling(ik).max()  + low.rolling(ik).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(ik)
    senkou_b = ((high.rolling(isb).max() + low.rolling(isb).min()) / 2).shift(ik)
    chikou   = close.shift(-ik)  # Bugünün kapanışı, ik bar geriye

    # Chikou teyidi: bugünün kapanışı, ik bar önceki kapanıştan yüksek mi?
    # (Hosoda: Chikou geçmiş fiyatların üstünde → boğa, altında → ayı)
    chikou_bull = close > close.shift(ik)
    chikou_bear = close < close.shift(ik)

    ct = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cb = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    sig = np.where((tenkan > kijun) & (close > ct) & chikou_bull,  1,
                   np.where((tenkan < kijun) & (close < cb) & chikou_bear, -1, 0))
    return pd.Series(sig, index=close.index), tenkan, kijun, senkou_a, senkou_b, chikou


def sig_kama_fn(close, kp, kf, ks, er_threshold=0.30, slope_window=3):
    """KAMA sinyali — Kaufman'ın felsefesine uygun: eğim + ER filtresi.

    Cross değil, eğim:
    - KAMA son `slope_window` barda yukarı eğimli VE ER yeterli → AL
    - KAMA son `slope_window` barda aşağı eğimli VE ER yeterli → SAT
    - ER < threshold → trendsiz, sinyal sıfırla (ATR filtresine gerek yok;
      ER zaten yön kalitesini doğrudan ölçüyor)

    Notlar:
    - ATR filtresi kaldırıldı: ATR mutlak volatiliteyi ölçer, KAMA için
      kritik olan yön etkinliği (ER). Yüksek ATR + düşük ER = yatay zikzak.
    - Eğim 1-bar diff yerine `slope_window` barlık fark: tek bar gürültüsünü
      filtreler, küçük yatay dalgalanmalarda sinyal çıkmaz.
    """
    kama, er = calc_kama(close, period=kp, fast=kf, slow=ks)
    slope    = kama.diff(slope_window)

    sig = np.where((slope > 0) & (er >= er_threshold),  1,
          np.where((slope < 0) & (er >= er_threshold), -1, 0))
    sig = np.where(kama.isna() | er.isna(), 0, sig)
    return pd.Series(sig, index=close.index), kama, er


def sig_supertrend_fn(high, low, close, stp, stm):
    """SuperTrend — flip-based event sinyali.

    Klasik ATR-trailing yapı (Seban 2008) state machine olarak çalışır:
    yön değişimi yalnızca fiyat bandı kırınca olur.

    Sinyal mantığı:
    - direction[t] != direction[t-1] AND direction[t] == 1  → AL  (+1, flip-up)
    - direction[t] != direction[t-1] AND direction[t] == -1 → SAT (-1, flip-down)
    - aksi halde 0 (yön korunuyor, yeni sinyal yok)

    Notlar:
    - ATR filtresi KALDIRILDI: SuperTrend zaten ATR-bazlı bir göstergedir
      (band genişliği = ATR × multiplier). Düşük volatilitede band daralır,
      flip nadir olur — ek ATR filtresi çift filtre olur.
    - Eski "her bar direction" davranışı SMA200 trend filtresi gibi çalışıyordu;
      bu sürüm SuperTrend'in özgün event-based karakterini ortaya çıkarır.
    - Yön bilgisi (direction serisi) ayrıca döndürülür — grafik renklendirme,
      trailing stop kullanımı ve rejim göstergesi olarak ihtiyaç var.
    """
    st, std, lb, ub = calc_supertrend(high, low, close, period=stp, multiplier=stm)
    d = std.values
    flip = np.zeros(len(d), dtype=float)
    for i in range(1, len(d)):
        if not np.isnan(d[i]) and not np.isnan(d[i-1]) and d[i] != d[i-1]:
            flip[i] = d[i]   # +1 flip-up, -1 flip-down
    flip = np.where(st.isna(), 0, flip)
    return pd.Series(flip, index=close.index), st, std, lb, ub


def sig_lrc(close, lrc_period, lrc_std_mult=2.0):
    """LR Channel sinyali — slope-aware mean reversion.

    Klasik bant dokunma + slope filtresi:
    - slope >= 0 (yükselen/yatay regresyon):
        close < lower  → AL  (trend yönünde dip alımı)
        close > upper  → 0   (trend yönüne ters mean reversion — ele)
    - slope < 0 (düşen regresyon):
        close > upper  → SAT (trend yönünde tepe satışı)
        close < lower  → 0   (trend yönüne ters — ele)

    Felsefe:
    LRC'nin gücü "kanalın eğimli" olmasıdır. Slope yönü zaten bir trend filtresidir.
    Trende ters mean reversion sinyalleri (yükselen kanalda üst banda dokunma → SAT)
    whipsaw üretir; bu sürüm onları siler.
    """
    mid, up, lo, slope, r2 = calc_linear_regression_channel(
        close, period=lrc_period, std_mult=lrc_std_mult)
    cv = close.values
    sl_v = slope.values
    up_v = up.values
    lo_v = lo.values

    bullish_trend = sl_v >= 0
    bearish_trend = sl_v <  0

    # AL: yükselen/yatay kanalda alt banda dokunma
    bull_signal = bullish_trend & (cv < lo_v)
    # SAT: düşen kanalda üst banda dokunma
    bear_signal = bearish_trend & (cv > up_v)

    sig = np.where(bull_signal,  1, np.where(bear_signal, -1, 0))
    sig = np.where(mid.isna() | slope.isna(), 0, sig)
    return pd.Series(sig, index=close.index), mid, up, lo, slope, r2


def sig_vwap_fn(high, low, close, volume, vwap_band_pct):
    vwap = calc_vwap_daily(high, low, close, volume)
    band = vwap * (vwap_band_pct / 100)
    sig  = np.where(close > vwap + band, 1, np.where(close < vwap - band, -1, 0))
    sig  = np.where(vwap.isna(), 0, sig)
    return pd.Series(sig, index=close.index), vwap


def sig_wavetrend_fn(high, low, close, rsi_series, rsi_ma_series, n1=10, n2=21, ob=60, os_=-60):
    """WaveTrend (LazyBear 2014) — bölge + cross + RSI MA momentum filtreli sinyal.

    Çift teyit (klasik):
    1. Aşırı satım (WT1 < os_) VE WT1 yukarı kesti WT2'yi → AL bölge sinyali
    2. Aşırı alım  (WT1 > ob)  VE WT1 aşağı kesti WT2'yi → SAT bölge sinyali

    Üçüncü teyit (RSI MA momentum filtresi):
    - AL  yalnızca RSI > RSI_MA ise geçerli (momentum yukarı eğimli)
    - SAT yalnızca RSI < RSI_MA ise geçerli (momentum aşağı eğimli)

    Felsefe:
    WaveTrend salt mean reversion göstergesi olarak trendli piyasada whipsaw üretir.
    StochRSI'da uyguladığımız aynı RSI_MA filtresi WaveTrend'e de uygulanır —
    iki gösterge artık simetrik mimaride: farklı veri kaynağından (RSI vs fiyat)
    aynı kalite filtresinden geçen mean reversion sinyalleri.
    """
    wt1, wt2   = calc_wavetrend(high, low, close, n1=n1, n2=n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    # Klasik bölge + cross sinyalleri
    bull = cross_up   & (wt1 < os_)
    bear = cross_down & (wt1 > ob)

    # RSI MA momentum filtresi
    rsi_ma_v = rsi_ma_series.values
    rsi_v    = rsi_series.values
    momentum_up   = rsi_v > rsi_ma_v
    momentum_down = rsi_v < rsi_ma_v
    valid_ma      = ~np.isnan(rsi_ma_v)

    sig = np.where(valid_ma & bull & momentum_up,   1,
          np.where(valid_ma & bear & momentum_down, -1, 0))
    return pd.Series(sig, index=close.index), wt1, wt2


# ============================================================
# 6. BACKTEST YARDIMCISI
# ============================================================
def bars_per_year_from_interval(interval):
    """Interval string'i yıllık bar sayısına çevirir (Sharpe yıllıklandırması için)."""
    m = {
        "1m":  252 * 390,  "2m":  252 * 195,  "5m":  252 * 78,
        "15m": 252 * 26,   "30m": 252 * 13,   "60m": 252 * 6.5,
        "1h":  252 * 6.5,  "4h":  252 * 1.625, "8h": 252 * 0.8125,
        "1d":  252,        "1wk": 52,         "1mo": 12,
    }
    return m.get(interval, 252)


def _strategy_bar_returns(sig_vals, close_arr):
    """Sinyal + fiyat → bar-bazlı strateji log getirisi (pozisyon 1 bar geciktirilmiş)."""
    sig_vals  = np.asarray(sig_vals)
    close_arr = np.asarray(close_arr, dtype=float)
    if len(sig_vals) < 2 or not (close_arr > 0).all():
        return np.array([])
    position = np.zeros(len(sig_vals))
    in_pos = False
    for i in range(1, len(sig_vals)):
        if not in_pos and sig_vals[i] == 1 and sig_vals[i-1] != 1: in_pos = True
        elif in_pos and sig_vals[i] == -1 and sig_vals[i-1] != -1: in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    pos_lag = np.concatenate(([0.0], position[:-1]))
    log_ret = np.diff(np.log(close_arr), prepend=np.log(close_arr[0]))
    return pos_lag * log_ret


def permutation_pvalue(strat_ret, observed_sharpe, bars_per_year, n_perm=200, seed=42):
    """(Geriye dönük uyumluluk için) Basit permutation test.
    YENİ KODDA stationary_bootstrap_pvalue TERCİH EDİN."""
    strat_ret = np.asarray(strat_ret)
    if len(strat_ret) < 10 or strat_ret.std() == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        shuf = rng.permutation(strat_ret)
        if shuf.std() == 0: continue
        s = float(shuf.mean() / shuf.std() * np.sqrt(bars_per_year))
        if s >= observed_sharpe:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


def stationary_bootstrap_pvalue(strat_ret, observed_sharpe, bars_per_year,
                                 n_boot=200, avg_block_len=10, seed=42):
    """Politis & Romano (1994) Stationary Bootstrap.

    Finansal getirilerin bağımsız olmadığı gerçeğini dikkate alır.
    Blok uzunlukları geometrik dağılımdan seçilir (ortalama = avg_block_len).
    Zaman serisi yapısı (volatility clustering, autocorrelation) korunur.

    Basit permutation'a göre p-değeri genellikle daha yüksek (daha dürüst) çıkar.
    """
    strat_ret = np.asarray(strat_ret)
    n = len(strat_ret)
    if n < 20 or strat_ret.std() == 0:
        return 1.0

    p_geom = 1.0 / max(avg_block_len, 2)  # blok başlangıç olasılığı
    rng = np.random.default_rng(seed)
    count_ge = 0
    valid_iters = 0

    for _ in range(n_boot):
        # Stationary bootstrap örneği oluştur
        boot = np.empty(n, dtype=strat_ret.dtype)
        idx = int(rng.integers(0, n))
        for i in range(n):
            boot[i] = strat_ret[idx]
            # Yeni blok başlatma olasılığı
            if rng.random() < p_geom:
                idx = int(rng.integers(0, n))
            else:
                idx = (idx + 1) % n  # aynı bloğa devam

        if boot.std() == 0:
            continue
        valid_iters += 1
        # Null dağılım: sharpe'ı "getirileri merkezileştirilmiş" örnekle ölç
        # (H0: gerçek Sharpe = 0 varsayımı altında)
        centered = boot - boot.mean()
        if centered.std() == 0:
            continue
        s_boot = float(centered.mean() / centered.std() * np.sqrt(bars_per_year))
        if s_boot >= observed_sharpe:
            count_ge += 1

    if valid_iters == 0:
        return 1.0
    return (count_ge + 1) / (valid_iters + 1)


def _norm_ppf(p):
    """Inverse of the standard normal CDF (scipy bağımsız).
    Peter Acklam's algorithm (1/2003), stdlib-only, ~1e-9 doğruluk.
    Girdi: 0 < p < 1. Çıktı: Φ⁻¹(p).
    """
    from math import sqrt, log
    if p <= 0.0 or p >= 1.0:
        # Aşırı uçlar için yaklaşım (pratik kullanımda olmaz ama güvenli)
        if p <= 0.0: return -float("inf")
        if p >= 1.0: return  float("inf")

    # Katsayılar (Acklam 2003)
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    p_low  = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = sqrt(-2.0 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0):
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Multiple testing ('data snooping') cezasını çıkarır.
    n_trials: kaç parametre kombinasyonu denendiği (örn. grid size)
    n_obs:    örneklem boyutu (bar sayısı)
    skew:     getiri dağılımının çarpıklığı
    kurt:     getiri dağılımının basıklığı (normal = 3)

    DSR > 0 → Gerçekten rastgeleden iyi.
    DSR 0   → Eşik: istatistiksel olarak anlamsız.
    DSR < 0 → Bu Sharpe muhtemelen şans eseri.
    """
    from math import log, sqrt, exp
    if n_trials <= 1 or n_obs <= 1:
        return observed_sharpe  # düzeltme gerekmiyor

    # Euler-Mascheroni sabiti
    emc = 0.5772156649
    # Expected Max Sharpe under null (Bailey & López de Prado 2014, Eq. 6)
    # E[max SR] ≈ sqrt(V[SR]) × ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
    try:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * exp(1)))
        expected_max_sr = (1.0 - emc) * z1 + emc * z2
    except Exception:
        # Fallback: N büyükse Gumbel'den yaklaşık
        expected_max_sr = sqrt(2.0 * log(max(n_trials, 2)))

    # DSR: Probabilistic SR'nin deflate edilmiş hali
    # σ(SR_hat) = sqrt((1 - skew·SR + (kurt-1)/4 · SR²) / (n_obs - 1))
    sr = observed_sharpe
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(n_obs - 1, 1)
    if var_sr <= 0:
        return sr - expected_max_sr
    std_sr = sqrt(var_sr)
    if std_sr == 0:
        return sr - expected_max_sr

    # DSR = (gözlemlenen SR - beklenen max SR) / std(SR)
    # Yüksek pozitif = gerçek, 0 civarı = sınırda, negatif = şans
    dsr = (sr - expected_max_sr) / std_sr
    return dsr


def run_backtest(signal_series, close_arr, cost_pct, bars_per_year=252):
    sig    = signal_series.values if hasattr(signal_series, "values") else signal_series
    sig    = np.asarray(sig)
    close_arr = np.asarray(close_arr, dtype=float)
    trades = []
    in_pos = False
    entry_p = 0.0
    for i in range(1, len(sig)):
        if not in_pos and sig[i] == 1 and sig[i-1] != 1:
            entry_p = float(close_arr[i])
            in_pos  = True
        elif in_pos and sig[i] == -1 and sig[i-1] != -1:
            ep = float(close_arr[i])
            trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)
            in_pos = False
    if in_pos:
        ep = float(close_arr[-1])
        trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)

    # ── Bar-bazlı yıllıklandırılmış Sharpe (akademik standart) ──
    strat_ret = _strategy_bar_returns(sig, close_arr)
    if len(strat_ret) > 1 and strat_ret.std() > 0:
        sharpe_bar = float(strat_ret.mean() / strat_ret.std() * np.sqrt(bars_per_year))
    else:
        sharpe_bar = 0.0

    if not trades:
        return {"total_ret": 0.0, "sharpe": round(sharpe_bar, 4), "sharpe_trade": 0.0, "n": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_dd": 0.0, "pf": 0.0}
    r      = np.array(trades)
    cumul  = 1.0
    peak   = 1.0
    max_dd = 0.0
    for rv in r:
        cumul *= (1 + rv / 100)
        if cumul > peak: peak = cumul
        dd = ((peak - cumul) / peak) * 100
        if dd > max_dd: max_dd = dd
    wins      = r[r > 0]
    losses    = r[r <= 0]
    total_ret = (cumul - 1) * 100
    wr        = len(wins) / len(r) * 100
    sharpe_trade = float(np.mean(r) / np.std(r)) * np.sqrt(len(r)) if len(r) > 1 and np.std(r) > 0 else 0.0
    pf        = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    return {"total_ret": round(total_ret, 4),
            "sharpe":       round(sharpe_bar, 4),     # yıllıklandırılmış bar-bazlı
            "sharpe_trade": round(sharpe_trade, 4),   # eski metrik (referans)
            "n": len(r),
            "win_rate": round(wr, 2),
            "avg_win":  round(float(wins.mean())   if len(wins)   > 0 else 0.0, 4),
            "avg_loss": round(float(losses.mean())  if len(losses) > 0 else 0.0, 4),
            "max_dd":   round(max_dd, 4),
            "pf":       round(pf, 4) if pf != float("inf") else float("inf")}


# ============================================================
# KOMBİNE SKOR + Z-SCORE KARAR MOTORU
# (app.py paneli ve trader.py botu ortak kullanır)
# ============================================================

# Skora katkı veren 13 sinyal kolonu. Her biri +1 (AL) / 0 (TUT) / -1 (SAT).
SIGNAL_COLS = [
    "Sig_SMA", "Sig_RSI", "Sig_BB", "Sig_MACD", "Sig_OBV", "Sig_ADX",
    "Sig_StochRSI", "Sig_Ichimoku", "Sig_KAMA", "Sig_SuperTrend",
    "Sig_LRC", "Sig_VWAP", "Sig_WaveTrend",
]

# Varsayılan indikatör parametreleri (app.py sidebar defaults ile uyumlu).
DEFAULT_PARAMS = {
    "sma_short": 20, "sma_long": 100,
    "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70, "rsi_ma_period": 14,
    "bb_period": 20, "bb_std": 2.0,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "obv_short": 10, "obv_long": 30,
    "adx_period": 14, "adx_threshold": 25,
    "stoch_rsi_period": 14, "stoch_d_period": 3, "stoch_lower": 20, "stoch_upper": 80,
    "ichi_tenkan": 9, "ichi_kijun": 26, "ichi_senkou_b": 52,
    "kama_period": 10, "kama_fast": 2, "kama_slow": 30,
    "st_period": 10, "st_multiplier": 3.0,
    "lrc_period": 50, "lrc_std_mult": 2.0,
    "atr_period": 14,
    "vwap_band_pct": 0.5,
    "wt_n1": 10, "wt_n2": 21, "wt_ob": 60, "wt_os": -60,
}


def compute_indicators(df, params=None, is_intraday=True):
    """
    Ham OHLCV DataFrame'i alır, tüm Sig_* sinyal kolonlarını ve
    indikatör kolonlarını hesaplayıp ekler. Streamlit gerektirmez.

    df: en az Open/High/Low/Close/Volume kolonlarına sahip DataFrame.
    params: DEFAULT_PARAMS üzerine yazılacak sözlük (opsiyonel).
    Döner: kolonları eklenmiş df (kopya).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    df["Sig_SMA"], df["SMA_SHORT"], df["SMA_LONG"] = sig_sma(
        close, p["sma_short"], p["sma_long"])
    df["SMA200"] = close.rolling(200, min_periods=200).mean()

    df["Sig_RSI"], df["RSI"] = sig_rsi_fn(
        close, p["rsi_period"], p["rsi_lower"], p["rsi_upper"])
    df["RSI_MA"] = df["RSI"].rolling(p["rsi_ma_period"]).mean()

    df["Sig_BB"], df["Mid"], df["Up"], df["Low_BB"] = sig_bb(
        close, p["bb_period"], p["bb_std"])

    df["Sig_MACD"], df["MACD"], df["MACD_S"] = sig_macd(
        close, p["macd_fast"], p["macd_slow"], p["macd_signal"])

    df["Sig_OBV"], df["OBV"], _obv_s, _obv_l = sig_obv(
        close, volume, p["obv_short"], p["obv_long"])

    df["Sig_ADX"], df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = sig_adx_fn(
        high, low, close, p["adx_period"], p["adx_threshold"])

    df["Sig_StochRSI"], df["StochRSI_K"], df["StochRSI_D"] = sig_stochrsi(
        close, df["RSI"], df["RSI_MA"], p["stoch_rsi_period"],
        p["stoch_d_period"], p["stoch_lower"], p["stoch_upper"])

    (df["Sig_Ichimoku"], df["Tenkan"], df["Kijun"],
     df["Senkou_A"], df["Senkou_B"], df["Chikou"]) = sig_ichimoku(
        high, low, close, p["ichi_tenkan"], p["ichi_kijun"], p["ichi_senkou_b"])

    df["Sig_KAMA"], df["KAMA"], df["KAMA_ER"] = sig_kama_fn(
        close, p["kama_period"], p["kama_fast"], p["kama_slow"])

    (df["Sig_SuperTrend"], df["SuperTrend"], df["ST_Direction"],
     df["ST_Lower"], df["ST_Upper"]) = sig_supertrend_fn(
        high, low, close, p["st_period"], p["st_multiplier"])

    (df["Sig_LRC"], df["LRC_Mid"], df["LRC_Upper"],
     df["LRC_Lower"], df["LRC_Slope"], df["LRC_R2"]) = sig_lrc(
        close, p["lrc_period"], p["lrc_std_mult"])

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / p["atr_period"],
                        min_periods=p["atr_period"], adjust=False).mean()
    atr_ma   = atr_series.rolling(p["atr_period"], min_periods=p["atr_period"]).mean()
    df["ATR"]      = atr_series
    df["ATR_High"] = (atr_series > atr_ma)

    if is_intraday:
        df["Sig_VWAP"], df["VWAP"] = sig_vwap_fn(
            high, low, close, volume, p["vwap_band_pct"])
    else:
        df["Sig_VWAP"] = 0
        df["VWAP"]     = np.nan

    df["Sig_WaveTrend"], df["WT1"], df["WT2"] = sig_wavetrend_fn(
        high, low, close, df["RSI"], df["RSI_MA"],
        p["wt_n1"], p["wt_n2"], p["wt_ob"], p["wt_os"])

    df["Div_RSI"]  = detect_divergence(close, df["RSI"],  window=5)
    df["Div_MACD"] = detect_divergence(close, df["MACD"], window=5)
    df["Div_OBV"]  = detect_divergence(close, df["OBV"],  window=5)

    return df


def compute_score(df):
    """
    Sig_* kolonlarını toplayıp bar-bazlı kombine skor serisi döndürür.
    Pozitif = net AL eğilimi, negatif = net SAT eğilimi.
    """
    cols = [c for c in SIGNAL_COLS if c in df.columns]
    score = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return score


def zscore_signal(score, lookback=100, k=1.0):
    """
    Kombine skoru EMA bazlı z-score'a çevirir ve AL/SAT/TUT serisi üretir.

    z = (skor - EMA_ortalama) / EMA_std
      z > +k  -> "AL"
      z < -k  -> "SAT"
      arası   -> "TUT"

    lookback: EMA penceresi (span). Panelden ayarlanır.
    k: eşik katsayısı (kaç std uzaklıkta sinyal). Panelden ayarlanır.

    Döner: (decisions: pd.Series[str], z: pd.Series[float])
    """
    score = pd.Series(score).astype(float)
    ema_mean = score.ewm(span=lookback, adjust=False).mean()
    ema_var  = (score - ema_mean).pow(2).ewm(span=lookback, adjust=False).mean()
    ema_std  = np.sqrt(ema_var)

    z = (score - ema_mean) / ema_std.replace(0, np.nan)
    z = z.fillna(0.0)

    decisions = pd.Series("TUT", index=score.index)
    decisions[z > k]  = "AL"
    decisions[z < -k] = "SAT"
    return decisions, z


def zscore_position_series(score, lookback=100, k=1.0):
    """
    z-score kararlarını pozisyon serisine çevirir (backtest için).
    AL -> 1 (tam long), SAT -> 0 (nakit). Sinyal gelene kadar son pozisyon korunur.
    Döner: pd.Series[int] (0 veya 1), backtest sig_series olarak kullanılabilir.
    """
    decisions, _ = zscore_signal(score, lookback, k)
    pos = pd.Series(np.nan, index=decisions.index, dtype=float)
    pos[decisions == "AL"]  = 1.0
    pos[decisions == "SAT"] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pos.astype(int)


def backtest_from_date(df, start_date, lookback=100, k=1.0,
                       capital=1000.0, cost_pct=0.001, is_intraday=True):
    """
    Panel için: verilen tarihten itibaren z-score stratejisini simüle eder.

    df: ham OHLCV (tüm geçmiş). compute_indicators TÜM veride çalışır
        (indikatörlerin ısınması için), sonra start_date'ten itibaren kesilir.
    start_date: pd.Timestamp veya str. Bu tarihte 'capital' kadar sermaye ile başlanır.
    Döner: dict {pct_return, final_value, n_trades, start_price, last_price, ...}
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    enr = compute_indicators(df, is_intraday=is_intraday)
    score = compute_score(enr)
    pos = zscore_position_series(score, lookback=lookback, k=k)
    enr["_pos"] = pos.values

    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None and enr.index.tz is not None:
        start_ts = start_ts.tz_localize(enr.index.tz)
    seg = enr[enr.index >= start_ts]
    if len(seg) < 2:
        return {"ok": False, "msg": "Seçilen tarihten sonra yeterli veri yok."}

    close_arr = seg["Close"].values
    pos_arr   = seg["_pos"].values
    stats = run_backtest(pos_arr, close_arr, cost_pct=cost_pct,
                         bars_per_year=bars_per_year_from_interval("15m"))

    pct = stats["total_ret"]
    final_value = capital * (1 + pct / 100.0)
    dec, z = zscore_signal(score, lookback=lookback, k=k)
    return {
        "ok": True,
        "pct_return": pct,
        "final_value": final_value,
        "capital": capital,
        "n_trades": stats["n"],
        "win_rate": stats["win_rate"],
        "max_dd": stats["max_dd"],
        "start_price": float(close_arr[0]),
        "last_price": float(close_arr[-1]),
        "last_decision": dec.iloc[-1],
        "last_z": float(z.iloc[-1]),
        "n_bars": len(seg),
    }
