# ================================================================
# Michael? Charis? Santoso? gw perlu FORES
# ================================================================
# ANAK anak ayam, hobi turu
# ================================================================

import os, time, json, logging, requests
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ================================================================
# KONFIGURASI 4 PAIR
# Setiap pair punya konfigurasi sendiri karena karakteristiknya beda
# XAU/USD (emas) pip berbeda dari forex biasa
# ================================================================
PAIRS = {
    "EURUSD": {
        "yahoo"    : "EURUSD=X",
        "alpha"    : "EUR/USD",
        "twelve"   : "EUR/USD",
        "name"     : "EUR/USD",
        "pip_mult" : 10000,   # 1 pip = 0.0001
        "pip_val"  : 10.0,    # $10 per pip per lot standar
        "decimals" : 5,
    },
    "GBPUSD": {
        "yahoo"    : "GBPUSD=X",
        "alpha"    : "GBP/USD",
        "twelve"   : "GBP/USD",
        "name"     : "GBP/USD",
        "pip_mult" : 10000,
        "pip_val"  : 10.0,
        "decimals" : 5,
    },
    "USDJPY": {
        "yahoo"    : "USDJPY=X",
        "alpha"    : "USD/JPY",
        "twelve"   : "USD/JPY",
        "name"     : "USD/JPY",
        "pip_mult" : 100,     # 1 pip = 0.01 untuk JPY
        "pip_val"  : 9.09,
        "decimals" : 3,
    },
    "XAUUSD": {
        "yahoo"    : "GC=F",   # Gold Futures di Yahoo
        "alpha"    : None,     # Alpha tidak support XAU gratis
        "twelve"   : "XAU/USD",
        "name"     : "XAU/USD (Gold)",
        "pip_mult" : 10,       # 1 pip = $0.1 untuk emas
        "pip_val"  : 1.0,
        "decimals" : 2,
    },
}

CONFIG = {
    "timeframe"       : "15m",
    "risk_percent"    : 1.0,
    "min_confidence"  : 75,
    "rr_ratio"        : 1.5,
    "loop_interval"   : 60,    # Analisa ulang setiap 60 detik
    "account_balance" : 10000,
    "server_url"      : "http://localhost:5000/update_all",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("robot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# ================================================================
# DATA SOURCES
# ================================================================
def get_data_yahoo(symbol, period="5d", interval="15m"):
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return None
        df.index = pd.to_datetime(df.index)
        return df[['Open','High','Low','Close','Volume']]
    except Exception as e:
        log.error(f"[Yahoo:{symbol}] {e}"); return None

def get_data_alpha(symbol, interval="15min"):
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key or not symbol: return None
    try:
        r = requests.get("https://www.alphavantage.co/query", params={
            "function":"FX_INTRADAY","from_symbol":symbol[:3],
            "to_symbol":symbol[-3:],"interval":interval,
            "outputsize":"compact","apikey":key
        }, timeout=15)
        data = r.json()
        k = f"Time Series FX ({interval})"
        if k not in data: return None
        df = pd.DataFrame(data[k]).T.sort_index()
        df.columns = ['Open','High','Low','Close']
        df = df.astype(float); df['Volume'] = 0
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

def get_data_twelve(symbol, interval="15min"):
    key = os.getenv("TWELVE_DATA_KEY")
    if not key: return None
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params={
            "symbol":symbol,"interval":interval,"outputsize":200,"apikey":key
        }, timeout=15)
        data = r.json()
        if "values" not in data: return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        for c in ['Open','High','Low','Close','Volume']:
            if c not in df.columns: df[c] = 0
        return df[['Open','High','Low','Close','Volume']].astype(float)
    except: return None

def get_best_data(pair_key):
    cfg = PAIRS[pair_key]
    # Coba Twelve Data dulu (paling akurat)
    df = get_data_twelve(cfg['twelve'], "15min")
    # Fallback Alpha Vantage
    if (df is None or len(df) < 50) and cfg['alpha']:
        df = get_data_alpha(cfg['alpha'], "15min")
    # Fallback Yahoo
    if df is None or len(df) < 50:
        df = get_data_yahoo(cfg['yahoo'], "5d", "15m")
    if df is not None and len(df) >= 50:
        log.info(f"[Data:{pair_key}] OK — {len(df)} candle")
    return df

def get_macro_fred():
    results = {}
    endpoints = {
        "DXY"     : "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS",
        "FEDFUNDS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
        "T10Y2Y"  : "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
    }
    for name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=10)
            for line in reversed(resp.text.strip().split("\n")[1:]):
                parts = line.split(",")
                if len(parts)==2 and parts[1] != ".":
                    results[name] = float(parts[1]); break
        except: results[name] = None
    log.info(f"[FRED] DXY={results.get('DXY')} FED={results.get('FEDFUNDS')} Yield={results.get('T10Y2Y')}")
    return results


# ================================================================
# INDIKATOR (sama untuk semua pair)
# ================================================================
def hitung_indikator(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df = df.copy()
    df['MA50']        = c.rolling(50).mean()
    df['MA200']       = c.rolling(200).mean()
    df['EMA12']       = c.ewm(span=12, adjust=False).mean()
    df['EMA26']       = c.ewm(span=26, adjust=False).mean()
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']         = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['ATR']         = tr.rolling(14).mean()
    df['BB_mid']      = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['BB_upper']    = df['BB_mid'] + 2*std
    df['BB_lower']    = df['BB_mid'] - 2*std
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df['STOCH_K']     = 100*(c-lo14)/(hi14-lo14+1e-10)
    df['STOCH_D']     = df['STOCH_K'].rolling(3).mean()
    df['VOL_RATIO']   = v / v.rolling(20).mean().replace(0, 1)

    last = df.iloc[-1]; prev = df.iloc[-2]
    def f(x):
        try:
            v = float(x)
            return round(v,5) if v==v else None
        except: return None

    return {
        "close":f(last['Close']),"open":f(last['Open']),"high":f(last['High']),"low":f(last['Low']),
        "volume":f(last['Volume']),"ma50":f(last['MA50']),"ma200":f(last['MA200']),
        "ema12":f(last['EMA12']),"ema26":f(last['EMA26']),"macd":f(last['MACD']),
        "macd_signal":f(last['MACD_signal']),"macd_hist":f(last['MACD_hist']),
        "prev_macd_hist":f(prev['MACD_hist']),"rsi":round(float(last['RSI']),2),
        "atr":f(last['ATR']),"bb_upper":f(last['BB_upper']),"bb_lower":f(last['BB_lower']),
        "bb_mid":f(last['BB_mid']),"stoch_k":round(float(last['STOCH_K']),2),
        "stoch_d":round(float(last['STOCH_D']),2),
        "vol_ratio":round(float(last['VOL_RATIO']),2) if last['VOL_RATIO']==last['VOL_RATIO'] else 1.0,
        "prev_close":f(prev['Close']),
    }


# ================================================================
# PRICE ACTION
# ================================================================
def analisa_price_action(df):
    last = df.iloc[-1]; prev = df.iloc[-2]
    o,h,l,c = float(last['Open']),float(last['High']),float(last['Low']),float(last['Close'])
    po,pc = float(prev['Open']),float(prev['Close'])
    body = abs(c-o); range_ = h-l+1e-10

    if body < range_*0.08:                               pattern = "Doji"
    elif (min(o,c)-l > body*2) and (h-max(o,c)<body*.5): pattern = "Hammer (Bullish Reversal)"
    elif (h-max(o,c) > body*2) and (min(o,c)-l<body*.5): pattern = "Shooting Star (Bearish)"
    elif c>o and pc<po and c>po and o<pc:                pattern = "Bullish Engulfing"
    elif c<o and pc>po and c<po and o>pc:                pattern = "Bearish Engulfing"
    elif (min(o,c)-l) > (2*body):                        pattern = "Pin Bar Bullish"
    elif (h-max(o,c)) > (2*body):                        pattern = "Pin Bar Bearish"
    else:                                                 pattern = "Candle Bullish" if c>o else "Candle Bearish"

    recent = df.tail(100)
    ma50   = float(df['Close'].tail(50).mean())
    if len(df) >= 200:
        ma200 = float(df['Close'].tail(200).mean())
        if c > ma50 > ma200:    trend = "Uptrend Kuat (Golden Cross)"
        elif c < ma50 < ma200:  trend = "Downtrend Kuat (Death Cross)"
        elif c > ma50:          trend = "Uptrend Lemah"
        else:                   trend = "Downtrend Lemah"
    else:
        trend = "Uptrend" if c > ma50 else "Downtrend"

    pivot = (float(recent['High'].max()) + float(recent['Low'].min()) + c) / 3
    return {
        "pattern":pattern,"trend":trend,
        "support":round(float(recent['Low'].min()),5),
        "resistance":round(float(recent['High'].max()),5),
        "pivot":round(pivot,5),
    }


# ================================================================
# SIGNAL ENGINE — Berlaku untuk semua pair
# ================================================================
def generate_signal(ind, pa, macro, pair_key):
    buy, sell = 0, 0
    reasons   = []
    close = ind['close'] or 0
    atr   = ind['atr'] or 0.001
    pair_name = PAIRS[pair_key]['name']

    # 1. Golden/Death Cross (MA50 vs MA200)
    if ind['ma50'] and ind['ma200']:
        if ind['ma50'] > ind['ma200'] and close > ind['ma50']:
            buy += 3; reasons.append(f"✅ Golden Cross — MA50 > MA200 | Trend naik kuat")
        elif ind['ma50'] < ind['ma200'] and close < ind['ma50']:
            sell += 3; reasons.append(f"🔻 Death Cross — MA50 < MA200 | Trend turun kuat")

    # 2. EMA 12 vs 26
    if ind['ema12'] and ind['ema26']:
        if ind['ema12'] > ind['ema26']:
            buy += 2; reasons.append(f"✅ EMA12({ind['ema12']:.5f}) > EMA26({ind['ema26']:.5f}) | Momentum naik")
        else:
            sell += 2; reasons.append(f"🔻 EMA12({ind['ema12']:.5f}) < EMA26({ind['ema26']:.5f}) | Momentum turun")

    # 3. MACD Histogram
    if ind['macd_hist'] is not None and ind['prev_macd_hist'] is not None:
        if ind['macd_hist'] > 0 and ind['macd_hist'] > ind['prev_macd_hist']:
            buy += 2; reasons.append(f"✅ MACD Histogram naik ({ind['macd_hist']:.5f}) | Bullish momentum menguat")
        elif ind['macd_hist'] < 0 and ind['macd_hist'] < ind['prev_macd_hist']:
            sell += 2; reasons.append(f"🔻 MACD Histogram turun ({ind['macd_hist']:.5f}) | Bearish momentum menguat")
        elif ind['macd'] and ind['macd_signal'] and ind['macd'] > ind['macd_signal']:
            buy += 1; reasons.append("✅ MACD di atas signal line")

    # 4. RSI
    rsi = ind['rsi']
    if rsi < 30:
        buy += 3; reasons.append(f"✅ RSI={rsi:.1f} — Oversold! Kemungkinan kuat bounce naik")
    elif rsi > 70:
        sell += 3; reasons.append(f"🔻 RSI={rsi:.1f} — Overbought! Kemungkinan kuat reversal turun")
    elif 40 <= rsi <= 60:
        buy += 1; reasons.append(f"✅ RSI={rsi:.1f} — Zona sehat, momentum positif")
    elif 60 < rsi <= 70:
        sell += 1; reasons.append(f"⚠ RSI={rsi:.1f} — Mendekati overbought, waspada")

    # 5. Stochastic
    k, d = ind['stoch_k'], ind['stoch_d']
    if k < 20 and k > d:
        buy += 2; reasons.append(f"✅ Stochastic oversold ({k:.1f}) & %K naik | Sinyal beli kuat")
    elif k > 80 and k < d:
        sell += 2; reasons.append(f"🔻 Stochastic overbought ({k:.1f}) & %K turun | Sinyal jual kuat")

    # 6. Bollinger Bands
    if ind['bb_lower'] and ind['bb_upper']:
        if close <= ind['bb_lower'] * 1.001:
            buy += 2; reasons.append(f"✅ Harga menyentuh BB Lower | Potensi bounce naik")
        elif close >= ind['bb_upper'] * 0.999:
            sell += 2; reasons.append(f"🔻 Harga menyentuh BB Upper | Potensi reversal turun")

    # 7. Volume Tick
    vol = ind['vol_ratio']
    if vol > 1.5:
        if buy >= sell: buy += 1; reasons.append(f"✅ Volume {vol:.1f}x rata-rata | Konfirmasi tekanan beli")
        else:           sell += 1; reasons.append(f"🔻 Volume {vol:.1f}x rata-rata | Konfirmasi tekanan jual")
    elif vol < 0.5:
        reasons.append(f"⚠ Volume {vol:.1f}x | Sinyal lemah, risiko false signal")

    # 8. Price Action
    pattern = pa.get('pattern','')
    if any(x in pattern for x in ['Bullish','Hammer','Pin Bar Bull']):
        buy += 2; reasons.append(f"✅ Pola: {pattern}")
    elif any(x in pattern for x in ['Bearish','Shooting','Pin Bar Bear']):
        sell += 2; reasons.append(f"🔻 Pola: {pattern}")

    # 9. Trend
    trend = pa.get('trend','')
    if 'Kuat' in trend and 'Up' in trend:
        buy += 1; reasons.append(f"✅ Trend: {trend}")
    elif 'Kuat' in trend and 'Down' in trend:
        sell += 1; reasons.append(f"🔻 Trend: {trend}")

    # 10. FRED Macro (khusus USD pairs)
    if pair_key in ['EURUSD','GBPUSD']:
        # DXY naik = USD kuat = pair ini turun
        if macro.get('DXY'):
            if macro['DXY'] > 104:
                sell += 1; reasons.append(f"🔻 DXY={macro['DXY']} — Dollar kuat, tekanan pada {pair_name}")
            elif macro['DXY'] < 100:
                buy += 1; reasons.append(f"✅ DXY={macro['DXY']} — Dollar lemah, {pair_name} cenderung naik")
    elif pair_key == 'USDJPY':
        # DXY naik = USD kuat = USD/JPY naik
        if macro.get('DXY'):
            if macro['DXY'] > 104:
                buy += 1; reasons.append(f"✅ DXY={macro['DXY']} — Dollar kuat, USD/JPY cenderung naik")
            elif macro['DXY'] < 100:
                sell += 1; reasons.append(f"🔻 DXY={macro['DXY']} — Dollar lemah, USD/JPY cenderung turun")
    elif pair_key == 'XAUUSD':
        # Emas kebalikan dollar — DXY naik = gold turun
        if macro.get('DXY'):
            if macro['DXY'] > 104:
                sell += 1; reasons.append(f"🔻 DXY={macro['DXY']} — Dollar kuat, tekanan pada Gold")
            elif macro['DXY'] < 100:
                buy += 1; reasons.append(f"✅ DXY={macro['DXY']} — Dollar lemah, Gold cenderung naik")
        if macro.get('T10Y2Y') is not None and macro['T10Y2Y'] < 0:
            buy += 1; reasons.append(f"✅ Yield curve terbalik — Sentimen risk-off, Gold menguat")

    if macro.get('T10Y2Y') is not None and macro['T10Y2Y'] < 0 and pair_key != 'XAUUSD':
        sell += 1; reasons.append(f"⚠ Yield curve terbalik ({macro['T10Y2Y']}) — Sentimen risk-off")

    # Hitung confidence
    total = buy + sell
    if total == 0:
        confidence, direction = 0, "WAIT"
    elif buy > sell:
        confidence = round((buy/(total+2))*100); direction = "BUY"
    else:
        confidence = round((sell/(total+2))*100); direction = "SELL"

    if confidence < CONFIG['min_confidence']:
        direction = "WAIT"
        reasons.append(f"⏸ Confidence {confidence}% belum cukup (min {CONFIG['min_confidence']}%)")

    # TP / SL pakai ATR
    sl_dist = atr * 1.2; tp_dist = atr * CONFIG['rr_ratio'] * 1.2
    if direction == "BUY":    entry=close; tp=round(entry+tp_dist,5); sl=round(entry-sl_dist,5)
    elif direction == "SELL": entry=close; tp=round(entry-tp_dist,5); sl=round(entry+sl_dist,5)
    else:                     entry=close; tp=sl=None
    rr = round(tp_dist/sl_dist,2) if sl_dist > 0 else 0

    return {
        "signal":direction,"confidence":confidence,"entry":entry,"tp":tp,"sl":sl,"rr":rr,
        "buy_score":buy,"sell_score":sell,"reasons":reasons,
        "pair":pair_name,"pair_key":pair_key,
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ================================================================
# MANAJEMEN RISIKO (per pair)
# ================================================================
def hitung_lot(entry, sl, pair_key):
    pcfg     = PAIRS[pair_key]
    risk_usd = CONFIG['account_balance'] * (CONFIG['risk_percent']/100)
    sl_dist  = abs(entry - sl) if sl else 0.001
    sl_pips  = sl_dist * pcfg['pip_mult']
    pip_val  = pcfg['pip_val']
    lot      = round(risk_usd / (sl_pips * pip_val), 2) if sl_pips > 0 else 0.01
    return {
        "lot"          : max(0.01, min(lot, 10.0)),
        "risk_usd"     : round(risk_usd, 2),
        "sl_pips"      : round(sl_pips, 1),
        "potensi_profit": round(sl_pips * pip_val * lot * CONFIG['rr_ratio'], 2),
    }


# ================================================================
# TELEGRAM
# ================================================================
def kirim_telegram(sinyal, risk):
    token = os.getenv("TELEGRAM_TOKEN"); chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    emoji  = "📈" if sinyal['signal']=="BUY" else "📉"
    alasan = "\n".join(sinyal['reasons'][:7])
    pesan  = f"""{emoji} <b>SINYAL {sinyal['signal']} — {sinyal['pair']}</b>

⏰ {sinyal['timestamp']}
💪 Confidence: <b>{sinyal['confidence']}%</b>
📊 BUY={sinyal['buy_score']} | SELL={sinyal['sell_score']}

💰 Entry : <b>{sinyal['entry']}</b>
🎯 TP    : <b>{sinyal['tp']}</b>
🛡 SL    : <b>{sinyal['sl']}</b>
📐 R:R   : 1:{sinyal['rr']} | Lot: {risk['lot']}
✨ Target: ${risk['potensi_profit']}

<b>📋 Alasan:</b>
{alasan}

<i>⚠ Hanya sinyal analisa, bukan saran investasi!</i>"""
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id":chat_id,"text":pesan,"parse_mode":"HTML"}, timeout=10)
        log.info(f"[Telegram] ✅ {sinyal['pair']} terkirim!")
    except Exception as e: log.error(f"[Telegram] ❌ {e}")


# ================================================================
# KIRIM KE SERVER
# ================================================================
def kirim_ke_server(all_results):
    try:
        requests.post(CONFIG['server_url'], json=all_results, timeout=5)
        log.info("[Server] ✅ Data 4 pair terkirim ke website!")
    except requests.exceptions.ConnectionError:
        log.warning("[Server] ⚠ server.py belum jalan! Buka terminal baru: python server.py")
    except Exception as e: log.error(f"[Server] ❌ {e}")

def simpan_sinyal(sinyal, risk):
    filepath = "signals_history.json"
    history  = []
    if os.path.exists(filepath):
        try:
            with open(filepath) as f: history = json.load(f)
        except: pass
    history.append({**sinyal,"risk":risk})
    with open(filepath,"w") as f: json.dump(history,f,indent=2,default=str)


# ================================================================
# ANALISA SATU PAIR
# ================================================================
def analisa_pair(pair_key, macro_data):
    log.info(f"\n  ── Analisa {PAIRS[pair_key]['name']} ──")
    df = get_best_data(pair_key)
    if df is None or len(df) < 50:
        log.warning(f"  Data {pair_key} tidak cukup")
        return None

    ind    = hitung_indikator(df)
    pa     = analisa_price_action(df)
    sinyal = generate_signal(ind, pa, macro_data, pair_key)
    risk   = hitung_lot(sinyal['entry'], sinyal['sl'], pair_key) if sinyal['sl'] else {"lot":0,"risk_usd":0,"sl_pips":0,"potensi_profit":0}

    log.info(f"  {sinyal['pair']}: {sinyal['signal']} ({sinyal['confidence']}%) | Close={ind['close']} RSI={ind['rsi']}")

    # Simpan & kirim Telegram hanya kalau ada sinyal valid
    if sinyal['signal'] in ['BUY','SELL']:
        kirim_telegram(sinyal, risk)
        simpan_sinyal(sinyal, risk)

    # Tambahkan price history untuk chart
    ph = []
    if df is not None:
        ph = [{"time":str(t),"price":float(p)} for t,p in zip(df.index[-150:], df['Close'].tail(150))]

    return {
        **sinyal,
        "indicators"   : ind,
        "price_action" : pa,
        "risk"         : risk,
        "price_history": ph,
        "history"      : [],
    }


# ================================================================
# MAIN LOOP
# ================================================================
def jalankan_robot():
    log.info("="*60)
    log.info("🤖 FOREX ROBOT v3.0 — 4 PAIR MULTI ANALISA")
    log.info("="*60)
    log.info("Pair   : EUR/USD | GBP/USD | USD/JPY | XAU/USD")
    log.info(f"R:R    : 1:{CONFIG['rr_ratio']} | Risk: {CONFIG['risk_percent']}% | Min Conf: {CONFIG['min_confidence']}%")
    log.info("="*60)

    iterasi   = 0
    last_macro = 0
    macro_data = {}

    # Load riwayat sinyal yang sudah ada
    history_cache = {}

    while True:
        iterasi += 1
        log.info(f"\n{'='*55}")
        log.info(f"🔄 Iterasi #{iterasi} — {datetime.now().strftime('%A %H:%M:%S')}")

        try:
            # Update makro tiap 1 jam
            if time.time() - last_macro > 3600:
                log.info("[FRED] Update data makro...")
                macro_data = get_macro_fred()
                last_macro = time.time()

            # Analisa semua 4 pair
            all_results = {}
            for pair_key in PAIRS:
                try:
                    result = analisa_pair(pair_key, macro_data)
                    if result:
                        # Tambah history dari cache
                        if pair_key not in history_cache:
                            history_cache[pair_key] = []
                        if result['signal'] in ['BUY','SELL']:
                            history_cache[pair_key].insert(0, {
                                "signal"    : result['signal'],
                                "confidence": result['confidence'],
                                "entry"     : result['entry'],
                                "tp"        : result['tp'],
                                "sl"        : result['sl'],
                                "timestamp" : result['timestamp'],
                            })
                            history_cache[pair_key] = history_cache[pair_key][:30]
                        result['history'] = history_cache.get(pair_key, [])
                        all_results[pair_key] = result
                except Exception as e:
                    log.error(f"Error analisa {pair_key}: {e}")
                time.sleep(2)  # Jeda 2 detik antar pair biar tidak kena rate limit

            # Tambahkan data makro ke payload
            all_results['_macro'] = macro_data
            all_results['_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Kirim semua hasil ke server
            kirim_ke_server(all_results)

            # Summary di log
            log.info("\n📊 SUMMARY:")
            for pk, r in all_results.items():
                if pk.startswith('_'): continue
                sig = r.get('signal','?')
                conf = r.get('confidence',0)
                icon = "📈" if sig=="BUY" else "📉" if sig=="SELL" else "⏸"
                log.info(f"  {icon} {PAIRS[pk]['name']:15} {sig:4} {conf:3}%")

        except Exception as e:
            log.error(f"❌ Error utama: {e}")
            import traceback; traceback.print_exc()

        log.info(f"\n⏳ Tunggu {CONFIG['loop_interval']} detik...")
        time.sleep(CONFIG['loop_interval'])


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║    🤖 FOREX ROBOT v3.0 — 4 PAIR             ║
║    EUR/USD | GBP/USD | USD/JPY | XAU/USD     ║
╠══════════════════════════════════════════════╣
║  1. python server.py        (terminal 1)     ║
║  2. python forex_robot_v3.py (terminal 2)    ║
║  3. Buka dashboard.html di browser           ║
╚══════════════════════════════════════════════╝
""")
    try: jalankan_robot()
    except KeyboardInterrupt: log.info("Robot dihentikan.")
