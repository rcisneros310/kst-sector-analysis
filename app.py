"""
S&P 500 KST Sector Breadth Analysis - Render LITE
=================================================
~100 stocks (10 per sector) to work within 30-second timeout
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings('ignore')

ENABLE_ML = True
if ENABLE_ML:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
else:
    ML_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'kst_sector_analysis_2024'
DATA_CACHE = {}
CACHE_TIMEOUT = 600

def get_sp500_stocks():
    """Top 10 stocks per sector = ~110 stocks total."""
    return {
        'AAPL': 'Information Technology', 'MSFT': 'Information Technology',
        'NVDA': 'Information Technology', 'AVGO': 'Information Technology',
        'AMD': 'Information Technology', 'CRM': 'Information Technology',
        'ADBE': 'Information Technology', 'CSCO': 'Information Technology',
        'INTC': 'Information Technology', 'QCOM': 'Information Technology',
        'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials',
        'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
        'MS': 'Financials', 'BLK': 'Financials', 'SPGI': 'Financials',
        'AXP': 'Financials',
        'UNH': 'Health Care', 'JNJ': 'Health Care', 'LLY': 'Health Care',
        'PFE': 'Health Care', 'ABBV': 'Health Care', 'MRK': 'Health Care',
        'TMO': 'Health Care', 'ABT': 'Health Care', 'DHR': 'Health Care',
        'BMY': 'Health Care',
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
        'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
        'NKE': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
        'BKNG': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',
        'GOOGL': 'Communication Services', 'META': 'Communication Services',
        'NFLX': 'Communication Services', 'DIS': 'Communication Services',
        'CMCSA': 'Communication Services', 'VZ': 'Communication Services',
        'T': 'Communication Services', 'TMUS': 'Communication Services',
        'CHTR': 'Communication Services', 'EA': 'Communication Services',
        'UNP': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials',
        'CAT': 'Industrials', 'BA': 'Industrials', 'GE': 'Industrials',
        'RTX': 'Industrials', 'LMT': 'Industrials', 'DE': 'Industrials',
        'MMM': 'Industrials',
        'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
        'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
        'MO': 'Consumer Staples', 'CL': 'Consumer Staples',
        'MDLZ': 'Consumer Staples', 'KHC': 'Consumer Staples',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
        'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy',
        'KMI': 'Energy',
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
        'D': 'Utilities', 'AEP': 'Utilities', 'EXC': 'Utilities',
        'SRE': 'Utilities', 'XEL': 'Utilities', 'PEG': 'Utilities',
        'ED': 'Utilities',
        'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
        'FCX': 'Materials', 'NEM': 'Materials', 'NUE': 'Materials',
        'DOW': 'Materials', 'DD': 'Materials', 'PPG': 'Materials',
        'VMC': 'Materials',
        'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
        'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate',
        'WELL': 'Real Estate', 'DLR': 'Real Estate', 'O': 'Real Estate',
        'VICI': 'Real Estate',
    }

def calculate_kst(prices, roc_periods=(10,15,20,30), sma_periods=(10,10,10,15), weights=(1,2,3,4)):
    if len(prices) < max(roc_periods) + max(sma_periods) + 10: return None, None, None
    kst = pd.Series(0.0, index=prices.index)
    for rp, sp, w in zip(roc_periods, sma_periods, weights):
        roc = ((prices - prices.shift(rp)) / prices.shift(rp)) * 100
        kst += roc.rolling(window=sp).mean() * w
    signal = kst.ewm(span=9, adjust=False).mean()
    return kst, signal, kst.iloc[-1] if not pd.isna(kst.iloc[-1]) else None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def calculate_sma(prices, period): return prices.rolling(window=period).mean()

def calculate_vma(df, length=10):
    price = df['Close'].values
    n = len(price)
    tmp1, tmp2 = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        if price[i] > price[i-1]: tmp1[i] = price[i] - price[i-1]
        if price[i-1] > price[i]: tmp2[i] = price[i-1] - price[i]
    d2 = pd.Series(tmp1).rolling(window=length).sum().values
    d4 = pd.Series(tmp2).rolling(window=length).sum().values
    ad3 = np.zeros(n)
    for i in range(n):
        if d2[i] + d4[i] != 0: ad3[i] = (d2[i] - d4[i]) / (d2[i] + d4[i]) * 100
    coeff = (2 / (length + 1)) * np.abs(ad3) / 100
    vma = np.zeros(n)
    vma[0] = price[0]
    for i in range(1, n):
        if np.isnan(coeff[i]): coeff[i] = 0
        vma[i] = coeff[i] * price[i] + vma[i-1] * (1 - coeff[i])
    return pd.Series(vma, index=df.index)

def calculate_vwma(df, period):
    return (df['Close'] * df['Volume']).rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

def calculate_stage(df, lookback_days=5):
    if len(df) < 50: return None
    vma = calculate_vma(df, 10)
    vwma8, vwma21, vwma34 = calculate_vwma(df, 8), calculate_vwma(df, 21), calculate_vwma(df, 34)
    df_a = df.copy()
    df_a['VMA'], df_a['VWMA8'], df_a['VWMA21'], df_a['VWMA34'] = vma, vwma8, vwma21, vwma34
    bull = (df_a['VWMA8'] > df_a['VWMA21']) & (df_a['VWMA21'] > df_a['VWMA34'])
    bear = (df_a['VWMA8'] < df_a['VWMA21']) & (df_a['VWMA21'] < df_a['VWMA34'])
    df_a['Stage'] = 'Distribution'
    df_a.loc[bull & (df_a['Close'] >= df_a['VMA']), 'Stage'] = 'Acceleration'
    df_a.loc[~bull & ~bear & (df_a['Close'] >= df_a['VMA']), 'Stage'] = 'Accumulation'
    df_a.loc[bear & (df_a['Close'] <= df_a['VMA']), 'Stage'] = 'Deceleration'
    transition, days = None, None
    for i in range(1, min(lookback_days + 1, len(df_a))):
        prev, curr = df_a['Stage'].iloc[-(i+1)], df_a['Stage'].iloc[-i]
        if prev != curr:
            if curr == 'Acceleration': transition, days = 'ENTERING_ACCELERATION', i; break
            elif curr == 'Deceleration': transition, days = 'ENTERING_DECELERATION', i; break
    return {'Stage': df_a['Stage'].iloc[-1], 'Price': float(df_a['Close'].iloc[-1]),
            'VMA': float(df_a['VMA'].iloc[-1]),
            'Price_vs_VMA': float(((df_a['Close'].iloc[-1] - df_a['VMA'].iloc[-1]) / df_a['VMA'].iloc[-1]) * 100),
            'Bullish_Stack': bool(df_a['VWMA8'].iloc[-1] > df_a['VWMA21'].iloc[-1] > df_a['VWMA34'].iloc[-1]),
            'Bearish_Stack': bool(df_a['VWMA8'].iloc[-1] < df_a['VWMA21'].iloc[-1] < df_a['VWMA34'].iloc[-1]),
            'Transition': transition, 'Transition_Days_Ago': days}

def get_stock_data(ticker, days=400):
    cache_key = f"{ticker}_{days}"
    now = datetime.now()
    if cache_key in DATA_CACHE:
        cached_time, cached_data = DATA_CACHE[cache_key]
        if (now - cached_time).seconds < CACHE_TIMEOUT: return cached_data
    try:
        df = yf.Ticker(ticker).history(start=now - timedelta(days=days), end=now)
        if df is not None and len(df) > 50: DATA_CACHE[cache_key] = (now, df); return df
        return None
    except: return None

def get_stock_kst(ticker, days=400):
    try:
        df = get_stock_data(ticker, days)
        if df is None or len(df) < 100: return (ticker, None, None, False)
        kst_s, sig_s, kst = calculate_kst(df['Close'])
        if kst is None: return (ticker, None, None, False)
        return (ticker, kst, sig_s.iloc[-1] if sig_s is not None else None, True)
    except: return (ticker, None, None, False)

def fetch_all_stocks_parallel(days=400, max_workers=10):
    stocks = get_sp500_stocks()
    all_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data, t, days): t for t in stocks.keys()}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and len(data) > 50: all_data[ticker] = data
    return all_data

def run_standard_analysis():
    stocks = get_sp500_stocks()
    results, failed = [], []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_stock_kst, t): t for t in stocks.keys()}
        for future in as_completed(futures):
            ticker, kst, signal, success = future.result()
            if success and kst is not None:
                results.append({'Ticker': ticker, 'Sector': stocks.get(ticker, 'Unknown'),
                    'KST': round(kst, 2), 'Signal': round(signal, 2) if signal else None,
                    'Above_Zero': kst > 0, 'Above_Signal': kst > signal if signal else False})
            else: failed.append(ticker)
    if not results: return {'error': 'No data retrieved', 'failed_count': len(failed)}
    df = pd.DataFrame(results)
    sector_breadth = df.groupby('Sector').agg(
        Total_Stocks=('Ticker', 'count'), Above_Zero=('Above_Zero', 'sum'),
        Pct_Above_Zero=('Above_Zero', lambda x: (x.sum() / len(x)) * 100),
        Avg_KST=('KST', 'mean')).round(1).sort_values('Pct_Above_Zero', ascending=False).reset_index()
    return {'stocks': df.to_dict('records'), 'sectors': sector_breadth.to_dict('records'),
        'overall_pct': round((df['Above_Zero'].sum() / len(df)) * 100, 1),
        'overall_avg': round(df['KST'].mean(), 1), 'total_stocks': len(df), 'failed_count': len(failed)}

def run_52week_trend():
    stocks = get_sp500_stocks()
    sectors = sorted(list(set(stocks.values())))
    all_data = fetch_all_stocks_parallel(days=400, max_workers=10)
    weekly_dates = pd.date_range(end=datetime.now(), periods=26, freq='W')
    sector_breadth = {s: [] for s in sectors}
    for week_end in weekly_dates:
        counts = {s: {'above': 0, 'total': 0} for s in sectors}
        for ticker, df in all_data.items():
            sector = stocks.get(ticker, 'Unknown')
            if sector == 'Unknown': continue
            try: df_week = df[df.index.tz_localize(None) <= week_end]
            except: df_week = df[df.index <= week_end]
            if len(df_week) < 60: continue
            _, _, kst = calculate_kst(df_week['Close'])
            if kst is not None:
                counts[sector]['total'] += 1
                if kst > 0: counts[sector]['above'] += 1
        for s in sectors:
            pct = (counts[s]['above'] / counts[s]['total'] * 100) if counts[s]['total'] > 0 else 0
            sector_breadth[s].append(pct)
    breadth_df = pd.DataFrame(sector_breadth, index=weekly_dates)
    ema_df = breadth_df.ewm(span=8, adjust=False).mean()
    trends = sorted([{'sector': s, 'current': round(float(breadth_df[s].iloc[-1]), 1),
        'ema': round(float(ema_df[s].iloc[-1]), 1),
        'diff': round(float(breadth_df[s].iloc[-1] - ema_df[s].iloc[-1]), 1),
        'signal': 'BULLISH' if breadth_df[s].iloc[-1] > ema_df[s].iloc[-1] else 'BEARISH',
        'history': [round(x, 1) for x in breadth_df[s].tolist()],
        'ema_history': [round(x, 1) for x in ema_df[s].tolist()]
    } for s in sectors], key=lambda x: x['current'], reverse=True)
    return {'trends': trends, 'dates': [d.strftime('%Y-%m-%d') for d in weekly_dates]}

def run_stage_analysis():
    stocks = get_sp500_stocks()
    sectors = sorted(list(set(stocks.values())))
    all_data = fetch_all_stocks_parallel(days=200, max_workers=10)
    results = []
    for ticker, df in all_data.items():
        sector = stocks.get(ticker, 'Unknown')
        if sector == 'Unknown': continue
        stage = calculate_stage(df, 5)
        if stage: results.append({'Ticker': ticker, 'Sector': sector, **stage})
    df_stages = pd.DataFrame(results)
    summary = []
    for s in sectors:
        sd = df_stages[df_stages['Sector'] == s]
        total = len(sd)
        if total == 0: continue
        accel = len(sd[sd['Stage'] == 'Acceleration'])
        accum = len(sd[sd['Stage'] == 'Accumulation'])
        distrib = len(sd[sd['Stage'] == 'Distribution'])
        decel = len(sd[sd['Stage'] == 'Deceleration'])
        summary.append({'Sector': s, 'Total': total, 'Accel': accel, 'Accel_Pct': round(accel/total*100, 1),
            'Accum': accum, 'Accum_Pct': round(accum/total*100, 1), 'Distrib': distrib, 'Distrib_Pct': round(distrib/total*100, 1),
            'Decel': decel, 'Decel_Pct': round(decel/total*100, 1),
            'Bullish_Pct': round((accel+accum)/total*100, 1), 'Bearish_Pct': round((distrib+decel)/total*100, 1)})
    total = len(df_stages)
    if total == 0: return {'error': 'No data retrieved'}
    return {'stocks': df_stages.to_dict('records'),
        'sectors': sorted(summary, key=lambda x: x['Bullish_Pct'], reverse=True),
        'entering_accel': df_stages[df_stages['Transition'] == 'ENTERING_ACCELERATION'].to_dict('records'),
        'entering_decel': df_stages[df_stages['Transition'] == 'ENTERING_DECELERATION'].to_dict('records'),
        'overall': {'Accel_Pct': round(len(df_stages[df_stages['Stage']=='Acceleration'])/total*100, 1),
            'Accum_Pct': round(len(df_stages[df_stages['Stage']=='Accumulation'])/total*100, 1),
            'Distrib_Pct': round(len(df_stages[df_stages['Stage']=='Distribution'])/total*100, 1),
            'Decel_Pct': round(len(df_stages[df_stages['Stage']=='Deceleration'])/total*100, 1)},
        'total_stocks': total}

def run_sector_drilldown(sector_name):
    stocks = get_sp500_stocks()
    sector_tickers = [t for t, s in stocks.items() if s == sector_name]
    all_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_stock_data, t, 400): t for t in sector_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and len(data) > 60: all_data[ticker] = data
    analysis = []
    for ticker, df in all_data.items():
        kst_s, sig_s, kst = calculate_kst(df['Close'])
        if kst_s is None: continue
        sig = sig_s.iloc[-1]
        kst_1w = kst_s.iloc[-5] if len(kst_s) > 5 else kst
        kst_4w = kst_s.iloc[-20] if len(kst_s) > 20 else kst
        prev_kst, prev_sig = kst_s.iloc[-2] if len(kst_s) > 1 else kst, sig_s.iloc[-2] if len(sig_s) > 1 else sig
        if (prev_kst <= prev_sig) and (kst > sig): status = "BULLISH CROSS"
        elif (prev_kst >= prev_sig) and (kst < sig): status = "BEARISH CROSS"
        elif kst > sig: status = "Above Signal"
        else: status = "Below Signal"
        analysis.append({'Ticker': ticker, 'KST': round(float(kst), 2), 'Signal': round(float(sig), 2),
            'KST_Momentum_4W': round(float(kst - kst_4w), 2), 'KST_Momentum_1W': round(float(kst - kst_1w), 2),
            'Crossover_Status': status, 'Above_Zero': kst > 0})
    if not analysis: return {'error': f'No data for {sector_name}', 'sector': sector_name}
    df_a = pd.DataFrame(analysis)
    return {'sector': sector_name, 'all_stocks': analysis,
        'improving': df_a.nlargest(10, 'KST_Momentum_4W').to_dict('records'),
        'weakening': df_a.nsmallest(10, 'KST_Momentum_4W').to_dict('records'), 'total': len(analysis)}

def run_ml_backtest(years=1, holding_days=10, return_threshold=0.05):
    if not ML_AVAILABLE: return {'error': 'ML not available'}
    stocks = get_sp500_stocks()
    all_data = fetch_all_stocks_parallel(days=years*365+100, max_workers=10)
    features = []
    for ticker, df in all_data.items():
        sector = stocks.get(ticker, 'Unknown')
        if sector == 'Unknown': continue
        kst_s, sig_s, _ = calculate_kst(df['Close'])
        if kst_s is None: continue
        df = df.copy()
        df['KST'], df['KST_Signal'] = kst_s, sig_s
        df['RSI'] = calculate_rsi(df['Close'])
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        df['Forward_Return'] = df['Close'].shift(-holding_days) / df['Close'] - 1
        df['KST_Momentum'] = df['KST'] - df['KST'].shift(20)
        for i in range(100, len(df) - holding_days):
            r = df.iloc[i]
            if any(pd.isna([r['KST'], r['RSI'], r['Price_vs_SMA50'], r['Forward_Return']])): continue
            features.append({'Ticker': ticker, 'Sector': sector, 'KST': float(r['KST']),
                'KST_Signal': float(r['KST_Signal']), 'KST_Above_Zero': 1 if r['KST'] > 0 else 0,
                'KST_Above_Signal': 1 if r['KST'] > r['KST_Signal'] else 0,
                'KST_Signal_Diff': float(r['KST'] - r['KST_Signal']),
                'KST_Momentum': float(r['KST_Momentum']) if not pd.isna(r['KST_Momentum']) else 0,
                'RSI': float(r['RSI']), 'Price_vs_SMA50': float(r['Price_vs_SMA50']),
                'Forward_Return': float(r['Forward_Return'])})
    if not features: return {'error': 'Not enough data for ML backtest'}
    df_f = pd.DataFrame(features)
    df_f['Target'] = (df_f['Forward_Return'] > return_threshold).astype(int)
    cols = ['KST', 'KST_Signal', 'KST_Above_Zero', 'KST_Above_Signal', 'KST_Signal_Diff', 'KST_Momentum', 'RSI', 'Price_vs_SMA50']
    X, y = df_f[cols].values, df_f['Target'].values
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50,
        min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    importance = sorted(zip(cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    latest = df_f.groupby('Ticker').last().reset_index()
    latest['Pred_Proba'] = model.predict_proba(latest[cols].values)[:, 1]
    top = latest.nlargest(20, 'Pred_Proba')[['Ticker', 'Sector', 'Pred_Proba', 'KST', 'RSI', 'Price_vs_SMA50']].to_dict('records')
    df_test = df_f.iloc[split:].copy()
    df_test['Pred_Proba'] = y_proba
    high_conf = df_test[df_test['Pred_Proba'] > 0.60]
    return {'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1), 'samples_train': len(X_train),
        'samples_test': len(X_test), 'feature_importance': [{'feature': f, 'importance': round(i, 3)} for f, i in importance],
        'top_picks': top, 'high_conf_count': len(high_conf),
        'high_conf_avg_return': round(high_conf['Forward_Return'].mean()*100, 2) if len(high_conf) > 0 else 0,
        'high_conf_win_rate': round((high_conf['Forward_Return'] > 0).mean()*100, 1) if len(high_conf) > 0 else 0,
        'target_distribution': {'strong': int(df_f['Target'].sum()), 'other': int(len(df_f) - df_f['Target'].sum())}}

@app.route('/')
def index(): return render_template('index.html', ml_available=ML_AVAILABLE)

@app.route('/api/standard-analysis')
def api_standard():
    try: return jsonify(run_standard_analysis())
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/52week-trend')
def api_52week():
    try: return jsonify(run_52week_trend())
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/stage-analysis')
def api_stage():
    try: return jsonify(run_stage_analysis())
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/sector-drilldown/<sector>')
def api_drilldown(sector):
    try: return jsonify(run_sector_drilldown(sector))
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/ml-backtest')
def api_ml():
    if not ML_AVAILABLE: return jsonify({'error': 'ML not available'}), 400
    try:
        years = int(request.args.get('years', 1))
        holding = int(request.args.get('holding_days', 10))
        thresh = float(request.args.get('threshold', 0.05))
        return jsonify(run_ml_backtest(years, holding, thresh))
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/sectors')
def api_sectors(): return jsonify({'sectors': sorted(list(set(get_sp500_stocks().values())))})

@app.route('/api/test')
def api_test():
    try:
        df = yf.Ticker('AAPL').history(period='5d')
        if df is not None and len(df) > 0:
            return jsonify({'status': 'OK', 'message': 'yfinance working', 'aapl_price': round(float(df['Close'].iloc[-1]), 2)})
        return jsonify({'status': 'ERROR', 'message': 'No data returned'})
    except Exception as e: return jsonify({'status': 'ERROR', 'message': str(e)})

if __name__ == '__main__': app.run(debug=True)
