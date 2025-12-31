"""
S&P 500 KST Sector Breadth Analysis - PythonAnywhere Flask App
==============================================================
Optimized for PythonAnywhere Free Tier (~400MB)
- ML is optional (set ENABLE_ML = True/False)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_ML = True  # Set to False to save ~150MB

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
CACHE_TIMEOUT = 300


# =============================================================================
# S&P 500 STOCK DATA
# =============================================================================

def get_sp500_stocks():
    """Returns dictionary of S&P 500 stocks with their sectors."""
    return {
        # Information Technology
        'NVDA': 'Information Technology', 'AAPL': 'Information Technology', 
        'MSFT': 'Information Technology', 'AVGO': 'Information Technology',
        'ORCL': 'Information Technology', 'CRM': 'Information Technology',
        'AMD': 'Information Technology', 'CSCO': 'Information Technology',
        'ACN': 'Information Technology', 'ADBE': 'Information Technology',
        'IBM': 'Information Technology', 'INTC': 'Information Technology',
        'TXN': 'Information Technology', 'QCOM': 'Information Technology',
        'INTU': 'Information Technology', 'AMAT': 'Information Technology',
        'MU': 'Information Technology', 'LRCX': 'Information Technology',
        'NOW': 'Information Technology', 'PANW': 'Information Technology',
        'KLAC': 'Information Technology', 'SNPS': 'Information Technology',
        'CDNS': 'Information Technology', 'ADSK': 'Information Technology',
        'CRWD': 'Information Technology', 'ADI': 'Information Technology',
        'APH': 'Information Technology', 'ANET': 'Information Technology',
        'FTNT': 'Information Technology', 'MSI': 'Information Technology',
        'MCHP': 'Information Technology', 'HPE': 'Information Technology',
        'NXPI': 'Information Technology', 'ROP': 'Information Technology',
        'KEYS': 'Information Technology', 'FSLR': 'Information Technology',
        'MPWR': 'Information Technology', 'PLTR': 'Information Technology',
        'ADP': 'Information Technology', 'GLW': 'Information Technology', 
        'FISV': 'Information Technology', 'FIS': 'Information Technology',
        'STX': 'Information Technology', 'WDC': 'Information Technology',
        'HPQ': 'Information Technology', 'PTC': 'Information Technology',
        'ON': 'Information Technology', 'NTAP': 'Information Technology',
        'FFIV': 'Information Technology', 'AKAM': 'Information Technology',
        'SMCI': 'Information Technology', 'WDAY': 'Information Technology', 
        'FICO': 'Information Technology', 'TTD': 'Information Technology', 
        'CDW': 'Information Technology', 'IT': 'Information Technology',
        'VRSN': 'Information Technology', 'TRMB': 'Information Technology',
        'TYL': 'Information Technology', 'GDDY': 'Information Technology',
        'GEN': 'Information Technology', 'CTSH': 'Information Technology',
        'DELL': 'Information Technology', 'PYPL': 'Information Technology',
        'ZBRA': 'Information Technology', 'SWKS': 'Information Technology',
        'AXON': 'Information Technology',
        # Financials
        'BRK-B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
        'MA': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
        'AXP': 'Financials', 'SPGI': 'Financials', 'BLK': 'Financials',
        'SCHW': 'Financials', 'PGR': 'Financials', 'CB': 'Financials',
        'BX': 'Financials', 'CME': 'Financials', 'ICE': 'Financials',
        'MMC': 'Financials', 'KKR': 'Financials', 'USB': 'Financials',
        'PNC': 'Financials', 'MCO': 'Financials', 'AON': 'Financials',
        'TRV': 'Financials', 'TFC': 'Financials', 'AJG': 'Financials',
        'AIG': 'Financials', 'MET': 'Financials', 'NDAQ': 'Financials',
        'ALL': 'Financials', 'AFL': 'Financials', 'COF': 'Financials',
        'PRU': 'Financials', 'AMP': 'Financials', 'HIG': 'Financials',
        'BK': 'Financials', 'STT': 'Financials', 'ACGL': 'Financials',
        'APO': 'Financials', 'ARES': 'Financials', 'MSCI': 'Financials',
        'CBOE': 'Financials', 'RJF': 'Financials', 'PAYX': 'Financials', 
        'VRSK': 'Financials', 'BR': 'Financials', 'WTW': 'Financials', 
        'NTRS': 'Financials', 'FITB': 'Financials', 'MTB': 'Financials', 
        'SYF': 'Financials', 'IBKR': 'Financials', 'CFG': 'Financials', 
        'HBAN': 'Financials', 'RF': 'Financials', 'BRO': 'Financials', 
        'CINF': 'Financials', 'KEY': 'Financials', 'CPAY': 'Financials', 
        'TROW': 'Financials', 'GPN': 'Financials', 'WRB': 'Financials', 
        'JKHY': 'Financials', 'PFG': 'Financials', 'L': 'Financials', 
        'AIZ': 'Financials', 'IVZ': 'Financials', 'GL': 'Financials', 
        'FDS': 'Financials', 'BEN': 'Financials', 'ERIE': 'Financials',
        # Health Care
        'LLY': 'Health Care', 'UNH': 'Health Care', 'JNJ': 'Health Care',
        'ABBV': 'Health Care', 'MRK': 'Health Care', 'TMO': 'Health Care',
        'ABT': 'Health Care', 'ISRG': 'Health Care', 'AMGN': 'Health Care',
        'DHR': 'Health Care', 'PFE': 'Health Care', 'BSX': 'Health Care',
        'GILD': 'Health Care', 'VRTX': 'Health Care', 'MDT': 'Health Care',
        'SYK': 'Health Care', 'BMY': 'Health Care', 'ELV': 'Health Care',
        'CI': 'Health Care', 'HCA': 'Health Care', 'MCK': 'Health Care',
        'CVS': 'Health Care', 'REGN': 'Health Care', 'BDX': 'Health Care',
        'ZTS': 'Health Care', 'EW': 'Health Care', 'IDXX': 'Health Care',
        'CAH': 'Health Care', 'COR': 'Health Care', 'IQV': 'Health Care',
        'GEHC': 'Health Care', 'RMD': 'Health Care', 'HUM': 'Health Care',
        'A': 'Health Care', 'MTD': 'Health Care', 'DXCM': 'Health Care',
        'STE': 'Health Care', 'COO': 'Health Care', 'HOLX': 'Health Care',
        'BIIB': 'Health Care', 'INCY': 'Health Care', 'WAT': 'Health Care',
        'LH': 'Health Care', 'PODD': 'Health Care', 'WST': 'Health Care',
        'DGX': 'Health Care', 'ALGN': 'Health Care', 'ZBH': 'Health Care',
        'RVTY': 'Health Care', 'VTRS': 'Health Care', 'CNC': 'Health Care',
        'MOH': 'Health Care', 'UHS': 'Health Care', 'CRL': 'Health Care',
        'DVA': 'Health Care', 'TECH': 'Health Care', 'BAX': 'Health Care',
        'HSIC': 'Health Care', 'MRNA': 'Health Care',
        # Communication Services
        'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
        'META': 'Communication Services', 'NFLX': 'Communication Services',
        'DIS': 'Communication Services', 'T': 'Communication Services',
        'VZ': 'Communication Services', 'TMUS': 'Communication Services',
        'CMCSA': 'Communication Services', 'WBD': 'Communication Services',
        'CHTR': 'Communication Services', 'EA': 'Communication Services',
        'TTWO': 'Communication Services', 'LYV': 'Communication Services',
        'OMC': 'Communication Services', 'FOXA': 'Communication Services',
        'FOX': 'Communication Services', 'NWSA': 'Communication Services',
        'NWS': 'Communication Services', 'MTCH': 'Communication Services',
        'UBER': 'Communication Services',
        # Consumer Discretionary  
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
        'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
        'BKNG': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
        'LOW': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
        'NKE': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
        'ORLY': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',
        'RCL': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary',
        'YUM': 'Consumer Discretionary', 'HLT': 'Consumer Discretionary',
        'ROST': 'Consumer Discretionary', 'DHI': 'Consumer Discretionary',
        'CVNA': 'Consumer Discretionary', 'ABNB': 'Consumer Discretionary',
        'AZO': 'Consumer Discretionary', 'CPRT': 'Consumer Discretionary',
        'TGT': 'Consumer Discretionary', 'F': 'Consumer Discretionary',
        'EBAY': 'Consumer Discretionary', 'CCL': 'Consumer Discretionary',
        'EXPE': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary',
        'GRMN': 'Consumer Discretionary', 'DG': 'Consumer Discretionary',
        'ULTA': 'Consumer Discretionary', 'TSCO': 'Consumer Discretionary',
        'TPR': 'Consumer Discretionary', 'WSM': 'Consumer Discretionary',
        'LULU': 'Consumer Discretionary', 'PHM': 'Consumer Discretionary',
        'DLTR': 'Consumer Discretionary', 'LEN': 'Consumer Discretionary',
        'GPC': 'Consumer Discretionary', 'LVS': 'Consumer Discretionary',
        'NVR': 'Consumer Discretionary', 'BBY': 'Consumer Discretionary',
        'POOL': 'Consumer Discretionary', 'DPZ': 'Consumer Discretionary',
        'WYNN': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary',
        'APTV': 'Consumer Discretionary', 'NCLH': 'Consumer Discretionary',
        'HAS': 'Consumer Discretionary', 'RL': 'Consumer Discretionary',
        'MAS': 'Consumer Discretionary', 'BLDR': 'Consumer Discretionary',
        'DECK': 'Consumer Discretionary',
        # Industrials
        'GE': 'Industrials', 'CAT': 'Industrials', 'RTX': 'Industrials',
        'UNP': 'Industrials', 'HON': 'Industrials', 'BA': 'Industrials',
        'DE': 'Industrials', 'ETN': 'Industrials', 'LMT': 'Industrials',
        'PH': 'Industrials', 'TT': 'Industrials', 'GD': 'Industrials',
        'WM': 'Industrials', 'NOC': 'Industrials', 'MMM': 'Industrials',
        'EMR': 'Industrials', 'ITW': 'Industrials', 'JCI': 'Industrials',
        'TDG': 'Industrials', 'UPS': 'Industrials', 'CMI': 'Industrials',
        'CSX': 'Industrials', 'ECL': 'Industrials', 'NSC': 'Industrials',
        'CTAS': 'Industrials', 'PWR': 'Industrials', 'FDX': 'Industrials',
        'AME': 'Industrials', 'FAST': 'Industrials', 'DAL': 'Industrials',
        'GWW': 'Industrials', 'RSG': 'Industrials', 'CARR': 'Industrials',
        'HWM': 'Industrials', 'LHX': 'Industrials', 'URI': 'Industrials',
        'PCAR': 'Industrials', 'TEL': 'Industrials', 'ODFL': 'Industrials',
        'WAB': 'Industrials', 'UAL': 'Industrials', 'DOV': 'Industrials',
        'EFX': 'Industrials', 'OTIS': 'Industrials', 'XYL': 'Industrials', 
        'VLTO': 'Industrials', 'HUBB': 'Industrials', 'TDY': 'Industrials', 
        'LDOS': 'Industrials', 'EXPD': 'Industrials', 'LUV': 'Industrials', 
        'SNA': 'Industrials', 'ROL': 'Industrials', 'FTV': 'Industrials', 
        'PNR': 'Industrials', 'IEX': 'Industrials', 'NDSN': 'Industrials', 
        'EME': 'Industrials', 'J': 'Industrials', 'TXT': 'Industrials', 
        'JBHT': 'Industrials', 'HII': 'Industrials', 'ALLE': 'Industrials', 
        'AOS': 'Industrials', 'SWK': 'Industrials', 'CHRW': 'Industrials', 
        'GNRC': 'Industrials', 'LII': 'Industrials',
        # Consumer Staples
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 
        'COST': 'Consumer Staples', 'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples', 'PM': 'Consumer Staples',
        'MDLZ': 'Consumer Staples', 'MO': 'Consumer Staples',
        'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
        'SYY': 'Consumer Staples', 'KVUE': 'Consumer Staples',
        'GIS': 'Consumer Staples', 'ADM': 'Consumer Staples',
        'HSY': 'Consumer Staples', 'STZ': 'Consumer Staples',
        'KDP': 'Consumer Staples', 'KR': 'Consumer Staples',
        'MNST': 'Consumer Staples', 'MKC': 'Consumer Staples',
        'KHC': 'Consumer Staples', 'CHD': 'Consumer Staples',
        'EL': 'Consumer Staples', 'TSN': 'Consumer Staples',
        'CLX': 'Consumer Staples', 'BG': 'Consumer Staples',
        'SJM': 'Consumer Staples', 'CAG': 'Consumer Staples',
        'TAP': 'Consumer Staples', 'HRL': 'Consumer Staples',
        'LW': 'Consumer Staples', 'CPB': 'Consumer Staples',
        'BF-B': 'Consumer Staples',
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
        'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy',
        'WMB': 'Energy', 'KMI': 'Energy', 'HES': 'Energy',
        'DVN': 'Energy', 'FANG': 'Energy', 'HAL': 'Energy',
        'BKR': 'Energy', 'TRGP': 'Energy', 'OKE': 'Energy',
        'EQT': 'Energy', 'APA': 'Energy', 'CTRA': 'Energy',
        # Utilities
        'NEE': 'Utilities', 'SO': 'Utilities', 'DUK': 'Utilities',
        'CEG': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities',
        'D': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities',
        'PEG': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities',
        'ETR': 'Utilities', 'VST': 'Utilities', 'NRG': 'Utilities',
        'PCG': 'Utilities', 'AWK': 'Utilities', 'AEE': 'Utilities',
        'DTE': 'Utilities', 'PPL': 'Utilities', 'ES': 'Utilities',
        'FE': 'Utilities', 'EIX': 'Utilities', 'CMS': 'Utilities',
        'CNP': 'Utilities', 'LNT': 'Utilities', 'EVRG': 'Utilities',
        'NI': 'Utilities', 'ATO': 'Utilities', 'PNW': 'Utilities',
        'AES': 'Utilities',
        # Materials
        'LIN': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
        'NEM': 'Materials', 'APD': 'Materials', 'CTVA': 'Materials',
        'CRH': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
        'NUE': 'Materials', 'PPG': 'Materials', 'STLD': 'Materials', 
        'IFF': 'Materials', 'ALB': 'Materials', 'DD': 'Materials', 
        'DOW': 'Materials', 'AVY': 'Materials', 'BALL': 'Materials', 
        'CF': 'Materials', 'IP': 'Materials', 'SW': 'Materials', 
        'PKG': 'Materials', 'AMCR': 'Materials', 'MOS': 'Materials', 
        'LYB': 'Materials', 'WY': 'Materials',
        # Real Estate
        'PLD': 'Real Estate', 'AMT': 'Real Estate', 'EQIX': 'Real Estate',
        'WELL': 'Real Estate', 'SPG': 'Real Estate', 'DLR': 'Real Estate',
        'PSA': 'Real Estate', 'O': 'Real Estate', 'CCI': 'Real Estate',
        'CBRE': 'Real Estate', 'VICI': 'Real Estate', 'EXR': 'Real Estate',
        'VTR': 'Real Estate', 'AVB': 'Real Estate', 'IRM': 'Real Estate',
        'EQR': 'Real Estate', 'SBAC': 'Real Estate', 'ESS': 'Real Estate', 
        'MAA': 'Real Estate', 'UDR': 'Real Estate', 'CPT': 'Real Estate', 
        'HST': 'Real Estate', 'INVH': 'Real Estate', 'REG': 'Real Estate', 
        'KIM': 'Real Estate', 'BXP': 'Real Estate', 'DOC': 'Real Estate', 
        'ARE': 'Real Estate', 'FRT': 'Real Estate', 'CSGP': 'Real Estate',
    }


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_kst(prices, roc_periods=(10, 15, 20, 30), 
                  sma_periods=(10, 10, 10, 15), weights=(1, 2, 3, 4)):
    if len(prices) < max(roc_periods) + max(sma_periods) + 10:
        return None, None, None
    kst_series = pd.Series(0.0, index=prices.index)
    for roc_period, sma_period, weight in zip(roc_periods, sma_periods, weights):
        roc = ((prices - prices.shift(roc_period)) / prices.shift(roc_period)) * 100
        smoothed_roc = roc.rolling(window=sma_period).mean()
        kst_series += smoothed_roc * weight
    signal_line = kst_series.ewm(span=9, adjust=False).mean()
    current_kst = kst_series.iloc[-1] if not pd.isna(kst_series.iloc[-1]) else None
    return kst_series, signal_line, current_kst


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices, period):
    return prices.rolling(window=period).mean()


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
    return (df['Close'] * df['Volume']).rolling(window=period).sum() / \
           df['Volume'].rolling(window=period).sum()


def calculate_stage(df, lookback_days=5):
    if len(df) < 50: return None
    vma = calculate_vma(df, length=10)
    vwma8, vwma21, vwma34 = calculate_vwma(df, 8), calculate_vwma(df, 21), calculate_vwma(df, 34)
    df_a = df.copy()
    df_a['VMA'], df_a['VWMA8'], df_a['VWMA21'], df_a['VWMA34'] = vma, vwma8, vwma21, vwma34
    bull = (df_a['VWMA8'] > df_a['VWMA21']) & (df_a['VWMA21'] > df_a['VWMA34'])
    bear = (df_a['VWMA8'] < df_a['VWMA21']) & (df_a['VWMA21'] < df_a['VWMA34'])
    df_a['Stage'] = 'Distribution'
    df_a.loc[bull & (df_a['Close'] >= df_a['VMA']), 'Stage'] = 'Acceleration'
    df_a.loc[~bull & ~bear & (df_a['Close'] >= df_a['VMA']), 'Stage'] = 'Accumulation'
    df_a.loc[bear & (df_a['Close'] <= df_a['VMA']), 'Stage'] = 'Deceleration'
    transition, transition_days = None, None
    for i in range(1, min(lookback_days + 1, len(df_a))):
        prev, curr = df_a['Stage'].iloc[-(i+1)], df_a['Stage'].iloc[-i]
        if prev != curr:
            if curr == 'Acceleration': transition, transition_days = 'ENTERING_ACCELERATION', i; break
            elif curr == 'Deceleration': transition, transition_days = 'ENTERING_DECELERATION', i; break
    return {
        'Stage': df_a['Stage'].iloc[-1], 'Price': float(df_a['Close'].iloc[-1]),
        'VMA': float(df_a['VMA'].iloc[-1]),
        'Price_vs_VMA': float(((df_a['Close'].iloc[-1] - df_a['VMA'].iloc[-1]) / df_a['VMA'].iloc[-1]) * 100),
        'Bullish_Stack': bool(df_a['VWMA8'].iloc[-1] > df_a['VWMA21'].iloc[-1] > df_a['VWMA34'].iloc[-1]),
        'Bearish_Stack': bool(df_a['VWMA8'].iloc[-1] < df_a['VWMA21'].iloc[-1] < df_a['VWMA34'].iloc[-1]),
        'Transition': transition, 'Transition_Days_Ago': transition_days
    }


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_stock_data(ticker, days=400):
    cache_key = f"{ticker}_{days}"
    now = datetime.now()
    if cache_key in DATA_CACHE:
        cached_time, cached_data = DATA_CACHE[cache_key]
        if (now - cached_time).seconds < CACHE_TIMEOUT: return cached_data
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=now - timedelta(days=days), end=now)
        if df is not None and len(df) > 50: 
            DATA_CACHE[cache_key] = (now, df)
            return df
        return None
    except Exception as e:
        return None


def get_stock_kst(ticker, days=400):
    try:
        df = get_stock_data(ticker, days)
        if df is None or len(df) < 100: 
            return (ticker, None, None, False)
        kst_series, signal_line, current_kst = calculate_kst(df['Close'])
        if current_kst is None:
            return (ticker, None, None, False)
        return (ticker, current_kst, signal_line.iloc[-1] if signal_line is not None else None, True)
    except Exception as e:
        return (ticker, None, None, False)


def fetch_all_stocks_parallel(days=400, max_workers=8):
    stocks = get_sp500_stocks()
    all_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data, t, days): t for t in stocks.keys()}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and len(data) > 100: all_data[ticker] = data
    return all_data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_standard_analysis():
    stocks = get_sp500_stocks()
    results, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_stock_kst, t): t for t in stocks.keys()}
        for future in as_completed(futures):
            ticker, kst, signal, success = future.result()
            if success and kst is not None:
                results.append({'Ticker': ticker, 'Sector': stocks.get(ticker, 'Unknown'),
                    'KST': round(kst, 2), 'Signal': round(signal, 2) if signal else None,
                    'Above_Zero': kst > 0, 'Above_Signal': kst > signal if signal else False})
            else: failed.append(ticker)
    if not results: 
        return {'error': 'No data retrieved', 'failed_count': len(failed)}
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
    all_data = fetch_all_stocks_parallel(days=500, max_workers=8)
    weekly_dates = pd.date_range(end=datetime.now(), periods=52, freq='W')
    sector_breadth = {s: [] for s in sectors}
    for week_end in weekly_dates:
        counts = {s: {'above': 0, 'total': 0} for s in sectors}
        for ticker, df in all_data.items():
            sector = stocks.get(ticker, 'Unknown')
            if sector == 'Unknown': continue
            try: df_week = df[df.index.tz_localize(None) <= week_end]
            except: df_week = df[df.index <= week_end]
            if len(df_week) < 100: continue
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
        'ema_history': [round(x, 1) for x in ema_df[s].tolist()]} for s in sectors],
        key=lambda x: x['current'], reverse=True)
    return {'trends': trends, 'dates': [d.strftime('%Y-%m-%d') for d in weekly_dates]}


def run_stage_analysis():
    stocks = get_sp500_stocks()
    sectors = sorted(list(set(stocks.values())))
    all_data = fetch_all_stocks_parallel(days=200, max_workers=8)
    results = []
    for ticker, df in all_data.items():
        sector = stocks.get(ticker, 'Unknown')
        if sector == 'Unknown': continue
        stage = calculate_stage(df, lookback_days=5)
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
            'Accum': accum, 'Accum_Pct': round(accum/total*100, 1), 'Distrib': distrib,
            'Distrib_Pct': round(distrib/total*100, 1), 'Decel': decel, 'Decel_Pct': round(decel/total*100, 1),
            'Bullish_Pct': round((accel+accum)/total*100, 1), 'Bearish_Pct': round((distrib+decel)/total*100, 1)})
    total = len(df_stages)
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
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_stock_data, t, 500): t for t in sector_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and len(data) > 100: all_data[ticker] = data
    analysis = []
    for ticker, df in all_data.items():
        kst_s, sig_s, kst = calculate_kst(df['Close'])
        if kst_s is None: continue
        sig = sig_s.iloc[-1]
        kst_1w = kst_s.iloc[-5] if len(kst_s) > 5 else kst
        kst_4w = kst_s.iloc[-20] if len(kst_s) > 20 else kst
        prev_kst = kst_s.iloc[-2] if len(kst_s) > 1 else kst
        prev_sig = sig_s.iloc[-2] if len(sig_s) > 1 else sig
        if (prev_kst <= prev_sig) and (kst > sig): status = "BULLISH CROSS"
        elif (prev_kst >= prev_sig) and (kst < sig): status = "BEARISH CROSS"
        elif kst > sig: status = "Above Signal"
        else: status = "Below Signal"
        analysis.append({'Ticker': ticker, 'KST': round(float(kst), 2), 'Signal': round(float(sig), 2),
            'KST_Momentum_4W': round(float(kst - kst_4w), 2), 'KST_Momentum_1W': round(float(kst - kst_1w), 2),
            'Crossover_Status': status, 'Above_Zero': kst > 0})
    df_a = pd.DataFrame(analysis)
    return {'sector': sector_name, 'all_stocks': analysis,
        'improving': df_a.nlargest(10, 'KST_Momentum_4W').to_dict('records'),
        'weakening': df_a.nsmallest(10, 'KST_Momentum_4W').to_dict('records'), 'total': len(analysis)}


def run_ml_backtest(years=1, holding_days=10, return_threshold=0.05):
    if not ML_AVAILABLE: return {'error': 'ML not available'}
    stocks = get_sp500_stocks()
    all_data = fetch_all_stocks_parallel(days=years*365+100, max_workers=8)
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


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html', ml_available=ML_AVAILABLE)

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
def api_sectors():
    return jsonify({'sectors': sorted(list(set(get_sp500_stocks().values())))})


@app.route('/api/test')
def api_test():
    """Test endpoint with just 5 stocks to verify yfinance works."""
    import time
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    results = []
    errors = []
    
    for ticker in test_tickers:
        time.sleep(0.5)  # Delay between requests
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1mo', timeout=10)
            if df is not None and len(df) > 0:
                results.append({
                    'ticker': ticker,
                    'status': 'OK',
                    'rows': len(df),
                    'last_close': round(float(df['Close'].iloc[-1]), 2)
                })
            else:
                errors.append({'ticker': ticker, 'error': 'No data returned'})
        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})
    
    return jsonify({
        'success': results,
        'errors': errors,
        'message': f'{len(results)}/5 stocks retrieved successfully'
    })


if __name__ == '__main__':
    app.run(debug=True)
