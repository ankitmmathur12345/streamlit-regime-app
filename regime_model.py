import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

def analyze_weekly_regime(ticker: str = "TSLA", start_date: str = "2020-01-01", end_date: str = None,
                          n_clusters: int = 5, sticky_kappa: float = 0.85):

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # ✅ Step 1: Download & keep index for resample
    

    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    

    df.columns = ['date','Open','High','Low','Close','Volume']
    df.set_index('date', inplace=True)
    


    df_weekly = df.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
    }).dropna().reset_index()

    df_weekly.columns = ['date','open','high','low','close','volume']


    # Feature Engineering
    df_weekly['logret'] = np.log(df_weekly['close'] / df_weekly['close'].shift(1))
    df_weekly['mom_5w'] = df_weekly['close'].pct_change(5)
    df_weekly['rsi14'] = RSIIndicator(df_weekly['close'], window=14).rsi()
    df_weekly['price_slope_21w'] = df_weekly['close'].rolling(21).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    df_weekly['momentum_accel'] = df_weekly['mom_5w'].diff()
    df_weekly['vol_spike'] = df_weekly['volume'] / df_weekly['volume'].rolling(20).mean()

    def rsi_zone(r):
        if r < 40: return 0
        elif r < 50: return 1
        elif r < 65: return 2
        else: return 3
    df_weekly['rsi_zone_encoded'] = df_weekly['rsi14'].apply(rsi_zone)
    df_weekly.dropna(inplace=True)

    # Feature Selection
    feat_cols = ['logret','mom_5w','price_slope_21w','momentum_accel','vol_spike','rsi_zone_encoded']
    X = StandardScaler().fit_transform(df_weekly[feat_cols])

    # Clustering & Regime Detection
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    init_states = kmeans.fit_predict(X)
    df_weekly['kmeans_state'] = init_states

    means = kmeans.cluster_centers_
    covars = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    for s in range(n_clusters):
        members = X[init_states == s]
        covars[s] = np.cov(members.T) + 1e-6*np.eye(X.shape[1])

    transmat = np.full((n_clusters, n_clusters), (1-sticky_kappa)/(n_clusters-1))
    np.fill_diagonal(transmat, sticky_kappa)
    startprob = np.bincount(init_states, minlength=n_clusters).astype(float)
    startprob /= startprob.sum()

    hmm = GaussianHMM(n_components=n_clusters, covariance_type='full', n_iter=50, init_params='')
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = covars
    hmm.fit(X)

    df_weekly['hmm_state'] = hmm.predict(X)
    state_score = df_weekly.groupby('hmm_state')[['mom_5w','price_slope_21w']].mean().sum(axis=1).sort_values()
    ranked = state_score.index.tolist()
    regime_map = {ranked[0]:'HyperBear', ranked[1]:'Bearish', ranked[2]:'Sideways', ranked[3]:'Bullish', ranked[4]:'HyperBull'}
    df_weekly['regime_smoothed'] = df_weekly['hmm_state'].map(regime_map)

    # Plot
    palette = {'HyperBear':'#ff4d4d', 'Bearish':'#ffa07a', 'Sideways':'#d3d3d3', 'Bullish':'#90ee90', 'HyperBull':'#32cd32'}
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_weekly['date'], df_weekly['close'], color='black', lw=1.2, label='Close')
    for reg, col in palette.items():
        mask = df_weekly['regime_smoothed'] == reg
        if mask.any():
            ax.fill_between(df_weekly['date'], df_weekly['close'].min(), df_weekly['close'].max(),
                            where=mask, color=col, alpha=0.25, label=reg)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax.set_title(f"{ticker.upper()} — Weekly Sticky-HMM Regimes")
    ax.set_xlabel('Date'); ax.set_ylabel('Price')
    ax.grid(True); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Latest Row for Analysis
    latest = df_weekly.iloc[-1]
    rsi = latest['rsi14']
    price = latest['close']
    momentum = latest['mom_5w']
    volume_spike = latest['vol_spike']
    slope = latest['price_slope_21w']
    regime = latest['regime_smoothed']

    # Action Plan Mapping
    regime_action_table = {
        "HyperBull": {
            "action": "Keep holding or buy more if there is more conviction",
            "recommendation": "Monitor for momentum/volume deterioration → possible trend change."
        },
        "Bullish": {
            "action": "Hold or buy if you don't have a position",
            "recommendation": "Watch for potential breakdown into neutral regime."
        },
        "Sideways": {
            "action": "Wait for breakout or confirmation",
            "recommendation": "Buy on Bull regime transition, sell on Bear."
        },
        "Bearish": {
            "action": "Hold or initiate short positions with conviction",
            "recommendation": "Watch for reversal into Sideways or Bullish."
        },
        "HyperBear": {
            "action": "Hold shorts or add if conviction is high",
            "recommendation": "Monitor for divergence or reversal signals."
        }
    }

    # Narratives
    wyckoff_narrative = (
        f"The recent **weekly** action in {ticker.upper()} shows signs of {('buying pressure' if slope > 0 else 'selling pressure')}. "
        f"Volume {'expanded' if volume_spike > 1.2 else 'remained muted'}, "
        f"suggesting that {'effort is increasing' if volume_spike > 1.2 else 'conviction is low'} "
        f"but price has shown {('a gain' if momentum > 0 else 'a loss')}, "
        f"indicating {('absorption' if momentum < 0 and volume_spike > 1.2 else 'fatigue or imbalance')}.\n\n"
        f"The trend appears to be {('climbing' if slope > 0 else 'drifting')}, with momentum looking "
        f"{('strong' if rsi > 60 else 'neutral' if 45 < rsi < 60 else 'weak')}. "
        f"Weekly behavior hints at {'accumulation' if slope > 0 and volume_spike < 1 else 'distribution'}."
    )

    general_technical = (
        f"Trend: **{'Up' if slope > 0 else 'Down'}**, Strength: **{'gaining' if abs(slope) > 0.1 else 'weakening'}**.\n"
        f"Price: ${round(price, 2)} | RSI: {int(rsi)} → {('Overbought' if rsi > 70 else 'Neutral' if rsi > 45 else 'Oversold')}"
    )

    summary = f"📌 {ticker.upper()} is in a **{regime}** regime this week."

    return {
        "ticker": ticker.upper(),
        "current_regime": regime,
        "wyckoff_narrative": wyckoff_narrative.strip(),
        "technical_view": general_technical.strip(),
        "summary": summary.strip(),
        "image_base64": image_base64,
        "regime_table": df_weekly[['date','close','regime_smoothed']].to_dict(orient='records'),
        "action_plan": regime_action_table.get(regime, {})
    }


