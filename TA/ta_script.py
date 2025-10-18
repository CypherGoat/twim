#!/usr/bin/env python3
"""
Generate Monero (XMR) technical analysis snapshot.

Fetches OHLCV (via CoinGecko), computes EMA20/50, RSI(14), MACD, and Bollinger Bands,
then produces a single comprehensive PNG chart and a Markdown snapshot saved to
newsletter_assets/ta/YYYY-MM-DD.
"""

import os
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Indicator helpers ----------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return ma, upper, lower

# Candlestick plotting helper
def plot_candlestick(ax, df, width=0.6, up_color='#00D9A3', down_color='#E94B3C'):
    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    idx = df.index
    colors = np.where(closes >= opens, up_color, down_color)
    bottoms = np.minimum(opens, closes)
    bodies = np.abs(closes - opens)
    # Wicks
    ax.vlines(idx, lows, highs, colors=colors, linewidth=1, alpha=0.9, zorder=3)
    # Bodies
    ax.bar(idx, bodies, bottom=bottoms, width=width, color=colors, edgecolor='none', alpha=0.9, zorder=4)

# ---------- Comprehensive all-in-one chart ----------
def create_comprehensive_chart(df, date_str, support, resistance, out_dir):
    # Create figure with subplots for price, volume, RSI, and MACD
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0e27')
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.05)
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[3], sharex=ax_price)
    
    # Common styling for all subplots
    for ax in [ax_price, ax_volume, ax_rsi, ax_macd]:
        ax.set_facecolor('#0f1429')
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        ax.tick_params(colors='#8892b0', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#4A90E2')
        ax.spines['bottom'].set_color('#4A90E2')
    
    # Hide x-labels for all but the bottom subplot
    for ax in [ax_price, ax_volume, ax_rsi]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # 1. Price chart with indicators
    # Bollinger Bands
    ax_price.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.15, color='#4A90E2', label='Bollinger Bands')
    ax_price.plot(df.index, df['BB_upper'], linewidth=1, linestyle='--', color='#4A90E2', alpha=0.4)
    ax_price.plot(df.index, df['BB_lower'], linewidth=1, linestyle='--', color='#4A90E2', alpha=0.4)
    
    # EMAs
    ax_price.plot(df.index, df['EMA50'], label='EMA 50', linewidth=2, alpha=0.9, color='#E94B3C', linestyle='-')
    ax_price.plot(df.index, df['EMA20'], label='EMA 20', linewidth=2, alpha=0.9, color='#50E3C2', linestyle='-')
    
    # Support and resistance levels
    ax_price.axhline(support, color='#00D9A3', linestyle='--', linewidth=1.5, label=f'Support: ${support:.2f}')
    ax_price.axhline(resistance, color='#E94B3C', linestyle='--', linewidth=1.5, label=f'Resistance: ${resistance:.2f}')
    
    # Plot candlesticks
    plot_candlestick(ax_price, df)
    
    # Add current price annotation
    last_price = df['close'].iloc[-1]
    ax_price.annotate(f'${last_price:.2f}', 
                xy=(df.index[-1], last_price), 
                xytext=(10, 0), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF6B35', alpha=0.8, edgecolor='none'),
                fontsize=10, fontweight='bold', color='white',
                va='center')
    
    # Labels and title
    ax_price.set_title(f'MONERO (XMR) Technical Analysis — {date_str}', 
                fontsize=18, fontweight='bold', color='white', pad=20, loc='left')
    ax_price.set_ylabel('Price (USD)', fontsize=12, fontweight='bold', color='#8892b0')
    ax_price.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True, 
                fontsize=10, facecolor='#1a1f3a', edgecolor='#4A90E2')
    
    # 2. Volume subplot
    volume_colors = ['#00D9A3' if df['close'].iloc[i] >= df['open'].iloc[i] else '#E94B3C' for i in range(len(df))]
    ax_volume.bar(df.index, df['volume'], alpha=0.6, color=volume_colors, edgecolor='none')
    
    # Add volume SMA
    vol_sma = df['volume'].rolling(window=20).mean()
    ax_volume.plot(df.index, vol_sma, color='#A78BFA', linewidth=1.5, label='Volume SMA(20)')
    
    # Volume average line
    avg_volume = df['volume'].mean()
    ax_volume.axhline(avg_volume, color='#8892b0', linestyle='--', linewidth=1, alpha=0.5, 
                    label=f'Avg Volume: {avg_volume:.0f}')
    
    ax_volume.set_ylabel('Volume', fontsize=11, fontweight='bold', color='#8892b0')
    ax_volume.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True,
                    fontsize=9, facecolor='#1a1f3a', edgecolor='#4A90E2')
    
    # 3. RSI subplot
    ax_rsi.fill_between(df.index, 70, 100, alpha=0.2, color='#E94B3C', label='Overbought')
    ax_rsi.fill_between(df.index, 0, 30, alpha=0.2, color='#00D9A3', label='Oversold')
    ax_rsi.fill_between(df.index, 30, 70, alpha=0.05, color='#4A90E2')
    ax_rsi.plot(df.index, df['RSI14'], linewidth=2, color='#A78BFA', label='RSI (14)')
    ax_rsi.axhline(70, linewidth=1.5, linestyle='--', color='#E94B3C', alpha=0.6)
    ax_rsi.axhline(30, linewidth=1.5, linestyle='--', color='#00D9A3', alpha=0.6)
    ax_rsi.axhline(50, linewidth=1, linestyle=':', color='#8892b0', alpha=0.4)
    
    # Add RSI value annotation
    last_rsi = df['RSI14'].iloc[-1]
    rsi_color = '#E94B3C' if last_rsi > 70 else ('#00D9A3' if last_rsi < 30 else '#A78BFA')
    ax_rsi.annotate(f'RSI: {last_rsi:.1f}', 
                xy=(df.index[-1], last_rsi), 
                xytext=(10, 0), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=rsi_color, alpha=0.8, edgecolor='none'),
                fontsize=10, fontweight='bold', color='white',
                va='center')
    
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI', fontsize=11, fontweight='bold', color='#8892b0')
    ax_rsi.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True,
                fontsize=9, facecolor='#1a1f3a', edgecolor='#4A90E2')
    
    # 4. MACD subplot
    macd_colors = ['#00D9A3' if val >= 0 else '#E94B3C' for val in df['MACD_hist']]
    ax_macd.bar(df.index, df['MACD_hist'], alpha=0.6, color=macd_colors, edgecolor='none', width=0.8, label='Histogram')
    ax_macd.plot(df.index, df['MACD_line'], linewidth=2, color='#50E3C2', label='MACD Line')
    ax_macd.plot(df.index, df['MACD_signal'], linewidth=2, color='#FF6B35', label='Signal Line')
    ax_macd.axhline(0, linewidth=1.5, linestyle='--', color='#8892b0', alpha=0.5)
    
    # Add MACD crossover highlighting
    macd_diff = df['MACD_line'] - df['MACD_signal']
    ax_macd.fill_between(df.index, 0, macd_diff, where=(macd_diff >= 0), alpha=0.1, color='#00D9A3', interpolate=True)
    ax_macd.fill_between(df.index, 0, macd_diff, where=(macd_diff < 0), alpha=0.1, color='#E94B3C', interpolate=True)
    
    # Add MACD value annotation
    last_macd = df['MACD_line'].iloc[-1]
    last_signal = df['MACD_signal'].iloc[-1]
    macd_color = '#00D9A3' if last_macd > last_signal else '#E94B3C'
    ax_macd.annotate(f'MACD: {last_macd:.4f}', 
                xy=(df.index[-1], last_macd), 
                xytext=(10, 0), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=macd_color, alpha=0.8, edgecolor='none'),
                fontsize=10, fontweight='bold', color='white',
                va='center')
    
    ax_macd.set_ylabel('MACD', fontsize=11, fontweight='bold', color='#8892b0')
    ax_macd.set_xlabel('Date', fontsize=11, fontweight='bold', color='#8892b0')
    ax_macd.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True,
                fontsize=9, facecolor='#1a1f3a', edgecolor='#4A90E2')
    
    # Add market insights annotation box in the top right
    price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    
    # Determine trend status based on multiple indicators
    ema_trend = 'Bullish ↗' if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else 'Bearish ↘'
    rsi_status = 'Overbought' if df['RSI14'].iloc[-1] > 70 else ('Oversold' if df['RSI14'].iloc[-1] < 30 else 'Neutral')
    macd_status = 'Bullish' if df['MACD_line'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'Bearish'
    
    insights_text = (
        f"MARKET INSIGHTS\n"
        f"Period: {df.index[0].strftime('%b %d')} - {df.index[-1].strftime('%b %d, %Y')}\n"
        f"Price Change: {price_change:.1f}%\n"
        f"EMA Trend: {ema_trend}\n" 
        f"RSI Status: {rsi_status} ({df['RSI14'].iloc[-1]:.1f})\n"
        f"MACD Signal: {macd_status}"
    )
    
    # Add text box with insights
    props = dict(boxstyle='round,pad=0.5', facecolor='#1a1f3a', alpha=0.8, edgecolor='#4A90E2')
    ax_price.text(0.98, 0.97, insights_text, transform=ax_price.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props, color='white')
    
    # Save the comprehensive chart
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'xmr_comprehensive.png'), dpi=150, bbox_inches='tight', facecolor='#0a0e27')
    plt.close(fig)

# ---------- Data fetch (CoinGecko public API) ----------
def fetch_ohlcv_coingecko(coin_id='monero', vs_currency='usd', days=180):
    try:
        import requests
        
        # First fetch most recent data (with higher granularity)
        recent_url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc'
        recent_params = {'vs_currency': vs_currency, 'days': '7'}  # Always get last 7 days in high resolution
        recent_response = requests.get(recent_url, params=recent_params, timeout=30)
        recent_response.raise_for_status()
        recent_data = recent_response.json()
        
        if not recent_data or len(recent_data) == 0:
            print("CoinGecko returned empty recent data")
            return None
        
        recent_df = pd.DataFrame(recent_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'], unit='ms')
        recent_df.set_index('timestamp', inplace=True)
        
        # If only recent data is needed, return it
        if days <= 7:
            if len(recent_df) > days:
                recent_df = recent_df.tail(days)
            return recent_df
            
        # For longer timeframes, also fetch historical data
        # Map requested days to supported values, but exclude what we already have
        if days <= 14:
            api_days = 14
        elif days <= 30:
            api_days = 30
        elif days <= 90:
            api_days = 90
        elif days <= 180:
            api_days = 180
        else:
            api_days = 365
        
        hist_url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc'
        hist_params = {'vs_currency': vs_currency, 'days': str(api_days)}
        hist_response = requests.get(hist_url, params=hist_params, timeout=30)
        hist_response.raise_for_status()
        hist_data = hist_response.json()
        
        if not hist_data or len(hist_data) == 0:
            print("CoinGecko returned empty historical data, using recent data only")
            return recent_df
            
        hist_df = pd.DataFrame(hist_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], unit='ms')
        hist_df.set_index('timestamp', inplace=True)
        
        # Combine datasets, with recent data overriding historical data when timestamps overlap
        combined_df = pd.concat([hist_df, recent_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        # If we got more data than requested, trim to the requested timeframe
        if len(combined_df) > days:
            combined_df = combined_df.tail(days)
        
        print(f"Fetched {len(combined_df)} data points from CoinGecko (with {len(recent_df)} recent points)")
        return combined_df
        
    except Exception as e:
        print(f"CoinGecko fetch failed: {e}")
        return None



# ---------- Main ----------
def make_outputs(base_out_dir='newsletter_assets/ta', pair_hint='XMR/USDT', days=365):
    # Prepare output folder for today's run (ISO date)
    run_date = datetime.now(timezone.utc).date().isoformat()  # e.g. 2025-10-11
    out_dir = os.path.join(base_out_dir, run_date)
    os.makedirs(out_dir, exist_ok=True)

    # Attempt to fetch real OHLCV from CoinGecko
    df = fetch_ohlcv_coingecko(coin_id='monero', vs_currency='usd', days=days)
    if df is None or len(df) == 0:
        raise RuntimeError("Failed to fetch real market data from CoinGecko. Cannot proceed without real data.")

    # Ensure numeric columns  
    df = df.astype({'close': float, 'open': float, 'high': float, 'low': float})
    
    # Add volume data from CoinGecko markets endpoint if not available
    if 'volume' not in df.columns:
        try:
            import requests
            # Use same day mapping as OHLC fetch to ensure consistency
            if days <= 1:
                api_days = 1
            elif days <= 7:
                api_days = 7
            elif days <= 14:
                api_days = 14
            elif days <= 30:
                api_days = 30
            elif days <= 90:
                api_days = 90
            elif days <= 180:
                api_days = 180
            else:
                api_days = 365
                
            url = f'https://api.coingecko.com/api/v3/coins/monero/market_chart'
            params = {'vs_currency': 'usd', 'days': str(api_days)}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            volume_data = response.json().get('total_volumes', [])
            
            if volume_data:
                volume_df = pd.DataFrame(volume_data, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                
                # Align volume data with OHLC timeframe
                if len(volume_df) > days and days < api_days:
                    volume_df = volume_df.tail(days)
                    
                # Resample to match OHLC frequency and merge
                if len(df) > 0 and len(volume_df) > 0:
                    # Determine frequency based on data points
                    time_diff = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
                    if time_diff / len(df) < 1:  # Sub-daily data
                        freq = 'H'  # Hourly
                    else:
                        freq = 'D'  # Daily
                    
                    volume_df = volume_df.resample(freq).last()
                    df = df.merge(volume_df, left_index=True, right_index=True, how='left')
                    df['volume'] = df['volume'].fillna(df['volume'].mean())
                    print(f"Successfully fetched volume data: {len(volume_df)} volume points")
                else:
                    raise Exception("No volume data available after filtering")
            else:
                raise Exception("No volume data available")
        except Exception as e:
            print(f"Warning: Could not fetch volume data: {e}")
            # If volume fetch fails, we cannot proceed as volume is required for the analysis
            raise RuntimeError("Volume data is required for technical analysis but could not be fetched from CoinGecko")
    
    df = df.astype({'volume': float})

    # Indicators
    df['EMA20'] = ema(df['close'], span=20)
    df['EMA50'] = ema(df['close'], span=50)
    df['RSI14'] = rsi(df['close'], period=14)
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = macd(df['close'], fast=12, slow=26, signal=9)
    df['BB_MA'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'], window=20, num_std=2)

    # Snapshot (most recent)
    latest = df.iloc[-1]
    date_str = df.index[-1].strftime('%B %d, %Y')
    price = latest['close']
    
    # Calculate 7-day change more robustly
    try:
        # Find data point closest to 7 days ago
        seven_days_ago = df.index[-1] - pd.Timedelta(days=7)
        # Get closest available data point
        seven_day_idx = df.index.get_indexer([seven_days_ago], method='nearest')[0]
        if seven_day_idx >= 0 and seven_day_idx < len(df) - 1:
            seven_day_change = (df['close'].iloc[-1] / df['close'].iloc[seven_day_idx] - 1) * 100
        else:
            # Fallback to using 7 data points back if available
            if len(df) > 7:
                seven_day_change = (df['close'].iloc[-1] / df['close'].iloc[-8] - 1) * 100
            else:
                seven_day_change = float('nan')
    except Exception:
        seven_day_change = float('nan')

    rsi_val = round(latest['RSI14'], 1)
    macd_line = latest['MACD_line']
    macd_signal = latest['MACD_signal']
    ema20 = latest['EMA20']
    ema50 = latest['EMA50']
    bb_upper = latest['BB_upper']
    bb_lower = latest['BB_lower']

    # Signals text
    rsi_signal = 'Neutral'
    if rsi_val > 70:
        rsi_signal = 'Overbought'
    elif rsi_val < 30:
        rsi_signal = 'Oversold'
    macd_signal_text = 'Bullish' if macd_line > macd_signal else 'Bearish'
    ema_trend = 'Uptrend' if ema20 > ema50 else 'Downtrend'
    bb_pos = 'Near upper band' if (not math.isnan(bb_upper) and price >= 0.9 * bb_upper) else ('Near lower band' if (not math.isnan(bb_lower) and price <= 1.1 * bb_lower) else 'Middle')

    # Very simple pattern heuristic
    recent = df[-40:]
    lows = recent['low'][-20:]
    highs = recent['high'][-20:]
    higher_lows = False
    if len(lows) >= 2:
        higher_lows = lows.iloc[-1] > lows.iloc[0]
    flat_highs = highs.std() < (highs.mean() * 0.015) if highs.mean() != 0 else False
    resistance = float(highs.max())
    support = float(lows.min())
    if higher_lows and flat_highs:
        pattern_text = f'**Ascending triangle** forming — potential breakout above **{resistance:.2f}** if volume confirms.'
    else:
        pattern_text = 'No clear pattern detected.'

    # ---------- Create charts ----------
    plt.style.use('dark_background')
    
    # Generate comprehensive all-in-one chart only
    create_comprehensive_chart(df, date_str, support, resistance, out_dir)
    
    plt.style.use('default')

    # ---------- Write markdown snapshot ----------
    md_lines = []
    md_lines.append(f"### Monero (XMR) Technical Overview")
    md_lines.append(f"**Date:** {date_str}  ")
    md_lines.append(f"**Pair/source:** {pair_hint}  ")
    md_lines.append("")
    md_lines.append(f"**Price:** `${price:.2f}`  ")
    md_lines.append(f"**7-Day Change:** `{seven_day_change:.2f}%`  ")
    vol_trend_symbol = '↑' if df['volume'][-7:].mean() > df['volume'][-30:].mean() else '↓'
    md_lines.append(f"**Volume Trend:** {vol_trend_symbol} compared to 30-day average  ")
    md_lines.append("")
    md_lines.append("**Indicators**")
    md_lines.append("| Indicator | Value | Signal |")
    md_lines.append("|---|---:|---|")
    md_lines.append(f"| RSI (14) | {rsi_val} | {rsi_signal} |")
    md_lines.append(f"| MACD | {'Bullish' if macd_line>macd_signal else 'Bearish'} | {'Strong' if macd_line>macd_signal else 'Caution'} |")
    md_lines.append(f"| 20-EMA vs 50-EMA | {ema20:.2f} vs {ema50:.2f} | {ema_trend} |")
    md_lines.append(f"| Bollinger Bands | Price {bb_pos} | {'Caution' if 'upper' in bb_pos or 'lower' in bb_pos else 'Stable'} |")
    md_lines.append("")
    md_lines.append(f"![](./xmr_comprehensive.png)")
    md_lines.append("")
    md_lines.append("**Chart Pattern**  ")
    md_lines.append(f"{pattern_text}  ")
    md_lines.append(f"Support around **{support:.2f}**; next resistance near **{resistance:.2f}**.")
    md_lines.append("")
    md_lines.append("**Outlook**  ")
    md_lines.append(f"Momentum: {macd_signal_text.lower()}. Short-term trend: {ema_trend.lower()}. As long as XMR stays above **{support:.2f}**, setup favors bulls. A close below **{support*0.98:.2f}** would weaken the setup.")
    md_content = "\n".join(md_lines)

    md_path = os.path.join(out_dir, f"xmr_ta_{run_date}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    manifest = {
        "date": run_date,
        "comprehensive_chart": os.path.relpath(os.path.join(out_dir, 'xmr_comprehensive.png')),
        "markdown": os.path.relpath(md_path)
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved outputs to:", out_dir)
    return out_dir, manifest

if __name__ == "__main__":
    out_dir, manifest = make_outputs()
    print("Manifest:", manifest)
