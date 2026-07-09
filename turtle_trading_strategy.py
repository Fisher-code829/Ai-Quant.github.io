#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海龟交易法则（Turtle Trading System）完整回测脚本
==================================================
功能：
  1. 加载本地存储的股价 CSV 数据
  2. 计算 Donchian 高低价格通道（可调周期：20/55/自定义）
  3. 计算 ATR（Average True Range）用于仓位管理和止损
  4. 生成买入/卖出交易信号
  5. 可视化：K线 + 通道 + 买卖标记 + 净值曲线 + 回撤
  6. 模拟交易回测，计算量化指标
  7. 参数调节 + 多股票对比 + 心得总结

海龟法则核心逻辑：
  - 入场：价格突破 N 日最高价（买入）/ 价格跌破 N 日最低价（卖出）
  - 出场：价格跌破 M 日最低价（平多）/ 价格突破 M 日最高价（平空）
  - 仓位：1 Unit = (资金 × 风险%) / ATR，最大 4 Units
  - 止损：2 × ATR 硬止损
  - 加仓：每 0.5 ATR 盈利加 1 Unit（最多 4 次）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from pathlib import Path
import requests
import warnings
import sys
import os
from io import StringIO

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 中文字体回退处理
for f in plt.rcParams['font.sans-serif']:
    try:
        plt.rcParams['font.family'] = f
        break
    except:
        continue

# 中文 → 英文 label 映射（万一中文字体不可用时的回退）
USE_CN = True
def L(cn, en):
    return cn if USE_CN else en

# 颜色
C_UP = '#f85149'   # 红涨
C_DN = '#3fb950'   # 绿跌
C_BUY = '#ff4757'
C_SELL = '#2ed573'
C_CH_U = '#58a6ff'
C_CH_D = '#f0883e'
C_NAV = '#f85149'
C_DD = '#3fb950'

CHART_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / 'turtle_charts'
CHART_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 数据加载
# ============================================================
def load_local_csv(filepath):
    """加载本地 CSV 文件，返回标准化 DataFrame"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    required = ['date', 'open', 'high', 'low', 'close']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")
    return df

def fetch_eastmoney_data(code, start_date='20200101', end_date='20260701', market=None):
    """从东方财富 API 获取前复权日线数据"""
    if market is None:
        market = 1 if code[0] == '6' or code[:3] == '688' else 0
    secid = f'{market}.{code}'
    url = ('https://push2his.eastmoney.com/api/qt/stock/kline/get'
           f'?secid={secid}&fields1=f1,f2,f3,f4,f5,f6'
           f'&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
           f'&klt=101&fqt=1&beg={start_date}&end={end_date}&lmt=2000')
    resp = requests.get(url, timeout=30)
    data = resp.json()
    if not data.get('data') or not data['data'].get('klines'):
        raise ValueError(f"获取 {code} 数据失败")
    
    klines = data['data']['klines']
    rows = []
    for line in klines:
        parts = line.split(',')
        rows.append({
            'date': parts[0],
            'open': float(parts[1]),
            'close': float(parts[2]),
            'high': float(parts[3]),
            'low': float(parts[4]),
            'volume': float(parts[5]),
            'amount': float(parts[6]) if len(parts) > 6 else 0
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df

def ensure_data(code, name=None):
    """确保数据存在：优先读本地 CSV，否则从 API 拉取并缓存"""
    csv_path = DATA_DIR / f'stock_{code}_daily.csv'
    if csv_path.exists():
        print(f"  [本地] 加载 {name or code} ({csv_path.name})")
        return load_local_csv(csv_path)
    print(f"  [API]  拉取 {name or code} ...")
    df = fetch_eastmoney_data(code)
    if not csv_path.exists():
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  [缓存] 已保存 {csv_path.name}")
    return df

# ============================================================
# 2. 计算 Donchian 高低价格通道
# ============================================================
def calc_donchian(df, period):
    """计算 Donchian 通道（周期内最高价/最低价）"""
    upper = df['high'].rolling(window=period).max().shift(1)   # 用前 N 天的最高
    lower = df['low'].rolling(window=period).min().shift(1)    # 用前 N 天的最低
    return upper, lower

# ============================================================
# 3. 计算 ATR
# ============================================================
def calc_atr(df, period=20):
    """计算 ATR（Average True Range），Turtle 用 EMA"""
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Turtle 用 EMA 而非 SMA
    atr = np.zeros(len(close))
    atr[:period] = np.nan
    atr[period] = np.mean(tr[:period+1])
    alpha = 2.0 / (period + 1)
    for i in range(period + 1, len(close)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr, tr

# ============================================================
# 4. 生成海龟交易信号
# ============================================================
def generate_turtle_signals(df, entry_period=20, exit_period=10, atr_period=20):
    """
    海龟法则信号生成
    - entry_period: 入场通道周期（经典值 20 或 55）
    - exit_period:  出场通道周期（经典值 10 或 20）
    - atr_period:   ATR 周期（经典值 20）

    返回 signals DataFrame，包含：
      - entry_upper, entry_lower: 入场通道上下轨
      - exit_upper, exit_lower:   出场通道上下轨
      - atr:  ATR 值
      - signal: 1=买入开仓, -1=卖出平仓, 0=无信号
    """
    s = pd.DataFrame(index=df.index)
    s['date'] = df['date']
    s['close'] = df['close']

    # Donchian 通道
    s['entry_upper'], s['entry_lower'] = calc_donchian(df, entry_period)
    s['exit_upper'], s['exit_lower'] = calc_donchian(df, exit_period)

    # ATR
    atr, tr = calc_atr(df, atr_period)
    s['atr'] = atr

    # 交易信号
    n = len(df)
    signal = np.zeros(n, dtype=int)
    in_position = False

    for i in range(max(entry_period, exit_period, atr_period) + 1, n):
        close = df['close'].iloc[i]
        if not in_position:
            # 无持仓 → 等待入场信号
            if (not pd.isna(s['entry_upper'].iloc[i]) and
                close > s['entry_upper'].iloc[i]):
                signal[i] = 1
                in_position = True
        else:
            # 有持仓 → 等待出场信号
            if (not pd.isna(s['exit_lower'].iloc[i]) and
                close < s['exit_lower'].iloc[i]):
                signal[i] = -1
                in_position = False

    s['signal'] = signal
    return s

# ============================================================
# 5. 模拟交易和回测
# ============================================================
def run_backtest(df, signals, initial_capital=100000, risk_pct=0.02,
                 use_pyramid=True, use_stop_loss=True):
    """
    模拟交易回测
    - risk_pct: 每笔交易风险占比（Turtle 经典 2%）
    - use_pyramid: 是否加仓（每 0.5N 加 1 unit，最多 4 units）
    - use_stop_loss: 是否启用 2N 硬止损
    """
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = signals['atr'].values
    signal = signals['signal'].values

    cash = float(initial_capital)
    shares = 0
    total_value = np.zeros(n)
    position_atr_cost = 0  # 记录入场时的每 share 的 N 值

    trades = []
    pyramid_units = 0
    pyramid_entries = []  # 记录每次加仓价格

    entry_date, entry_price = None, 0
    in_position = False

    for i in range(n):
        price = close[i]
        curr_atr = atr[i]

        # --- 入场逻辑 ---
        if signal[i] == 1 and not in_position:
            unit_size = int((cash * risk_pct) / curr_atr) if (curr_atr > 0 and not pd.isna(curr_atr)) else 0
            if unit_size > 0:
                # 以开盘价或当前价买入（模拟）
                buy_price = price
                cost = unit_size * buy_price
                if cost <= cash:
                    cash -= cost
                    shares = unit_size
                    in_position = True
                    pyramid_units = 1
                    entry_price = buy_price
                    entry_date = df['date'].iloc[i]
                    position_atr_cost = curr_atr
                    pyramid_entries = [buy_price]
        # --- 加仓逻辑 ---
        elif in_position and use_pyramid and pyramid_units < 4 and position_atr_cost > 0:
            add_price = pyramid_entries[0] + 0.5 * position_atr_cost * pyramid_units
            if price >= add_price:
                unit_size = int((cash * risk_pct) / position_atr_cost) if position_atr_cost > 0 else 0
                if unit_size > 0 and unit_size * price <= cash:
                    cash -= unit_size * price
                    shares += unit_size
                    pyramid_units += 1
                    pyramid_entries.append(price)

        # --- 止损逻辑 ---
        if in_position and use_stop_loss and position_atr_cost > 0:
            stop_price = entry_price - 2 * position_atr_cost
            if price <= stop_price:
                # 触发了 2N 止损
                proceeds = shares * price
                cash += proceeds
                ret = (price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': df['date'].iloc[i].strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(price, 2),
                    'return': round(ret, 4),
                    'type': '止损',
                    'units': pyramid_units
                })
                shares = 0
                in_position = False
                pyramid_units = 0
                pyramid_entries = []
                position_atr_cost = 0
                entry_price = 0

        # --- 出场逻辑 ---
        if signal[i] == -1 and in_position:
            proceeds = shares * price
            cash += proceeds
            ret = (price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': df['date'].iloc[i].strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 2),
                'exit_price': round(price, 2),
                'return': round(ret, 4),
                'type': '正常出场',
                'units': pyramid_units
            })
            shares = 0
            in_position = False
            pyramid_units = 0
            pyramid_entries = []
            entry_price = 0
            position_atr_cost = 0

        total_value[i] = cash + shares * price

    # 最后强制平仓
    if in_position:
        last_price = close[-1]
        proceeds = shares * last_price
        cash += proceeds
        ret = (last_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': df['date'].iloc[n-1].strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'exit_price': round(last_price, 2),
            'return': round(ret, 4),
            'type': '期末平仓',
            'units': pyramid_units
        })
        total_value[n-1] = cash

    # 基准（买入持有）
    first_price = close[0]
    bench_shares = initial_capital / first_price
    bench_value = close * bench_shares

    return total_value, bench_value, trades

# ============================================================
# 6. 量化指标计算
# ============================================================
def calc_metrics(nav, bench, trades, initial_capital, df):
    """计算全套量化指标"""
    n = len(nav)
    years = max(n / 252, 0.1)
    total_ret = (nav[-1] - initial_capital) / initial_capital
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    bench_total_ret = (bench[-1] - initial_capital) / initial_capital
    bench_ann_ret = (1 + bench_total_ret) ** (1 / years) - 1

    # 日收益率
    daily_rets = np.diff(nav) / nav[:-1]
    bench_daily_rets = np.diff(bench) / bench[:-1]

    # 最大回撤
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = np.min(dd)
    max_dd_idx = np.argmin(dd)
    max_dd_date = df['date'].iloc[max_dd_idx].strftime('%Y-%m-%d')

    # 夏普 (无风险利率 2%)
    rf = 0.02 / 252
    excess = daily_rets - rf
    sharpe = np.mean(excess) / max(np.std(excess, ddof=1), 1e-10) * np.sqrt(252)

    # 胜率 & 盈亏比
    trade_rets = [t['return'] for t in trades]
    wins = [r for r in trade_rets if r > 0]
    losses = [r for r in trade_rets if r <= 0]
    win_rate = len(wins) / max(len(trade_rets), 1)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins)) / max(abs(sum(losses)), 1e-10)

    # 年化波动率
    ann_vol = np.std(daily_rets, ddof=1) * np.sqrt(252)

    # Calmar
    calmar = ann_ret / max(abs(max_dd), 1e-10)

    # 超额收益
    excess_ret_over_bench = total_ret - bench_total_ret
    info_ratio = np.mean(daily_rets - bench_daily_rets) / max(np.std(daily_rets - bench_daily_rets, ddof=1), 1e-10) * np.sqrt(252)

    return {
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'bench_total_ret': bench_total_ret,
        'bench_ann_ret': bench_ann_ret,
        'max_dd': max_dd,
        'max_dd_date': max_dd_date,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'n_trades': len(trades),
        'ann_vol': ann_vol,
        'calmar': calmar,
        'excess_ret': excess_ret_over_bench,
        'info_ratio': info_ratio,
        'n_win': len(wins),
        'n_loss': len(losses)
    }

# ============================================================
# 7. 可视化
# ============================================================
def plot_turtle_analysis(df, signals, nav, bench, trades, metrics,
                         stock_name, entry_period, exit_period,
                         save_path=None):
    """绘制完整海龟策略分析图"""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2.5, 1.5, 1.5, 1.5],
                  hspace=0.35, wspace=0.25)

    dates = df['date'].values

    # --- 图1: 股价 + Donchian通道 + 买卖信号 ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, signals['entry_lower'].values,
                     signals['entry_upper'].values,
                     alpha=0.08, color='#58a6ff', label=f'Donchian({entry_period}日通道)')
    ax1.plot(dates, df['close'].values, color='#333', linewidth=1.2, label='收盘价', zorder=3)
    ax1.plot(dates, signals['entry_upper'].values, color=C_UP, linewidth=0.8,
             linestyle='--', label=f'{entry_period}日高轨', alpha=0.7)
    ax1.plot(dates, signals['entry_lower'].values, color=C_DN, linewidth=0.8,
             linestyle='--', label=f'{entry_period}日低轨', alpha=0.7)
    ax1.plot(dates, signals['exit_upper'].values, color=C_UP, linewidth=0.5,
             linestyle=':', label=f'{exit_period}日出口上轨', alpha=0.5)
    ax1.plot(dates, signals['exit_lower'].values, color=C_DN, linewidth=0.5,
             linestyle=':', label=f'{exit_period}日出口下轨', alpha=0.5)

    # 买卖标记
    buy_idx = np.where(signals['signal'].values == 1)[0]
    sell_idx = np.where(signals['signal'].values == -1)[0]
    ax1.scatter(dates[buy_idx], df['close'].values[buy_idx] * 0.97,
                marker='^', s=80, color=C_BUY, edgecolors='white', linewidth=0.8,
                zorder=5, label=f'买入({len(buy_idx)}次)')
    ax1.scatter(dates[sell_idx], df['close'].values[sell_idx] * 1.03,
                marker='v', s=80, color=C_SELL, edgecolors='white', linewidth=0.8,
                zorder=5, label=f'卖出({len(sell_idx)}次)')

    # 止损标记
    stop_trades = [t for t in trades if t['type'] == '止损']
    if stop_trades:
        stop_dates = pd.to_datetime([t['exit_date'] for t in stop_trades])
        stop_prices = [t['exit_price'] for t in stop_trades]
        ax1.scatter(stop_dates, stop_prices, marker='x', s=60, color='orange',
                    linewidths=2, zorder=6, label=f'止损({len(stop_trades)}次)')

    ax1.set_title(f'{stock_name} · 海龟交易法则 (入场通道={entry_period}日, 出场通道={exit_period}日)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格 (¥)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    # --- 图2: ATR 走势 ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(dates, 0, signals['atr'].values, alpha=0.3, color=C_CH_U,
                     label='ATR(20)')
    ax2.plot(dates, signals['atr'].values, color=C_CH_U, linewidth=1)
    ax2.set_title('ATR (Average True Range)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ATR', fontsize=9)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # --- 图3: 策略净值 vs 买入持有 ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, nav / 10000, color=C_NAV, linewidth=1.5, label='策略净值')
    ax3.plot(dates, bench / 10000, color='#888', linewidth=1, linestyle='--',
             label='买入持有')
    ax3.axhline(y=nav[0]/10000, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('策略净值 vs 买入持有', fontsize=11, fontweight='bold')
    ax3.set_ylabel('净值 (万元)', fontsize=9)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # --- 图4: 回撤曲线 ---
    ax4 = fig.add_subplot(gs[2, 0])
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak * 100
    ax4.fill_between(dates, 0, dd, color=C_DD, alpha=0.4)
    ax4.plot(dates, dd, color=C_DD, linewidth=0.8)
    ax4.set_title(f'回撤曲线 | 最大回撤: {metrics["max_dd"]*100:.1f}% ({metrics["max_dd_date"]})',
                  fontsize=11, fontweight='bold')
    ax4.set_ylabel('回撤 (%)', fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # --- 图5: 每笔交易收益柱状图 ---
    ax5 = fig.add_subplot(gs[2, 1])
    trade_rets = [t['return'] * 100 for t in trades]
    colors_bar = [C_BUY if r > 0 else C_SELL for r in trade_rets]
    ax5.bar(range(len(trades)), trade_rets, color=colors_bar, alpha=0.85, edgecolor='white')
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    cum_rets = np.cumsum(trade_rets)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(range(len(trades)), cum_rets, 'o-', color=C_CH_U, markersize=4,
                  linewidth=1.2, label='累计收益(%)')
    ax5.set_title(f'每笔交易收益 (共{len(trades)}笔)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('交易序号', fontsize=9)
    ax5.set_ylabel('单笔收益 (%)', fontsize=9)
    ax5_twin.set_ylabel('累计收益 (%)', fontsize=9, color=C_CH_U)
    ax5_twin.legend(loc='upper left', fontsize=8)
    ax5.grid(alpha=0.3, axis='y')

    # --- 图6: 量化指标表 ---
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    # 指标文本
    m = metrics
    rows_data = [
        ['策略总收益', f'{m["total_ret"]*100:+.2f}%',
         '买入持有收益', f'{m["bench_total_ret"]*100:+.2f}%',
         '超额收益', f'{m["excess_ret"]*100:+.2f}%'],
        ['年化收益率', f'{m["ann_ret"]*100:+.2f}%',
         '基准年化', f'{m["bench_ann_ret"]*100:+.2f}%',
         '年化波动率', f'{m["ann_vol"]*100:.2f}%'],
        ['最大回撤', f'{m["max_dd"]*100:.2f}% ({m["max_dd_date"]})',
         '夏普比率', f'{m["sharpe"]:.2f}',
         'Calmar比率', f'{m["calmar"]:.2f}'],
        ['胜率', f'{m["win_rate"]*100:.1f}% ({m["n_win"]}赢/{m["n_loss"]}亏)',
         '盈亏比', f'{m["profit_factor"]:.2f}',
         '信息比率', f'{m["info_ratio"]:.2f}'],
        ['交易次数', f'{m["n_trades"]}',
         '平均盈利', f'{m["avg_win"]*100:+.2f}%',
         '平均亏损', f'{m["avg_loss"]*100:.2f}%'],
    ]

    col_widths = [0.12, 0.20, 0.12, 0.20, 0.12, 0.24]
    table = ax6.table(cellText=rows_data, colWidths=col_widths, loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#ddd')
        if key[0] == 0:  # header
            cell.set_facecolor('#f0f0f0')
            cell.set_fontsize(9)
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#fafafa' if key[0] % 2 == 0 else 'white')

    ax6.set_title(f'{stock_name} · 海龟交易法则回测指标汇总', fontsize=12,
                  fontweight='bold', y=1.02)

    fig.suptitle(f'海龟交易法则回测分析 - {stock_name}\n'
                 f'(入场={entry_period}日 / 出场={exit_period}日 / ATR=20)',
                 fontsize=14, fontweight='bold', y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [图表] 已保存: {save_path}")
    plt.close(fig)

# ============================================================
# 8. 参数扫描
# ============================================================
def param_sweep(df, name, entry_periods=[10, 20, 30, 55],
                exit_periods=[5, 10, 20], atr_period=20):
    """参数扫描：遍历不同的入场/出场周期组合"""
    results = []
    print(f"\n{'='*60}")
    print(f"  参数扫描: {name}")
    print(f"{'='*60}")
    print(f"  {'入场':>6} {'出场':>6} {'总收益':>8} {'年化':>8} {'最大回撤':>8} "
          f"{'夏普':>6} {'胜率':>6} {'交易数':>6}")
    print(f"  {'-'*60}")

    for ep in entry_periods:
        for xp in exit_periods:
            if xp >= ep:
                continue
            sig = generate_turtle_signals(df, entry_period=ep, exit_period=xp, atr_period=atr_period)
            nav, bench, trades = run_backtest(df, sig, use_pyramid=False, use_stop_loss=True)
            m = calc_metrics(nav, bench, trades, 100000, df)
            results.append({
                'entry_period': ep, 'exit_period': xp,
                'total_ret': m['total_ret'], 'ann_ret': m['ann_ret'],
                'max_dd': m['max_dd'], 'sharpe': m['sharpe'],
                'win_rate': m['win_rate'], 'n_trades': m['n_trades'],
                'profit_factor': m['profit_factor'],
                'nav': nav, 'bench': bench, 'trades': trades
            })
            print(f"  {ep:>6} {xp:>6} {m['total_ret']*100:>7.2f}% {m['ann_ret']*100:>7.2f}% "
                  f"{m['max_dd']*100:>7.2f}% {m['sharpe']:>5.2f} {m['win_rate']*100:>5.1f}% "
                  f"{m['n_trades']:>6}")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['nav', 'bench', 'trades']}
                               for r in results])
    return results_df, results

# ============================================================
# 9. 参数扫描可视化
# ============================================================
def plot_param_sweep(results_df, stock_name, save_path=None):
    """参数扫描热力图"""
    if results_df.empty:
        return

    # 透视表
    pivot_ret = results_df.pivot_table(values='total_ret', index='exit_period',
                                        columns='entry_period', aggfunc='mean') * 100
    pivot_sharpe = results_df.pivot_table(values='sharpe', index='exit_period',
                                           columns='entry_period', aggfunc='mean')
    pivot_dd = results_df.pivot_table(values='max_dd', index='exit_period',
                                       columns='entry_period', aggfunc='mean') * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    metrics_config = [
        (pivot_ret, '总收益 (%)', 'RdYlGn'),
        (pivot_sharpe, '夏普比率', 'RdYlGn'),
        (pivot_dd, '最大回撤 (%)', 'RdYlGn_r')
    ]

    for ax, (data, title, cmap) in zip(axes, metrics_config):
        im = ax.imshow(data.values, cmap=cmap, aspect='auto', vmin=data.values.min(),
                       vmax=data.values.max())
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, fontsize=10)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_xlabel('入场通道周期 (日)', fontsize=10)
        ax.set_ylabel('出场通道周期 (日)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # 标注数值
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.values[i, j]
                color = 'white' if abs(val) > abs(data.values).mean() else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f'{stock_name} · 海龟法则参数扫描热力图\n(入场周期 × 出场周期)', fontsize=13, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [图表] 已保存: {save_path}")
    plt.close(fig)

# ============================================================
# 10. 多股票对比
# ============================================================
def plot_multi_stock_comparison(stock_names, metrics_list, save_path=None):
    """多股票回测指标对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    idx = list(range(len(stock_names)))
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    metric_pairs = [
        ('total_ret', '策略总收益 (%)', True),
        ('ann_ret', '年化收益率 (%)', True),
        ('max_dd', '最大回撤 (%)', False),
        ('sharpe', '夏普比率', True),
        ('win_rate', '胜率', True),
        ('profit_factor', '盈亏比', True),
    ]

    for ax, (key, title, higher_is_better) in zip(axes, metric_pairs):
        values = [m[key] * 100 if key in ('total_ret', 'ann_ret', 'max_dd', 'win_rate')
                  else m[key] for m in metrics_list]
        bars = ax.bar(idx, values, color=colors_bar[:len(stock_names)], alpha=0.85,
                      edgecolor='white')
        if higher_is_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values) if key == 'max_dd' else np.argmax(values)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)

        for i, v in enumerate(values):
            ax.text(i, v + (max(values) - min(values)) * 0.04,
                    f'{v:.1f}' if key not in ('total_ret', 'ann_ret') else f'{v:.1f}%',
                    ha='center', fontsize=9, fontweight='bold' if i == best_idx else 'normal')

        ax.set_xticks(idx)
        ax.set_xticklabels(stock_names, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel(title)

    fig.suptitle('海龟交易法则 · 多股票回测对比 (入场=20日, 出场=10日)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [图表] 已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 11. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  海龟交易法则（Turtle Trading System）Python 回测")
    print("=" * 60)

    # --- 数据配置 ---
    stock_configs = [
        ('600900', '长江电力'),
        ('300251', '光线传媒'),
        ('600519', '贵州茅台'),
        ('300750', '宁德时代'),
        ('002594', '比亚迪'),
        ('601318', '中国平安'),
    ]

    # 默认回测参数
    ENTRY_PERIOD = 20
    EXIT_PERIOD = 10
    ATR_PERIOD = 20
    INITIAL_CAPITAL = 100000.0

    # --- 加载数据 ---
    print("\n[1] 加载股价数据...")
    stocks = {}
    for code, name in stock_configs:
        stocks[name] = ensure_data(code, name)
        print(f"      {name}({code}) — {len(stocks[name])} 条记录, "
              f"{stocks[name]['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
              f"{stocks[name]['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # --- 对每只股票运行回测 ---
    all_metrics = []
    all_names = []
    all_nav_bench = {}

    for name, df in stocks.items():
        print(f"\n[2] [{name}] 计算通道 + ATR + 生成信号...")
        sig = generate_turtle_signals(df, ENTRY_PERIOD, EXIT_PERIOD, ATR_PERIOD)

        n_signals_buy = (sig['signal'] == 1).sum()
        n_signals_sell = (sig['signal'] == -1).sum()
        print(f"      买入信号: {n_signals_buy} 次  卖出信号: {n_signals_sell} 次")

        print(f"    [{name}] 模拟交易回测...")
        nav, bench, trades = run_backtest(df, sig, INITIAL_CAPITAL,
                                          use_pyramid=False, use_stop_loss=True)
        m = calc_metrics(nav, bench, trades, INITIAL_CAPITAL, df)

        print(f"      策略收益: {m['total_ret']*100:+.2f}%  买入持有: {m['bench_total_ret']*100:+.2f}%")
        print(f"      夏普: {m['sharpe']:.2f}  最大回撤: {m['max_dd']*100:.2f}%  胜率: {m['win_rate']*100:.1f}%")
        print(f"      交易次数: {m['n_trades']}  盈亏比: {m['profit_factor']:.2f}")

        print(f"    [{name}] 绘制可视化图表...")
        save_path = CHART_DIR / f'turtle_{name}_ep{ENTRY_PERIOD}_xp{EXIT_PERIOD}.png'
        plot_turtle_analysis(df, sig, nav, bench, trades, m, name,
                             ENTRY_PERIOD, EXIT_PERIOD, save_path=save_path)

        all_metrics.append(m)
        all_names.append(name)
        all_nav_bench[name] = (nav, bench)

    # --- 多股票对比图 ---
    print(f"\n[3] 绘制多股票对比图...")
    plot_multi_stock_comparison(all_names, all_metrics,
                                save_path=CHART_DIR / 'turtle_multi_stock_comparison.png')

    # --- 参数扫描 ---
    print(f"\n[4] 参数扫描 (entry=[10,20,30,55], exit=[5,10,20]) ...")
    param_scan_stocks = ['长江电力', '光线传媒', '贵州茅台']
    all_param_results = {}
    for name in param_scan_stocks:
        df_param = stocks[name]
        res_df, res_list = param_sweep(df_param, name)
        all_param_results[name] = (res_df, res_list)
        best = res_df.loc[res_df['ann_ret'].idxmax()]
        print(f"    [{name}] 最优参数: 入场={best['entry_period']}日, 出场={best['exit_period']}日, "
              f"年化={best['ann_ret']*100:.2f}%, 夏普={best['sharpe']:.2f}")

        plot_param_sweep(res_df, name,
                         save_path=CHART_DIR / f'turtle_param_sweep_{name}.png')

    # --- CSV 导出 ---
    print(f"\n[5] 导出回测结果...")
    summary_rows = []
    for name, m in zip(all_names, all_metrics):
        summary_rows.append({
            '股票': name,
            '入场周期': ENTRY_PERIOD,
            '出场周期': EXIT_PERIOD,
            '策略总收益(%)': f"{m['total_ret']*100:.2f}",
            '买入持有收益(%)': f"{m['bench_total_ret']*100:.2f}",
            '超额收益(%)': f"{m['excess_ret']*100:.2f}",
            '年化收益率(%)': f"{m['ann_ret']*100:.2f}",
            '最大回撤(%)': f"{m['max_dd']*100:.2f}",
            '夏普比率': f"{m['sharpe']:.2f}",
            'Calmar比率': f"{m['calmar']:.2f}",
            '年化波动率(%)': f"{m['ann_vol']*100:.2f}",
            '胜率(%)': f"{m['win_rate']*100:.1f}",
            '交易次数': m['n_trades'],
            '盈亏比': f"{m['profit_factor']:.2f}",
            '信息比率': f"{m['info_ratio']:.2f}",
            '最大回撤日期': m['max_dd_date'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = CHART_DIR / 'turtle_trading_summary.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"    CSV: {summary_csv}")

    # 参数扫描结果导出
    for name in param_scan_stocks:
        if name in all_param_results:
            df, _ = all_param_results[name]
            csv_path = CHART_DIR / f'turtle_param_results_{name}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ============================================================
    # 策略心得总结
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  海龟交易法则回测总结与分析")
    print(f"{'='*60}")

    print(f"""
【海龟法则核心机制回顾】

  海龟交易法则由 Richard Dennis 创立，是趋势跟踪领域的经典策略。核心思路：
  - 入场: 价格突破 N 日 Donchian 通道上轨 → 买入（追随趋势）
  - 出场: 价格跌破 M 日 Donchian 通道下轨 → 卖出（趋势反转）
  - 仓位: Unit = (资金 × 2%) / ATR，波动大的标的仓位轻
  - 止损: 2N 硬止损，严格风控
  - 加仓: 每 0.5N 加 1 Unit（最多 4 次），盈利加仓金字塔

  本次回测:
  - 股票池: {', '.join(all_names)}
  - 默认参数: 入场={ENTRY_PERIOD}日 / 出场={EXIT_PERIOD}日 / ATR={ATR_PERIOD}日
  - 初始资金: CNY {INITIAL_CAPITAL:,.0f}
  - 交易成本: 未计入（实际交易需考虑手续费+滑点）

【回测结果速览】
""")
    for i, name in enumerate(all_names):
        m = all_metrics[i]
        flag = "✓" if m['excess_ret'] > 0 else "✗"
        print(f"  {flag} {name}: 策略 {m['total_ret']*100:+.2f}% | 基准 {m['bench_total_ret']*100:+.2f}% "
              f"| 超额 {m['excess_ret']*100:+.2f}% | 夏普 {m['sharpe']:.2f} | 胜率 {m['win_rate']*100:.1f}%")

    print("""
【策略适应场景与使用心得】

  1. 趋势市中表现优异
     - 海龟法则的核心是"截断亏损，让利润奔跑"（Cut losses short, let profits run）。
     - 在单边趋势行情（无论牛市还是熊市）中能充分吃到波段利润。
     - 在横盘震荡市中，频繁的虚假突破会导致连续亏损（锯齿效应），这是策略
       最大的弱点。

  2. 参数选择影响显著
     - 短周期（入场10-20日）: 信号更灵敏，交易频率高，适合波动较大的个股
       或期货品种。但假信号多，胜率可能偏低。
     - 长周期（入场30-55日）: 过滤噪音，交易频率低，适合趋势明确的长周期
       品种。但可能错过短期行情，入场偏晚。
     - 从参数扫描热力图中可以观察最优参数组合因股票而异。

  3. 止损是生命线
     - 2N（2倍ATR）的止损规则使单笔亏损被严格限制，即使连续止损也不会
       击穿账户。
     - 如果去掉止损，遇到极端行情（如利空暴跌），单笔亏损可能造成不可逆
       的损失。

  4. 心理层面是最大挑战
     - 海龟法则在震荡市中可能出现连续 5-10 次小亏，对交易者的心理素质
       要求极高。很多人在连续止损后放弃策略，恰好错过了下一波大趋势。
     - 成功的海龟交易员必须"无脑执行"，不凭主观判断。

  5. ATR 动态仓位管理
     - 基于 ATR 的仓位计算使高波动标的自动降低仓位，有效控制组合风险。
     - 相比固定股数/固定金额交易更加科学合理。

  6. 改进方向
     - 加入趋势过滤器（如 MA 方向、ADX 阈值），在震荡市中减少无效交易。
     - 引入大盘/板块联动作为过滤条件。
     - 结合基本面（如财报季避开黑天鹅）。
     - 多时间框架确认（如 System1 + System2 同时触发）。
     - 交易成本建模（手续费、滑点）对高频短线参数影响较大。

  7. 海龟法则的精髓
     - 不在于某个神奇参数，而在于整套体系：趋势跟随 + 资金管理 + 止损纪律。
     - 单一指标的胜率可能只有 35-45%，但因为盈亏比高（亏损有限、盈利
       无限），长期期望收益为正。
     - 这正是量化交易的核心思想：用数学期望而非单笔胜负评判策略。

【生成文件清单】
  - turtle_charts/turtle_<股票名>_ep20_xp10.png × 6 张（个股分析图）
  - turtle_charts/turtle_multi_stock_comparison.png（多股票对比）
  - turtle_charts/turtle_param_sweep_<股票名>.png × 3 张（参数扫描热力图）
  - turtle_charts/turtle_trading_summary.csv（回测汇总表）
  - turtle_charts/turtle_param_results_<股票名>.csv × 3 张（参数扫描数据）
""")

    print("\n全部完成！图表和结果已保存到 turtle_charts/ 目录。\n")


if __name__ == '__main__':
    main()
