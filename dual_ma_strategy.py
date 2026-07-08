#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双均线交叉策略 — 完整回测分析
================================
支持多股票、多参数组合的回测对比，自动生成可视化图表和分析报告。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import warnings, os, sys

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────── 1. 加载数据 ───────────────
def load_data(csv_path, name):
    """加载CSV股价数据，统一格式"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['name'] = name
    return df[['date','name','open','high','low','close','volume']]

# ─────────────── 2. 计算均线 ───────────────
def calc_ma(df, short=5, long=15):
    """计算短/长均线"""
    df = df.copy()
    df['MA_short'] = df['close'].rolling(window=short, min_periods=1).mean()
    df['MA_long']  = df['close'].rolling(window=long,  min_periods=1).mean()
    return df

# ─────────────── 3. 计算买卖信号 ───────────────
def calc_signals(df):
    """
    金叉(MA_short上穿MA_long) → 买入信号 +1
    死叉(MA_short下穿MA_long) → 卖出信号 -1
    """
    df = df.copy()
    df['signal'] = 0
    df['prev_short'] = df['MA_short'].shift(1)
    df['prev_long']  = df['MA_long'].shift(1)
    
    golden = (df['prev_short'] <= df['prev_long']) & (df['MA_short'] > df['MA_long'])
    death  = (df['prev_short'] >= df['prev_long']) & (df['MA_short'] < df['MA_long'])
    
    df.loc[golden, 'signal'] = 1
    df.loc[death,  'signal'] = -1
    
    # 持仓状态
    df['position'] = 0
    in_position = False
    for i in range(len(df)):
        if df.loc[i, 'signal'] == 1:
            in_position = True
        elif df.loc[i, 'signal'] == -1:
            in_position = False
        df.loc[i, 'position'] = 1 if in_position else 0
    return df

# ─────────────── 4. 模拟交易与回测 ───────────────
def backtest(df):
    """基于信号模拟交易，计算核心指标"""
    df = df.copy()
    df['daily_ret'] = df['close'].pct_change()
    df['strategy_ret'] = df['daily_ret'] * df['position'].shift(1)
    
    # 累计收益
    df['cum_market'] = (1 + df['daily_ret'].fillna(0)).cumprod()
    df['cum_strategy'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    
    # ── 核心指标计算 ──
    n_years = len(df) / 252
    market_tot  = df['cum_market'].iloc[-1] - 1
    strategy_tot = df['cum_strategy'].iloc[-1] - 1
    market_ann  = (1 + market_tot) ** (1 / max(n_years, 0.1)) - 1
    strategy_ann = (1 + strategy_tot) ** (1 / max(n_years, 0.1)) - 1
    
    # 最大回撤
    def max_drawdown(cum_series):
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        return dd.min()
    
    mkt_mdd = max_drawdown(df['cum_market'])
    str_mdd = max_drawdown(df['cum_strategy'])
    
    # 夏普比率
    rf = 0.02
    excess = df['strategy_ret'].fillna(0) - rf / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    
    # 胜率
    trades = []
    in_trade = False; entry_price = 0
    for i, row in df.iterrows():
        if row['signal'] == 1 and not in_trade:
            entry_price = row['close']; in_trade = True
        elif row['signal'] == -1 and in_trade:
            trades.append(row['close'] / entry_price - 1)
            in_trade = False
    if in_trade:
        trades.append(df['close'].iloc[-1] / entry_price - 1)
    
    win_rate = sum(1 for t in trades if t > 0) / max(len(trades), 1)
    avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
    avg_loss = np.mean([t for t in trades if t <= 0]) if any(t <= 0 for t in trades) else 0
    profit_factor = abs(sum(t for t in trades if t > 0) / sum(t for t in trades if t <= 0)) if sum(t for t in trades if t <= 0) != 0 else float('inf')
    
    # 日内波动率
    df['intra_vol'] = (df['high'] - df['low']) / df['open']
    
    return {
        'trades': len(trades),
        'market_total': market_tot, 'strategy_total': strategy_tot,
        'market_ann': market_ann, 'strategy_ann': strategy_ann,
        'market_mdd': mkt_mdd, 'strategy_mdd': str_mdd,
        'sharpe': sharpe, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'daily_ret_std': df['daily_ret'].std(),
        'mean_intra_vol': df['intra_vol'].mean(),
        'df': df, 'trades_list': trades
    }

# ─────────────── 5. 可视化 ───────────────
def plot_strategy(result, stock_name, short, long, save_path):
    """绘制策略可视化图：股价+均线+买卖信号"""
    df = result['df']
    df_v = df[df['MA_long'] > df['MA_long'].iloc[0]].copy()  # 跳过均线初始段
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'双均线策略 ({stock_name})  —  MA{short} / MA{long}', fontsize=18, fontweight='bold', y=0.98)
    
    # ── 上图：股价 + 均线 + 信号 ──
    ax1.plot(df_v['date'], df_v['close'], color='#333', linewidth=1.2, label='收盘价', alpha=0.9)
    ax1.plot(df_v['date'], df_v['MA_short'], color='#3498db', linewidth=1.2, linestyle='-', label=f'MA{short}')
    ax1.plot(df_v['date'], df_v['MA_long'], color='#e74c3c', linewidth=1.5, linestyle='-', label=f'MA{long}')
    
    # 标记买卖信号
    buys  = df[df['signal'] == 1]
    sells = df[df['signal'] == -1]
    ax1.scatter(buys['date'],  buys['close'] * 0.97,  marker='^', s=80, c='#e74c3c',
                edgecolors='white', linewidths=1, zorder=5, label=f'买入 (金叉, {len(buys)}次)')
    ax1.scatter(sells['date'], sells['close'] * 1.03, marker='v', s=80, c='#2ecc71',
                edgecolors='white', linewidths=1, zorder=5, label=f'卖出 (死叉, {len(sells)}次)')
    
    ax1.set_ylabel('价格 (元)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # ── 下图：持仓状态 + 日收益率 ──
    ax2.fill_between(df['date'], 0, df['position'], color='#e74c3c', alpha=0.15, label='持仓')
    ax2.bar(df['date'], df['daily_ret'].fillna(0) * 100, 
            color=['#e74c3c' if v >= 0 else '#2ecc71' for v in df['daily_ret'].fillna(0)],
            width=0.8, alpha=0.5, label='日收益率(%)')
    ax2.axhline(0, color='#333', linewidth=0.8)
    ax2.set_ylabel('日收益率 (%)', fontsize=13)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

def plot_comparison(results_dict, save_path):
    """多参数回测对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle('双均线策略 — 不同参数组合回测对比', fontsize=18, fontweight='bold')
    
    labels = list(results_dict.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # 累计收益曲线
    for i, (label, r) in enumerate(results_dict.items()):
        axes[0,0].plot(r['df']['date'], r['df']['cum_strategy'], color=colors[i%len(colors)],
                       linewidth=1.5, label=label, alpha=0.85)
    axes[0,0].plot(r['df']['date'], r['df']['cum_market'], color='gray', linewidth=2,
                   linestyle='--', label='买入持有', alpha=0.7)
    axes[0,0].set_title('图A: 累计净值曲线', fontsize=14)
    axes[0,0].set_ylabel('净值'); axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)
    
    # 总收益 / 年化收益 柱状图
    x = np.arange(len(labels))
    w = 0.35
    strat_tot  = [v['strategy_total']*100 for v in results_dict.values()]
    strat_ann  = [v['strategy_ann']*100 for v in results_dict.values()]
    mkt_tot    = [v['market_total']*100 for v in results_dict.values()]
    
    axes[0,1].bar(x - w/2, strat_tot, w, color='#e74c3c', alpha=0.85, label='策略总收益(%)')
    axes[0,1].bar(x + w/2, mkt_tot,   w, color='gray', alpha=0.5, label='买入持有(%)')
    axes[0,1].set_title('图B: 总收益率对比', fontsize=14)
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(labels, fontsize=9)
    axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].axhline(0, color='#333', linewidth=0.8)
    
    # 最大回撤
    mdd_vals = [v['strategy_mdd']*100 for v in results_dict.values()]
    bars = axes[1,0].bar(labels, mdd_vals, color=['#e74c3c' if m < -15 else '#f39c12' if m < -10 else '#3498db' for m in mdd_vals], alpha=0.85)
    axes[1,0].set_title('图C: 最大回撤 (%)', fontsize=14)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mdd_vals):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                       f'{val:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # 夏普比率 + 胜率
    sharpe_vals = [v['sharpe'] for v in results_dict.values()]
    win_rates   = [v['win_rate']*100 for v in results_dict.values()]
    ax_sh = axes[1,1].twinx()
    bars2 = axes[1,1].bar(x - w/2, sharpe_vals, w, color='#3498db', alpha=0.85, label='夏普比率')
    bars3 = ax_sh.bar(x + w/2, win_rates, w, color='#2ecc71', alpha=0.7, label='胜率(%)')
    axes[1,1].set_title('图D: 夏普比率 & 胜率', fontsize=14)
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(labels, fontsize=9)
    axes[1,1].set_ylabel('夏普比率', color='#3498db', fontsize=12)
    ax_sh.set_ylabel('胜率 (%)', color='#2ecc71', fontsize=12)
    axes[1,1].legend(loc='upper left', fontsize=9); ax_sh.legend(loc='upper right', fontsize=9)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, sharpe_vals):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

# ─────────────── 6. 主程序 ───────────────
def main():
    print("=" * 65)
    print("  双均线交叉策略 — 完整回测分析")
    print("=" * 65)
    
    # 股票配置
    stocks = {
        '光线传媒(300251)': os.path.join(OUTPUT_DIR, '光线传媒_300251_daily.csv'),
        '贵州茅台(600519)': os.path.join(OUTPUT_DIR, 'stock_600519_daily.csv'),
        '宁德时代(300750)': os.path.join(OUTPUT_DIR, 'stock_300750_daily.csv'),
    }
    
    # 均线参数组合
    ma_combos = [
        (5, 15, '(5,15)'),
        (5, 20, '(5,20)'),
        (10, 30, '(10,30)'),
        (5, 10, '(5,10)'),
    ]
    
    all_results = {}  # key: "股票名_MA(5,15)"
    
    for stock_name, csv_path in stocks.items():
        print(f"\n{'─'*60}")
        print(f"  ▸ 股票: {stock_name}")
        print(f"{'─'*60}")
        
        df_raw = load_data(csv_path, stock_name)
        print(f"  数据: {len(df_raw)} 个交易日 | {df_raw['date'].iloc[0].date()} ~ {df_raw['date'].iloc[-1].date()}")
        
        for short, long, label in ma_combos:
            key = f"{stock_name} MA{label}"
            df_ma  = calc_ma(df_raw, short, long)
            df_sig = calc_signals(df_ma)
            result = backtest(df_sig)
            all_results[key] = result
            
            n_buy = (df_sig['signal'] == 1).sum()
            n_sell = (df_sig['signal'] == -1).sum()
            
            print(f"\n  MA{label} | 交易{result['trades']}次 | 金叉{n_buy}次 死叉{n_sell}次")
            print(f"    策略总收益: {result['strategy_total']*100:+.2f}%  |  买入持有: {result['market_total']*100:+.2f}%")
            print(f"    策略年化: {result['strategy_ann']*100:+.2f}%  |  市场年化: {result['market_ann']*100:+.2f}%")
            print(f"    最大回撤: {result['strategy_mdd']*100:.2f}%  |  夏普比率: {result['sharpe']:.2f}")
            print(f"    胜率: {result['win_rate']*100:.1f}%  |  盈亏比: {result['profit_factor']:.2f}")
    
    # ── 生成可视化图表 ──
    print(f"\n{'='*60}")
    print("  生成可视化图表...")
    print(f"{'='*60}")
    
    charts_dir = os.path.join(OUTPUT_DIR, 'strategy_charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 图1：每只股票默认参数(5,15)的策略图
    primary_key = None
    for stock_name in stocks.keys():
        key = f"{stock_name} MA(5,15)"
        if key in all_results:
            result = all_results[key]
            path = os.path.join(charts_dir, f'signal_{stock_name[:4]}_5_15.png')
            plot_strategy(result, stock_name, 5, 15, path)
            print(f"  ✓ 信号图: {stock_name}")
            if primary_key is None:
                primary_key = key
    
    # 图2：光线传媒不同参数对比
    gx_keys = {k: v for k, v in all_results.items() if '光线传媒' in k}
    if gx_keys:
        path = os.path.join(charts_dir, 'comparison_光线传媒.png')
        plot_comparison(gx_keys, path)
        print(f"  ✓ 参数对比图: 光线传媒")
    
    # 图3：三只股票(5,15)策略横比
    cross_keys = {k: v for k, v in all_results.items() if 'MA(5,15)' in k}
    if cross_keys:
        path = os.path.join(charts_dir, 'comparison_cross_stocks.png')
        plot_comparison(cross_keys, path)
        print(f"  ✓ 跨股票对比图: 三只股票 MA(5,15)")
    
    # ── 生成汇总表 ──
    print(f"\n{'='*60}")
    print("  策略回测汇总表")
    print(f"{'='*60}")
    
    summary_rows = []
    for key, r in all_results.items():
        summary_rows.append({
            '参数组合': key,
            '交易次数': r['trades'],
            '策略总收益(%)': f"{r['strategy_total']*100:+.2f}",
            '买入持有(%)': f"{r['market_total']*100:+.2f}",
            '策略年化(%)': f"{r['strategy_ann']*100:+.2f}",
            '最大回撤(%)': f"{r['strategy_mdd']*100:.2f}",
            '夏普比率': f"{r['sharpe']:.2f}",
            '胜率(%)': f"{r['win_rate']*100:.1f}",
            '盈亏比': f"{r['profit_factor']:.2f}",
            '日收益标准差': f"{r['daily_ret_std']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(OUTPUT_DIR, '双均线策略_汇总表.csv')
    summary_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"  汇总表已保存: {csv_out}")
    print()
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("  完成！所有图表保存在 strategy_charts/ 目录")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
