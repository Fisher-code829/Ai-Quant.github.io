#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Tushare HTTP API 批量获取股票日线数据，生成 JSON 供前端看板读取
用法: python fetch_tushare_data.py
输出: data/tushare_stocks.json
"""

import json
import os
import sys
import time
import requests

# Fix Windows GBK encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Tushare HTTP API (from MCP config)
TUSHARE_URL = "https://api.tushare.pro"
TOKEN = "1ea9b43044d7c6ce05876aed092ed7e24af3774ef4946d67f599a8dd"

# ======================== 股票列表 ========================
A_STOCKS = {
    "600519.SH": "贵州茅台", "300750.SZ": "宁德时代", "300251.SZ": "光线传媒",
    "002594.SZ": "比亚迪", "601318.SH": "中国平安", "600036.SH": "招商银行",
    "600900.SH": "长江电力", "601012.SH": "隆基绿能", "600276.SH": "恒瑞医药",
    "603288.SH": "海天味业", "601888.SH": "中国中免", "600030.SH": "中信证券",
    "601899.SH": "紫金矿业", "600887.SH": "伊利股份", "000858.SZ": "五粮液",
    "002415.SZ": "海康威视", "300059.SZ": "东方财富", "000001.SZ": "平安银行",
    "002475.SZ": "立讯精密", "000333.SZ": "美的集团", "000725.SZ": "京东方A",
    "002271.SZ": "东方雨虹", "600809.SH": "山西汾酒", "601688.SH": "华泰证券",
    "002352.SZ": "顺丰控股", "300124.SZ": "汇川技术", "300274.SZ": "阳光电源",
    "300760.SZ": "迈瑞医疗", "000568.SZ": "泸州老窖", "688981.SH": "中芯国际",
    "688111.SH": "金山办公", "688008.SH": "澜起科技", "688012.SH": "中微公司",
    "688036.SH": "传音控股", "688599.SH": "天合光能",
}

ETFS = {
    "510300.SH": "沪深300ETF", "510050.SH": "上证50ETF", "510500.SH": "中证500ETF",
    "159915.SZ": "创业板ETF", "512000.SH": "券商ETF", "512880.SH": "证券ETF",
    "510900.SH": "H股ETF", "515030.SH": "新能源车ETF", "515790.SH": "光伏ETF",
    "512690.SH": "酒ETF", "512480.SH": "半导体ETF",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def tushare_api(api_name, **params):
    """调用 Tushare HTTP API"""
    payload = {
        "api_name": api_name,
        "token": TOKEN,
        "params": params,
        "fields": "",
    }
    try:
        resp = requests.post(TUSHARE_URL, json=payload, timeout=60)
        result = resp.json()
        if result.get("code") != 0:
            raise Exception(f"Tushare API error: {result.get('msg','unknown')}")
        return result.get("data", {})
    except Exception as e:
        print(f"  API call failed: {e}")
        raise


def fetch_batch(codes, start_date="20230101", end_date="20260709", api_name="daily"):
    """批量获取日线，返回 {ts_code: [records]}"""
    all_data = {}
    ts_code_str = ",".join(codes)
    try:
        data = tushare_api(
            api_name=api_name,
            ts_code=ts_code_str,
            start_date=start_date,
            end_date=end_date,
        )
        if not data or not data.get("items"):
            print(f"  No data returned for batch")
            return all_data

        items = data["items"]
        fields = data.get("fields", [])

        # 按 ts_code 分组
        code_idx = fields.index("ts_code") if "ts_code" in fields else 0
        date_idx = fields.index("trade_date") if "trade_date" in fields else 1
        open_idx = fields.index("open")
        close_idx = fields.index("close")
        high_idx = fields.index("high")
        low_idx = fields.index("low")
        vol_idx = fields.index("vol") if "vol" in fields else -1

        for code in codes:
            recs = [item for item in items if item[code_idx] == code]
            if not recs:
                print(f"  [W] {code}: no data")
                continue
            # Sort by date descending, then reverse
            recs.sort(key=lambda x: x[date_idx])
            records = []
            for r in recs:
                records.append({
                    "date": str(r[date_idx]),
                    "open": float(r[open_idx]),
                    "close": float(r[close_idx]),
                    "high": float(r[high_idx]),
                    "low": float(r[low_idx]),
                    "vol": float(r[vol_idx]) if vol_idx >= 0 else 0,
                })
            all_data[code] = records
            print(f"  OK {code}: {len(records)} records")
        time.sleep(0.5)
    except Exception as e:
        print(f"  Batch failed: {e}")
    return all_data


def main():
    print("=" * 60)
    print("Fetching data from Tushare HTTP API")
    print("=" * 60)

    all_stock_data = {}

    # Fetch A-shares in batches of 6
    stock_codes = list(A_STOCKS.keys())
    batch_size = 6
    print(f"\n[1] A-shares ({len(stock_codes)} stocks, batch size={batch_size})")
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i+batch_size]
        print(f"  Batch {i//batch_size+1}: {','.join(batch)}")
        try:
            result = fetch_batch(batch)
            all_stock_data.update(result)
        except Exception as e:
            print(f"  Batch failed: {e}")

    # Fetch ETFs - use daily API (fund_daily needs higher permissions)
    etf_codes = list(ETFS.keys())
    print(f"\n[2] ETFs ({len(etf_codes)} stocks, using daily API)")
    for i in range(0, len(etf_codes), batch_size):
        batch = etf_codes[i:i+batch_size]
        print(f"  Batch {i//batch_size+1}: {','.join(batch)}")
        try:
            result = fetch_batch(batch, api_name="daily")
            all_stock_data.update(result)
        except Exception as e:
            print(f"  ETF batch failed: {e} (需要更高的Tushare积分权限)")

    # Build meta
    meta = {}
    for code, name in {**A_STOCKS, **ETFS}.items():
        code6 = code.split(".")[0]
        meta[code6] = {
            "name": name,
            "ts_code": code,
            "market": "SH" if code.endswith(".SH") else "SZ",
            "category": "stock" if code in A_STOCKS else "etf",
        }

    # Build output
    output = {
        "meta": meta,
        "stocks": {},
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for code, records in all_stock_data.items():
        code6 = code.split(".")[0]
        output["stocks"][code6] = records

    # Write JSON
    json_path = os.path.join(OUTPUT_DIR, "tushare_stocks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in output["stocks"].values())
    print(f"\n{'='*60}")
    print(f"DONE: {len(output['stocks'])} symbols, {total} daily records")
    print(f"File: {json_path}")
    print(f"Size: {os.path.getsize(json_path) / 1024:.1f} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
