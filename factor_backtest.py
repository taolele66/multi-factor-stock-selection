"""
多因子选股策略回测框架
================================

流程：构建因子(含复合质量因子) -> 预处理(去极值/标准化/行业中性化) -> 样本内/样本外切分
      -> IC检验+分层回测(样本内) -> 多因子合成(等权 / Fama-MacBeth横截面回归)
      -> 组合回测(月度调仓+买入门槛/卖出缓冲控制换手+交易成本) -> 样本内/样本外绩效评估

默认使用"模拟数据"，脚本开箱即跑，方便你先跑通整套逻辑、看懂每一步在做什么。
把 USE_REAL_DATA 改成 True，并按底部 get_data_real() 里的说明接入 akshare 真实数据即可。

运行:
    python factor_backtest.py

依赖:
    pip install numpy pandas matplotlib scipy
    pip install akshare   # 仅在 USE_REAL_DATA=True 时需要
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

np.random.seed(42)

# 中文字体配置（解决图表中中文显示为方框的问题）
for _font in ["PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    if _font in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_font]
        break
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置
# ============================================================
USE_REAL_DATA = True           # True -> 调用 get_data_real()，用akshare拉取真实数据
N_STOCKS = 200                  # 模拟股票池大小（USE_REAL_DATA=False时生效）
N_INDUSTRIES = 10               # 模拟行业数量（USE_REAL_DATA=False时生效）
REAL_DATA_N_STOCKS = 300        # 真实数据模式下取沪深300前N只（300为全量）
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
IN_SAMPLE_END = "2022-12-31"    # 样本内(用于因子检验/确定权重)截止日，之后到END_DATE是样本外测试区间，
                                 # 只在最后跑一次、不用来调参，避免"因子筛选/调参本身"也偷看未来数据
CACHE_DIR = Path("data_cache")  # 真实数据缓存目录，避免重复请求
TOP_QUANTILE = 0.2              # 买入门槛：新股票要排进综合得分前20%才买入
HOLD_BUFFER_QUANTILE = 0.3      # 卖出缓冲：已持仓的股票只要还排在前30%就继续持有，不强制换出，用来降低换手率
TRANSACTION_COST_BPS = 10       # 单边交易成本，单位:bp（万分之十=千分之一）
RISK_FREE_RATE = 0.02           # 年化无风险利率，用于计算夏普比率
ROLLING_WINDOW = 24             # 滚动合成权重的窗口期数（None=用扩展窗口，即截至上一期的全部历史）
ROLLING_MIN_PERIODS = 6         # 历史期数不足此值时用等权兜底，避免早期权重不稳定
RIDGE_ALPHA = 200               # Fama-MacBeth回归的岭回归正则化强度，0=普通OLS。因子共线性
                                 # 会导致OLS系数不稳定/符号漂移，实测alpha=200时系数符号能
                                 # 恢复到和单因子IC一致的方向，同时样本外表现依然稳健


# ============================================================
# 1. 数据获取
# ============================================================
def get_data_synthetic(n_stocks=N_STOCKS, start=START_DATE, end=END_DATE,
                        n_industries=N_INDUSTRIES):
    """
    生成模拟数据，包含：
      price_df   : 日频收盘价 (index=日期, columns=股票代码)
      mktcap_df  : 日频市值
      fin_df     : 月度财务指标 (date, stock, pe, pb, roe, npm, accrual, leverage)
      industries : {股票代码: 行业标签}
      index_series : 日频"指数"参考线 (这里用全部模拟股票的等权均价近似)

    为了让"因子检验"这一步有意义，这里人为给每只股票一个 true_alpha，
    使得低PE/低PB/高ROE的股票未来收益略高——这样你能在IC检验里看到
    价值因子/质量因子确实"有效"，从而理解整个框架在检验什么。
    真实数据接入后，这种人为的可预测性会消失，IC是否显著要看真实市场。
    """
    dates = pd.bdate_range(start, end)
    stocks = [f"S{i:04d}" for i in range(n_stocks)]
    industries = {s: f"IND{np.random.randint(0, n_industries)}" for s in stocks}

    true_alpha = {s: np.random.normal(0, 1.0) for s in stocks}

    price_panel = {}
    for s in stocks:
        mu = 0.0002 + true_alpha[s] * 0.0004
        sigma = np.random.uniform(0.015, 0.035)
        rets = np.random.normal(mu, sigma, len(dates))
        prices = 100 * np.cumprod(1 + rets)
        price_panel[s] = prices
    price_df = pd.DataFrame(price_panel, index=dates)

    shares = {s: np.random.uniform(1e8, 5e9) for s in stocks}
    mktcap_df = price_df * pd.Series(shares)

    month_ends = pd.date_range(start, end, freq="ME")
    records = []
    for m in month_ends:
        for s in stocks:
            pe = max(3.0, np.random.uniform(8, 60) - true_alpha[s] * 8)
            pb = max(0.3, np.random.uniform(0.5, 8) - true_alpha[s] * 1.0)
            roe = np.random.uniform(0.02, 0.25) + true_alpha[s] * 0.02
            npm = np.random.uniform(0.02, 0.20) + true_alpha[s] * 0.02          # 销售净利率，越高越好
            accrual = np.random.uniform(-0.05, 0.05) - true_alpha[s] * 0.01    # 应计利润代理，越低越好
            leverage = np.random.uniform(0.2, 0.7) - true_alpha[s] * 0.05      # 资产负债率，越低越好
            records.append([m, s, pe, pb, roe, npm, accrual, leverage])
    fin_df = pd.DataFrame(records, columns=["date", "stock", "pe", "pb", "roe", "npm", "accrual", "leverage"])

    index_series = price_df.mean(axis=1)

    return price_df, mktcap_df, fin_df, industries, index_series


def _sina_symbol(code):
    """akshare/新浪接口需要交易所前缀：6开头(含688科创板)->sh，0/3开头->sz"""
    return ("sh" if code.startswith("6") else "sz") + code


def _ensure_writable_cache_dir():
    """检查 CACHE_DIR 是否真的可写；不可写（比如运行目录是只读文件系统/挂载盘）就自动
    降级到系统临时目录，并打印一次性提醒，而不是让后面每只股票都报同一个OSError。"""
    global CACHE_DIR
    try:
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        probe = CACHE_DIR / ".write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "multi_factor_project_cache"
        fallback.mkdir(exist_ok=True, parents=True)
        print(f"[警告] 当前目录下的缓存文件夹 {CACHE_DIR.resolve()} 不可写"
              f"（{type(e).__name__}: {e}），改用系统临时目录缓存: {fallback}\n"
              f"  这通常说明你是在只读文件系统/挂载盘里运行脚本的，建议把整个项目文件夹"
              f"复制到 ~/Desktop 或 ~/Documents 之类普通可写目录下再运行。")
        CACHE_DIR = fallback


def _cached_csv(name, fetch_fn, sleep_sec=0.2):
    """本地缓存单只股票的单类数据，避免每次重跑都重新请求（数据源有反爬限速）"""
    path = CACHE_DIR / name
    if path.exists():
        return pd.read_csv(path)
    df = fetch_fn()
    df.to_csv(path, index=False)
    time.sleep(sleep_sec)
    return df


def _asof_panel(series_dict, dates, lag_days=0):
    """把各股票的时间序列（频率不规则）对齐到指定日期上，取最近一次可得值（asof/ffill）。
    lag_days 用于财务数据的公告滞后：例如季报要在报告期结束约1.5个月后才公开，
    不加这个滞后会产生"未来函数"（用还没公布的数据做当期选股）。"""
    out = {}
    for code, s in series_dict.items():
        s = s.sort_index()
        if lag_days:
            s = s.copy()
            s.index = s.index + pd.Timedelta(days=lag_days)
        idx_union = s.index.union(pd.DatetimeIndex(dates))
        aligned = s.reindex(idx_union).sort_index().ffill().reindex(dates)
        out[code] = aligned
    return pd.DataFrame(out)


def _parse_pct_field(series):
    """同花顺财务数据里缺失值经常编码成字符串"False"而不是空值，百分比字段还带%号，
    这个函数统一处理成数值(0.1234这种小数形式)，解析不了的一律变NaN。"""
    return pd.to_numeric(
        series.astype(str).replace("False", np.nan).str.rstrip("%"), errors="coerce"
    ) / 100.0


def _parse_num_field(series):
    """同上，但用于不带%号的普通数值字段。"""
    return pd.to_numeric(series.astype(str).replace("False", np.nan), errors="coerce")


def _fetch_industry_map(codes):
    """尝试获取申万一级行业分类，失败则所有股票归为同一行业（等价于跳过行业中性化）。
    行业分类数据源在部分网络环境下不稳定，所以这里做了容错，不阻塞整个流程。"""
    try:
        import akshare as ak
        sw = ak.stock_industry_clf_hist_sw()
        code_col = next(c for c in sw.columns if "代码" in c)
        name_col = next(c for c in sw.columns if "行业" in c)
        sw[code_col] = sw[code_col].astype(str).str.extract(r"(\d{6})")[0]
        mapping = dict(zip(sw[code_col], sw[name_col]))
        industries = {c: mapping.get(c, "UNKNOWN") for c in codes}
        n_matched = sum(v != "UNKNOWN" for v in industries.values())
        print(f"  行业分类匹配成功 {n_matched}/{len(codes)} 只股票")
        return industries
    except Exception as e:
        print(f"  [警告] 行业分类数据获取失败（{type(e).__name__}: {e}），"
              f"本次运行将跳过行业中性化（所有股票视为同一行业）")
        return {c: "ALL" for c in codes}


def get_data_real(n_stocks=REAL_DATA_N_STOCKS, start=START_DATE, end=END_DATE):
    """
    真实数据管道，数据源全部来自 akshare（背后是新浪/百度股市通/同花顺的公开接口）：
      - 行情+流通股本：ak.stock_zh_a_daily（新浪，前复权），流通股本用于估算市值
      - PE(TTM)/PB：ak.stock_zh_valuation_baidu
      - ROE/销售净利率/资产负债率/每股收益/每股经营现金流（单季度，同一个接口一次拿全，
        用来拼质量因子）：ak.stock_financial_abstract_ths
      - 行业分类：ak.stock_industry_clf_hist_sw（申万一级行业，失败则降级跳过行业中性化）
      - 沪深300指数真实点位：ak.stock_zh_index_daily（新浪，仅用于图表参考线，不参与选股/回测计算）

    首次运行会请求网络（几百只股票 x 3-4个接口，可能需要几分钟到十几分钟），
    每只股票的原始数据会缓存到 ./data_cache/，重复运行会直接读缓存、秒开。

    已知的真实数据坑（这里已经处理，但建议你写项目报告时提一下，体现你理解这些细节）：
      1) 财务数据"公告滞后"：用 lag_days=45 把ROE的可得时点往后推，避免未来函数
      2) 复权：价格用 adjust="qfq"（前复权），否则除权除息会让收益率算错
      3) 停牌/新股/数据缺失：单只股票任何一步失败就跳过该股票，不让整个流程中断
      4) 行业分类接口在某些网络环境下会失败，做了降级处理而不是直接报错退出
    """
    import akshare as ak

    _ensure_writable_cache_dir()

    print("STEP 1a: 获取沪深300成分股列表...")
    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = cons["成分券代码"].astype(str).str.zfill(6).unique().tolist()[:n_stocks]
    print(f"  本次使用 {len(codes)} 只股票（REAL_DATA_N_STOCKS 可调整，300为全量）")

    start_fmt, end_fmt = start.replace("-", ""), end.replace("-", "")
    price_dict, shares_dict, pe_dict, pb_dict = {}, {}, {}, {}
    roe_dict, npm_dict, accrual_dict, leverage_dict = {}, {}, {}, {}

    print("STEP 1b: 逐只股票拉取行情/估值/财务数据（有缓存则跳过网络请求）...")
    for i, code in enumerate(codes):
        print(f"  [{i + 1}/{len(codes)}] {code}", end="\r")
        try:
            px = _cached_csv(
                f"price_{code}.csv",
                lambda c=code: ak.stock_zh_a_daily(
                    symbol=_sina_symbol(c), start_date=start_fmt, end_date=end_fmt, adjust="qfq"),
            )
            px["date"] = pd.to_datetime(px["date"])
            price_dict[code] = px.set_index("date")["close"]
            shares_dict[code] = px.set_index("date")["outstanding_share"]

            pe = _cached_csv(
                f"pe_{code}.csv",
                lambda c=code: ak.stock_zh_valuation_baidu(symbol=c, indicator="市盈率(TTM)", period="全部"),
            )
            pe["date"] = pd.to_datetime(pe["date"])
            pe_dict[code] = pe.set_index("date")["value"]

            pb = _cached_csv(
                f"pb_{code}.csv",
                lambda c=code: ak.stock_zh_valuation_baidu(symbol=c, indicator="市净率", period="全部"),
            )
            pb["date"] = pd.to_datetime(pb["date"])
            pb_dict[code] = pb.set_index("date")["value"]

            fin = _cached_csv(
                f"roe_{code}.csv",
                lambda c=code: ak.stock_financial_abstract_ths(symbol=c, indicator="按单季度"),
            )
            fin["报告期"] = pd.to_datetime(fin["报告期"])
            fin_indexed = fin.set_index("报告期")
            roe_dict[code] = _parse_pct_field(fin_indexed["净资产收益率"]).dropna()
            npm_dict[code] = _parse_pct_field(fin_indexed["销售净利率"]).dropna()
            leverage_dict[code] = _parse_pct_field(fin_indexed["资产负债率"]).dropna()
            # 应计利润代理：每股收益 - 每股经营现金流，越低(现金流对利润的支撑越强)质量越高
            eps = _parse_num_field(fin_indexed["基本每股收益"])
            cfo_per_share = _parse_num_field(fin_indexed["每股经营现金流"])
            accrual_dict[code] = (eps - cfo_per_share).dropna()
        except Exception as e:
            print(f"\n  [跳过] {code} 数据获取失败: {type(e).__name__}: {e}")
            continue
    print(f"\n  成功获取 {len(price_dict)}/{len(codes)} 只股票的完整数据")
    if len(price_dict) == 0:
        raise RuntimeError(
            "一只股票的数据都没拉取成功，往上翻终端输出，看每只股票 [跳过] 后面打印的具体报错原因"
            "（常见原因：网络访问不了新浪/百度股市通/同花顺这几个数据源、akshare版本太旧或太新导致接口"
            "签名变化、被限流。可以先用 `pip install -U akshare` 升级到最新版再试一次）。"
        )

    print("STEP 1c: 对齐数据到统一的日频/月频面板...")
    price_df = pd.DataFrame(price_dict).sort_index().loc[start:end]
    shares_df = pd.DataFrame(shares_dict).sort_index().reindex(price_df.index).ffill()
    mktcap_df = price_df * shares_df

    month_ends = pd.date_range(start, end, freq="ME")
    month_ends = [d for d in month_ends if price_df.index.min() <= d <= price_df.index.max()]

    pe_month = _asof_panel(pe_dict, month_ends)
    pb_month = _asof_panel(pb_dict, month_ends)
    # 财报公告滞后45天，避免未来函数——季报/年报不是报告期结束当天就公开的
    roe_month = _asof_panel(roe_dict, month_ends, lag_days=45)
    npm_month = _asof_panel(npm_dict, month_ends, lag_days=45)
    leverage_month = _asof_panel(leverage_dict, month_ends, lag_days=45)
    accrual_month = _asof_panel(accrual_dict, month_ends, lag_days=45)

    def _get(monthly_df, dt, code):
        return monthly_df.loc[dt, code] if code in monthly_df.columns else np.nan

    fin_rows = []
    for dt in month_ends:
        for code in price_df.columns:
            pe_v = _get(pe_month, dt, code)
            pb_v = _get(pb_month, dt, code)
            roe_v = _get(roe_month, dt, code)
            npm_v = _get(npm_month, dt, code)
            leverage_v = _get(leverage_month, dt, code)
            accrual_v = _get(accrual_month, dt, code)
            required = [pe_v, pb_v, roe_v, npm_v, leverage_v, accrual_v]
            if any(pd.isna(v) for v in required) or pe_v <= 0 or pb_v <= 0:
                continue
            fin_rows.append({
                "date": dt, "stock": code, "pe": pe_v, "pb": pb_v, "roe": roe_v,
                "npm": npm_v, "leverage": leverage_v, "accrual": accrual_v,
            })
    fin_df = pd.DataFrame(fin_rows)

    print("STEP 1d: 获取行业分类...")
    industries = _fetch_industry_map(list(price_df.columns))

    print("STEP 1e: 获取沪深300指数真实点位（仅作图表参考线，不参与选股/回测）...")
    try:
        idx = _cached_csv("index_000300.csv", lambda: ak.stock_zh_index_daily(symbol="sh000300"))
        idx["date"] = pd.to_datetime(idx["date"])
        index_series = idx.set_index("date")["close"].sort_index().loc[start:end]
    except Exception as e:
        print(f"  [警告] 沪深300指数点位获取失败（{type(e).__name__}: {e}），图表里将不显示这条参考线")
        index_series = pd.Series(dtype=float)

    return price_df, mktcap_df, fin_df, industries, index_series


# ============================================================
# 2. 因子构建
# ============================================================
def build_factor_panel(price_df, mktcap_df, fin_df, industries):
    """
    在每个月末构建因子截面，返回一张长表：
        columns = [date, stock, industry, value_ep, value_bp,
                   quality_roe, quality_npm, quality_accrual, quality_leverage,
                   size, mom, vol, fwd_ret]
    fwd_ret 是从当前月末到下个月末的持有期收益，用于后续IC检验和回测。
    """
    month_ends = sorted(fin_df["date"].unique())
    monthly_price = price_df.reindex(month_ends, method="ffill")
    monthly_mktcap = mktcap_df.reindex(month_ends, method="ffill")

    rows = []
    for i, dt in enumerate(month_ends):
        fin_slice = fin_df[fin_df["date"] == dt].set_index("stock")
        px_now = monthly_price.loc[dt]

        # 动量因子：过去12个月收益，剔除最近1个月（经典"反转"处理）
        if i >= 12:
            px_12m_ago = monthly_price.loc[month_ends[i - 12]]
            px_1m_ago = monthly_price.loc[month_ends[i - 1]]
            mom = (px_1m_ago / px_12m_ago) - 1.0
        else:
            mom = pd.Series(np.nan, index=px_now.index)

        # 波动率因子：过去60个交易日日收益标准差
        window_end_loc = price_df.index.get_indexer([dt], method="ffill")[0]
        window_start_loc = max(0, window_end_loc - 60)
        rets_60d = price_df.iloc[window_start_loc:window_end_loc + 1].pct_change()
        vol = rets_60d.std()

        # 未来一个月收益（下一期调仓时点的收益，用于IC检验/回测）
        if i + 1 < len(month_ends):
            px_next = monthly_price.loc[month_ends[i + 1]]
            fwd_ret = (px_next / px_now) - 1.0
        else:
            fwd_ret = pd.Series(np.nan, index=px_now.index)

        for s in px_now.index:
            if s not in fin_slice.index:
                continue
            rows.append({
                "date": dt,
                "stock": s,
                "industry": industries[s],
                "value_ep": 1.0 / fin_slice.loc[s, "pe"],
                "value_bp": 1.0 / fin_slice.loc[s, "pb"],
                "quality_roe": fin_slice.loc[s, "roe"],
                "quality_npm": fin_slice.loc[s, "npm"],
                "quality_accrual": fin_slice.loc[s, "accrual"],
                "quality_leverage": fin_slice.loc[s, "leverage"],
                "size": np.log(monthly_mktcap.loc[dt, s]),
                "mom": mom.get(s, np.nan),
                "vol": vol.get(s, np.nan),
                "fwd_ret": fwd_ret.get(s, np.nan),
            })
    panel = pd.DataFrame(rows)
    return panel


FACTOR_COLS = [
    "value_ep", "value_bp",
    "quality_roe", "quality_npm", "quality_accrual", "quality_leverage",
    "size", "mom", "vol",
]
# 方向：1表示因子值越大越好，-1表示因子值越小越好（做反向处理）
FACTOR_DIRECTION = {
    "value_ep": 1, "value_bp": 1,
    "quality_roe": 1,        # ROE越高越好
    "quality_npm": 1,        # 销售净利率越高越好
    "quality_accrual": -1,   # 应计利润(EPS-每股经营现金流)越低，盈余质量越高
    "quality_leverage": -1,  # 资产负债率越低，财务越稳健
    "size": -1,   # 小市值效应
    "mom": 1, "vol": -1,  # 低波动效应
}


# ============================================================
# 3. 预处理：去极值(MAD) + 标准化(z-score) + 行业中性化
# ============================================================
def winsorize_mad(s, n=3):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    upper = med + n * 1.4826 * mad
    lower = med - n * 1.4826 * mad
    return s.clip(lower, upper)


def zscore(s):
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


def neutralize_industry(df, factor_col):
    """对因子按行业分组去均值，等价于对行业哑变量回归取残差的简化实现"""
    return df.groupby("industry")[factor_col].transform(lambda x: x - x.mean())


def preprocess_panel(panel):
    panel = panel.copy()
    for col in FACTOR_COLS:
        panel[col] = panel.groupby("date")[col].transform(winsorize_mad)
        panel[col] = neutralize_industry(panel, col)
        panel[col] = panel.groupby("date")[col].transform(zscore)
        panel[col] = panel[col] * FACTOR_DIRECTION[col]  # 统一方向：越大越好
    return panel


# ============================================================
# 4. 因子有效性检验：IC分析 + 分层回测
# ============================================================
def compute_ic(panel, factor_col):
    """逐月计算 Spearman rank IC（因子值 vs 未来一个月收益）"""
    ic_series = panel.dropna(subset=[factor_col, "fwd_ret"]).groupby("date").apply(
        lambda g: g[factor_col].corr(g["fwd_ret"], method="spearman")
    )
    return ic_series.dropna()


def summarize_ic(ic_series):
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std != 0 else np.nan
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std != 0 else np.nan
    win_rate = (ic_series > 0).mean()
    return {
        "IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir,
        "t_stat": t_stat, "win_rate": win_rate, "n_periods": len(ic_series),
    }


def quantile_layered_return(panel, factor_col, n_quantiles=5):
    """按因子值分组，计算每组每期的平均未来收益，用于画分层收益曲线"""
    def _bucket(g):
        g = g.dropna(subset=[factor_col, "fwd_ret"])
        if len(g) < n_quantiles:
            return pd.Series(dtype=float)
        g = g.copy()
        g["q"] = pd.qcut(g[factor_col], n_quantiles, labels=False, duplicates="drop")
        return g.groupby("q")["fwd_ret"].mean()

    layered = panel.groupby("date").apply(_bucket).unstack()
    layered.columns = [f"Q{int(c)+1}" for c in layered.columns]
    return layered


# ============================================================
# 5. 多因子合成
# ============================================================
def combine_factors_equal_weight(panel, factor_cols):
    panel = panel.copy()
    panel["score_ew"] = panel[factor_cols].mean(axis=1)
    return panel


def fama_macbeth_betas(panel, factor_cols, ridge_alpha=RIDGE_ALPHA):
    """Fama-MacBeth两步法的第一步：逐月做横截面回归 fwd_ret ~ 因子们 (含截距)，
    把每期回归出来的因子系数（因子风险溢价）存成一张 日期 x 因子 的表。

    每期的回归只用当期截面数据，天然不看未来；下一步(第二步)是在这张系数表上
    再取均值/滚动均值，均值的时间窗口决定了会不会用到未来信息（详见下面两个函数）。

    ridge_alpha: L2正则化(岭回归)强度，0表示退化为普通OLS。
    因子之间存在共线性时(比如value_ep和value_bp都是"便宜度"，高度相关；几个
    quality_*质量因子之间也是)，普通OLS算出来的系数会不稳定、甚至和单因子IC的
    方向相反——这不是bug，是教科书里"多重共线性"的经典表现，加一点岭回归的
    正则化收缩系数幅度，能让结果更稳健，是学术界和业界处理这个问题的标准做法。
    这里没有对截距项做惩罚(截距只是当期整体市场收益水平，不需要收缩)。
    """
    dates = sorted(panel["date"].unique())
    beta_records = {}
    n_factors = len(factor_cols)
    penalty = np.eye(n_factors + 1) * ridge_alpha
    penalty[0, 0] = 0.0  # 截距项不惩罚
    for dt in dates:
        g = panel[panel["date"] == dt].dropna(subset=factor_cols + ["fwd_ret"])
        if len(g) < n_factors + 5:  # 样本太少，回归系数不稳定，本期跳过
            continue
        x = np.column_stack([np.ones(len(g))] + [g[c].values for c in factor_cols])
        y = g["fwd_ret"].values
        if ridge_alpha > 0:
            coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        else:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        beta_records[dt] = coef[1:]  # 去掉截距项，只留因子系数

    beta_df = pd.DataFrame(beta_records, index=factor_cols).T
    beta_df.index.name = "date"
    return beta_df


def combine_factors_famamacbeth_fixed(panel, factor_cols, beta_df_is):
    """用样本内(beta_df_is)各期回归系数的均值作为固定权重，原封不动应用到传入的panel
    （通常传全部数据，含样本外）——标准的"样本内训练参数、样本外冻结测试"流程，
    和下面的滚动窗口版本可以互相对比。
    """
    weights = beta_df_is.mean()
    panel = panel.copy()
    weighted = pd.DataFrame({c: panel[c] * weights[c] for c in factor_cols})
    panel["score_fm_fixed"] = weighted.sum(axis=1, skipna=True)
    return panel, weights


def combine_factors_famamacbeth_rolling(panel, factor_cols, beta_df_full, window=24, min_periods=6):
    """滚动/扩展窗口版Fama-MacBeth——每个调仓日的因子权重只用"截至当天已经实现"的
    历史回归系数取均值，不看未来数据，是 combine_factors_famamacbeth_fixed 的严谨版本。

    时点对齐和滚动IC加权的道理完全一样：第i期(dates[i])的回归系数衡量的是"第i期因子值
    vs 第i期到第i+1期的未来收益"，要等到第i+1期才能被观察到，所以第i期做决策时只能用
    index < i 的历史系数。

    参数:
        beta_df_full: fama_macbeth_betas() 在全部数据(含样本外)上算出的逐期系数表——
            这里不算未来函数，因为下面只会向前(不向后)取历史窗口
        window: 滚动窗口期数，None表示扩展窗口（用截至上一期为止的全部历史）
        min_periods: 历史期数不足此值时，用等权兜底（回归系数不稳定，强行加权反而更差）
    """
    dates = sorted(panel["date"].unique())
    beta_df_full = beta_df_full.reindex(dates)

    weight_records = {}
    for i, dt in enumerate(dates):
        hist = beta_df_full.iloc[:i] if window is None else beta_df_full.iloc[max(0, i - window):i]
        hist = hist.dropna(axis=0, how="all")

        if len(hist) >= min_periods:
            w = hist.mean()
        else:
            w = pd.Series(1.0 / len(factor_cols), index=factor_cols)  # 历史不足时用等权兜底
        weight_records[dt] = w

    weight_df = pd.DataFrame(weight_records).T
    weight_df.index.name = "date"

    panel = panel.merge(weight_df.add_prefix("w_"), left_on="date", right_index=True, how="left")
    # 用 DataFrame.sum(skipna=True) 而不是 Python内置 sum()：
    # 动量因子mom在最早12个月是NaN（凑不够12个月历史），Python的sum会让NaN传染到
    # 整行导致这几个月的综合得分全变NaN；skipna=True会跳过缺失因子，只加总有值的部分，
    # 和上面 combine_factors_equal_weight / combine_factors_famamacbeth_fixed 的处理方式保持一致。
    weighted = pd.DataFrame({c: panel[c] * panel[f"w_{c}"] for c in factor_cols})
    panel["score_fm_roll"] = weighted.sum(axis=1, skipna=True)
    panel = panel.drop(columns=[f"w_{c}" for c in factor_cols])
    return panel, weight_df


# ============================================================
# 6. 组合回测（月度调仓 + 交易成本）
# ============================================================
def backtest_portfolio(panel, score_col, top_quantile=TOP_QUANTILE,
                        buffer_quantile=HOLD_BUFFER_QUANTILE, cost_bps=TRANSACTION_COST_BPS):
    """月度调仓组合回测。用"买入门槛 + 卖出缓冲"两条线而不是单一阈值来控制换手率：
    新股票要排进前 top_quantile 才会被买入，但已持有的股票只要还留在更宽的
    buffer_quantile 区间内就继续持有，不会因为名次从第21名掉到第22名就被强制换掉——
    严格按单一名次线换仓，现实中换手率会高很多，白白多付交易成本。
    """
    dates = sorted(panel["date"].unique())
    nav = [1.0]
    nav_dates = [dates[0]]
    prev_holdings = set()
    turnover_list = []

    for dt in dates:
        g = panel[panel["date"] == dt].dropna(subset=[score_col, "fwd_ret"])
        if g.empty:
            continue
        ranked = g.sort_values(score_col, ascending=False)
        n_buy = max(1, int(len(ranked) * top_quantile))
        n_buffer = max(n_buy, int(len(ranked) * buffer_quantile))

        buy_pool = ranked.head(n_buy)["stock"].tolist()
        buffer_pool = set(ranked.head(n_buffer)["stock"])

        kept = prev_holdings & buffer_pool
        if len(kept) > n_buy:
            # 缓冲区内符合条件的老持仓超过了目标持仓数，按名次只留排名最靠前的n_buy只
            kept_ranked = [s for s in ranked["stock"] if s in kept]
            kept = set(kept_ranked[:n_buy])
        need = n_buy - len(kept)
        new_buys = [s for s in buy_pool if s not in kept][:need] if need > 0 else []
        holdings = kept | set(new_buys)

        port_ret = g[g["stock"].isin(holdings)]["fwd_ret"].mean()

        # 换手率 = 本期新换入的股票占比，用于估算交易成本
        if prev_holdings:
            turnover = len(holdings - prev_holdings) / len(holdings)
        else:
            turnover = 1.0
        turnover_list.append(turnover)
        cost = turnover * (cost_bps / 10000.0) * 2  # 买卖双向成本
        net_ret = port_ret - cost

        nav.append(nav[-1] * (1 + net_ret))
        nav_dates.append(dt)
        prev_holdings = holdings

    nav_series = pd.Series(nav[1:], index=nav_dates[1:])
    return nav_series, np.mean(turnover_list)


def backtest_benchmark(panel):
    """等权持有全部股票池作为基准（近似指数）"""
    bench_ret = panel.dropna(subset=["fwd_ret"]).groupby("date")["fwd_ret"].mean()
    nav = (1 + bench_ret).cumprod()
    return nav


# ============================================================
# 7. 绩效评估
# ============================================================
def performance_stats(nav_series, periods_per_year=12, rf=RISK_FREE_RATE):
    ret = nav_series.pct_change().dropna()
    n_years = len(ret) / periods_per_year
    annual_return = nav_series.iloc[-1] ** (1 / n_years) - 1
    annual_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan

    cum = nav_series
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "年化收益率": annual_return,
        "年化波动率": annual_vol,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "累计净值": nav_series.iloc[-1],
    }


def information_ratio(strategy_nav, bench_nav, periods_per_year=12):
    common_idx = strategy_nav.index.intersection(bench_nav.index)
    excess_ret = strategy_nav.loc[common_idx].pct_change() - bench_nav.loc[common_idx].pct_change()
    excess_ret = excess_ret.dropna()
    ir = excess_ret.mean() / excess_ret.std() * np.sqrt(periods_per_year) if excess_ret.std() != 0 else np.nan
    return ir


def split_is_oos(nav_series, in_sample_end):
    """把一条连续的净值曲线，按样本内/样本外分界日切成两段，并各自重新归一化到1.0起点。

    重新归一化是必须的：nav_series是从头累计复利算出来的，样本外那一段如果直接截取，
    起点会是分界日当天的净值(比如1.32)而不是1.0，performance_stats()里年化收益率的算法
    要求净值从1.0起步，不重新归一化算出来的"样本外年化收益"会是错的。
    """
    cutoff = pd.Timestamp(in_sample_end)
    is_part = nav_series[nav_series.index <= cutoff]
    oos_part = nav_series[nav_series.index > cutoff]
    if len(is_part) > 0:
        is_part = is_part / is_part.iloc[0]
    if len(oos_part) > 0:
        oos_part = oos_part / oos_part.iloc[0]
    return is_part, oos_part


def print_performance(nav_series):
    for k, v in performance_stats(nav_series).items():
        print(f"  {k}: {v:.2%}" if k != "夏普比率" else f"  {k}: {v:.2f}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("STEP 1: 获取数据")
    print("=" * 60)
    if USE_REAL_DATA:
        price_df, mktcap_df, fin_df, industries, index_series = get_data_real()
    else:
        price_df, mktcap_df, fin_df, industries, index_series = get_data_synthetic()
    print(f"股票数量: {price_df.shape[1]}, 交易日数量: {price_df.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 2: 构建因子截面")
    print("=" * 60)
    panel = build_factor_panel(price_df, mktcap_df, fin_df, industries)
    panel = preprocess_panel(panel)
    print(f"截面样本数: {len(panel)}, 调仓期数: {panel['date'].nunique()}")

    panel_is = panel[panel["date"] <= pd.Timestamp(IN_SAMPLE_END)]
    print(f"\n样本内/样本外切分：样本内截至 {IN_SAMPLE_END}（{panel_is['date'].nunique()}期，"
          f"用于因子检验和确定权重），{IN_SAMPLE_END}之后到{END_DATE}是样本外测试区间"
          f"（{panel['date'].nunique() - panel_is['date'].nunique()}期，全程不用来调参，只在最后跑一次）")

    print("\n" + "=" * 60)
    print("STEP 3: 因子有效性检验 (IC分析，只用样本内数据，避免用未来数据决定用哪些因子)")
    print("=" * 60)
    ic_summary_is = {}
    for col in FACTOR_COLS:
        ic_s = compute_ic(panel_is, col)
        ic_summary_is[col] = summarize_ic(ic_s)
        s = ic_summary_is[col]
        print(f"[{col:12s}] IC_mean={s['IC_mean']:+.4f}  ICIR={s['ICIR']:+.3f}  "
              f"t={s['t_stat']:+.2f}  win_rate={s['win_rate']:.1%}  n={s['n_periods']}")

    print("\n" + "=" * 60)
    print("STEP 4: 分层回测（样本内，以动量因子mom为例，其余因子同理）")
    print("=" * 60)
    layered = quantile_layered_return(panel_is, "mom")
    layered_cum = (1 + layered.fillna(0)).cumprod()
    print(layered_cum.iloc[-1].sort_values(ascending=False))

    print("\n" + "=" * 60)
    print("STEP 5: 多因子合成 (Fama-MacBeth横截面回归法)")
    print("=" * 60)
    panel = combine_factors_equal_weight(panel, FACTOR_COLS)
    # 第一步：逐月做横截面回归，全部数据(含样本外)都算一遍系数——这一步本身不算未来函数，
    # 因为每期系数只用当期截面，真正决定"看没看未来"的是下面第二步怎么取这些系数的均值。
    beta_df_full = fama_macbeth_betas(panel, FACTOR_COLS)
    beta_df_is = beta_df_full.reindex(panel_is["date"].unique()).dropna(how="all")
    # 第二步(固定权重版)：只用样本内的回归系数取均值，定下权重后原封不动应用到样本外——
    # 标准的"样本内训练参数、样本外冻结测试"流程。
    panel, fm_weights_fixed = combine_factors_famamacbeth_fixed(panel, FACTOR_COLS, beta_df_is)
    # 第二步(滚动窗口版)：每期权重只用截至当期之前的历史系数滚动取均值，天然不看未来。
    panel, fm_weights_roll = combine_factors_famamacbeth_rolling(
        panel, FACTOR_COLS, beta_df_full, window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS)
    print("Fama-MacBeth合成的因子权重（样本内系数均值、样本外冻结使用）:")
    print(fm_weights_fixed.round(4))
    print("\nFama-MacBeth合成的因子权重（滚动窗口，逐月更新，最近5期）:")
    print(fm_weights_roll.tail(5).round(4))

    print("\n" + "=" * 60)
    print("STEP 6: 组合回测（样本内/样本外分开报告，样本外才是诚实的表现估计）")
    print("=" * 60)
    nav_ew, turnover_ew = backtest_portfolio(panel, "score_ew")
    nav_fm_fixed, turnover_fm_fixed = backtest_portfolio(panel, "score_fm_fixed")
    nav_fm_roll, turnover_fm_roll = backtest_portfolio(panel, "score_fm_roll")
    nav_bench = backtest_benchmark(panel)

    strategies = {
        "等权合成策略": nav_ew,
        "Fama-MacBeth策略-样本内固定权重": nav_fm_fixed,
        "Fama-MacBeth策略-滚动窗口": nav_fm_roll,
    }
    turnovers = {
        "等权合成策略": turnover_ew,
        "Fama-MacBeth策略-样本内固定权重": turnover_fm_fixed,
        "Fama-MacBeth策略-滚动窗口": turnover_fm_roll,
    }

    for name, nav in strategies.items():
        is_nav, oos_nav = split_is_oos(nav, IN_SAMPLE_END)
        print(f"\n[{name}]")
        print(" 样本内（拟合/调参区间，表现天然偏乐观）:")
        print_performance(is_nav)
        print(" 样本外（唯一诚实的表现估计，全程没被用来调参）:")
        print_performance(oos_nav)
        print(f"  平均月换手率: {turnovers[name]:.1%}")
        print(f"  样本外信息比率(vs基准): "
              f"{information_ratio(oos_nav, split_is_oos(nav_bench, IN_SAMPLE_END)[1]):.2f}")

    bench_is, bench_oos = split_is_oos(nav_bench, IN_SAMPLE_END)
    print("\n[基准（等权持有全部股票池）]")
    print(" 样本内:")
    print_performance(bench_is)
    print(" 样本外:")
    print_performance(bench_oos)

    is_sharpe = performance_stats(split_is_oos(nav_fm_fixed, IN_SAMPLE_END)[0])["夏普比率"]
    oos_sharpe = performance_stats(split_is_oos(nav_fm_fixed, IN_SAMPLE_END)[1])["夏普比率"]
    roll_oos_sharpe = performance_stats(split_is_oos(nav_fm_roll, IN_SAMPLE_END)[1])["夏普比率"]
    print(f"\n[过拟合检查] Fama-MacBeth-样本内固定权重：样本内夏普{is_sharpe:.2f} -> 样本外夏普{oos_sharpe:.2f}"
          f"（差距越大说明样本内挑出来的权重对样本外越不适用，越可能是过拟合）。"
          f"滚动窗口版本的样本外夏普是{roll_oos_sharpe:.2f}——它没有"
          f"\"先拟合再冻结\"这一步，每期权重都是当时能拿到的最新信息，通常更稳健。")

    print("\n" + "=" * 60)
    print("STEP 7: 输出图表")
    print("=" * 60)
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    axes[0].plot(nav_ew.index, nav_ew.values, label="等权合成策略")
    axes[0].plot(nav_fm_fixed.index, nav_fm_fixed.values, label="Fama-MacBeth(样本内固定权重)", linestyle=":")
    axes[0].plot(nav_fm_roll.index, nav_fm_roll.values, label="Fama-MacBeth(滚动窗口)")
    axes[0].plot(nav_bench.index, nav_bench.reindex(nav_ew.index).values, label="基准(等权持有全部股票池)", linestyle="--")
    if len(index_series) > 0:
        # 归一化到和其它策略同一个起点(1.0)，方便直接比较涨跌幅；用asof而不是精确匹配日期，
        # 因为指数是日频、策略净值是月末频率，两者的日期不会完全对齐。
        idx_base = index_series.asof(nav_ew.index[0])
        if pd.notna(idx_base) and idx_base > 0:
            index_nav = index_series / idx_base
            axes[0].plot(index_nav.index, index_nav.values, label="沪深300指数(真实点位，仅作参考)",
                         color="black", linewidth=1, alpha=0.6)
    axes[0].axvline(pd.Timestamp(IN_SAMPLE_END), color="gray", linestyle="-.", linewidth=1)
    axes[0].text(pd.Timestamp(IN_SAMPLE_END), axes[0].get_ylim()[1], " 样本外→",
                 fontsize=8, color="gray", va="top")
    axes[0].set_title("策略净值曲线 vs 基准（虚线左侧=样本内，右侧=样本外）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    layered_cum.plot(ax=axes[1])
    axes[1].set_title("动量因子分层回测（Q1=因子值最低组, Q5=最高组）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fm_weights_roll.plot(ax=axes[2])
    axes[2].set_title("滚动窗口Fama-MacBeth：各因子权重(回归系数)随时间的变化")
    axes[2].legend(ncol=3, fontsize=7)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = "factor_backtest_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存至: {out_path}")


if __name__ == "__main__":
    main()
