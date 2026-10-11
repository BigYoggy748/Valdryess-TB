"""
Liquidity Sweep Strategy Backtester (v1)
==========================================
Implements the locked spec from strategy design session:

1. SWEEP: price breaks a 4H/1H structural high/low
   (2-candle swing pattern: up-candle then down-candle = high;
   down-candle then up-candle = low; level = the extreme wick)

2. REVERSAL CONFIRMATION: sweep -> 1-min BOS -> retrace into the FVG
   created by that BOS -> entry on FVG fill.
   Reversal candle validity: range >= 1.5x avg range of prior 20 candles,
   closes beyond the swept level's prior range, within 2 candles of sweep
   (else setup invalidated).

3. RETRACE / CONTINUATION: pullback must not close beyond the original
   swept level. Continuation = fresh 1-min BOS in original direction
   within 5 candles of pullback start (else setup invalidated).

4. INDEX ALIGNMENT FILTER: ES and NQ 5-min trend must agree
   (last 2 completed 5-min candles both up, or both down, on EACH
   instrument) or the trade is skipped.

5. EXIT: nearest opposing 1H swing high/low beyond entry, capped at 3R
   if no level found within 10x stop distance.

6. STOP: a few ticks beyond the swept level itself.

------------------------------------------------------------------
INPUT DATA FORMAT (required)
------------------------------------------------------------------
Two CSV files, one per instrument (e.g. NQ and ES), each with 1-minute
OHLCV bars, columns (case-insensitive):
    timestamp, open, high, low, close, volume
timestamp must be parseable by pandas.to_datetime, and both files must
cover the SAME date range for the alignment filter to work.

This script resamples 1-min data up to 5-min, 1H, and 4H internally --
you do not need to supply separate timeframe files.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
python liquidity_sweep_backtest.py \
    --primary NQ_1min.csv \
    --confirm ES_1min.csv \
    --tick-size 0.25 --tick-value 5.00 --stop-ticks 4 \
    --risk-per-trade 100

Trades are generated on the PRIMARY instrument; the CONFIRM instrument
is used only for the alignment filter.

This is a v1 mechanical translation of a discretionary strategy.
Thresholds (1.5x range, 2-candle window, 5-candle continuation window,
3R cap) are placeholder-but-locked values agreed on during spec design,
NOT guaranteed-optimal parameters. Treat the output as a first read on
whether this rule set has any edge at all, not a final verdict -- and
re-check results by hand against a few flagged trades before trusting
the aggregate stats.
"""

import argparse
import sys
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Data loading / resampling
# ----------------------------------------------------------------------

def load_1min(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df[["open", "high", "low", "close", "volume"]]


def resample(df, rule):
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return out.dropna(subset=["open", "high", "low", "close"])


# ----------------------------------------------------------------------
# Structural swing highs/lows (2-candle pattern) on a given timeframe
# ----------------------------------------------------------------------

def find_swings(df):
    """
    High: candle i is an up-candle (close>open) and candle i+1 is a
    down-candle (close<open) -> level = high of candle i.
    Low: candle i is a down-candle and candle i+1 is an up-candle
    -> level = low of candle i.
    Returns two lists of (timestamp, price) tuples.
    """
    highs, lows = [], []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    idx = df.index
    for i in range(len(df) - 1):
        up_i = c[i] > o[i]
        down_i = c[i] < o[i]
        up_ip1 = c[i + 1] > o[i + 1]
        down_ip1 = c[i + 1] < o[i + 1]
        if up_i and down_ip1:
            highs.append((idx[i], h[i]))
        if down_i and up_ip1:
            lows.append((idx[i], l[i]))
    return highs, lows


# ----------------------------------------------------------------------
# Alignment filter (5-min ES/NQ trend agreement)
# ----------------------------------------------------------------------

def trend_direction_at(df5, ts):
    """
    Last 2 completed 5-min candles before ts, both up or both down.
    Returns 'up', 'down', or None (disagreement / insufficient data).
    """
    prior = df5[df5.index < ts]
    if len(prior) < 2:
        return None
    last2 = prior.iloc[-2:]
    ups = (last2["close"] > last2["open"]).all()
    downs = (last2["close"] < last2["open"]).all()
    if ups:
        return "up"
    if downs:
        return "down"
    return None


# ----------------------------------------------------------------------
# Core strategy scan
# ----------------------------------------------------------------------

def average_range(df1m, i, window=20):
    lo = max(0, i - window)
    seg = df1m.iloc[lo:i]
    if len(seg) == 0:
        return np.nan
    return (seg["high"] - seg["low"]).mean()


def run_backtest(primary_1m, confirm_1m, tick_size, tick_value,
                  stop_ticks, risk_per_trade):

    primary_5m = resample(primary_1m, "5min")
    confirm_5m = resample(confirm_1m, "5min")
    primary_1h = resample(primary_1m, "1h")
    primary_4h = resample(primary_1m, "4h")

    highs_1h, lows_1h = find_swings(primary_1h)
    highs_4h, lows_4h = find_swings(primary_4h)

    # Merge 1H and 4H levels into one sweep-level pool per side
    sweep_highs = sorted(highs_1h + highs_4h, key=lambda x: x[0])
    sweep_lows = sorted(lows_1h + lows_4h, key=lambda x: x[0])

    o = primary_1m["open"].values
    h = primary_1m["high"].values
    l = primary_1m["low"].values
    c = primary_1m["close"].values
    idx = primary_1m.index

    trades = []
    n = len(primary_1m)

    # Precompute swing level values as plain lists for fast lookback checks
    high_levels = [(ts, price) for ts, price in sweep_highs]
    low_levels = [(ts, price) for ts, price in sweep_lows]

    i = 0
    while i < n - 10:
        ts_i = idx[i]

        # find most recent 1H/4H high/low level established strictly before ts_i
        active_high = None
        for ts, price in reversed(high_levels):
            if ts < ts_i:
                active_high = price
                break
        active_low = None
        for ts, price in reversed(low_levels):
            if ts < ts_i:
                active_low = price
                break

        swept_dir = None
        swept_level = None
        if active_high is not None and h[i] > active_high:
            swept_dir = "short"  # swept a high -> expect reversal down
            swept_level = active_high
        elif active_low is not None and l[i] < active_low:
            swept_dir = "long"   # swept a low -> expect reversal up
            swept_level = active_low

        if swept_dir is None:
            i += 1
            continue

        # --- confluence 2: reversal confirmation within 2 candles ---
        avg_rng = average_range(primary_1m, i, 20)
        reversal_idx = None
        for j in range(i, min(i + 3, n)):
            rng = h[j] - l[j]
            if np.isnan(avg_rng) or avg_rng == 0:
                continue
            impulsive = rng >= 1.5 * avg_rng
            if swept_dir == "short":
                closes_beyond = c[j] < swept_level
            else:
                closes_beyond = c[j] > swept_level
            if impulsive and closes_beyond:
                reversal_idx = j
                break

        if reversal_idx is None:
            i += 1
            continue

        # --- confluence 3: retrace must not close beyond swept level,
        #     continuation = fresh 1-min BOS within 5 candles ---
        entry_idx = None
        pullback_start = reversal_idx + 1
        window_end = min(pullback_start + 5, n)
        invalidated = False
        for k in range(pullback_start, window_end):
            if swept_dir == "short" and c[k] > swept_level:
                invalidated = True
                break
            if swept_dir == "long" and c[k] < swept_level:
                invalidated = True
                break
            # continuation BOS: new local extreme beyond reversal candle's extreme
            if swept_dir == "short" and l[k] < l[reversal_idx]:
                entry_idx = k
                break
            if swept_dir == "long" and h[k] > h[reversal_idx]:
                entry_idx = k
                break

        if invalidated or entry_idx is None:
            i = reversal_idx + 1
            continue

        entry_ts = idx[entry_idx]

        # --- confluence 4: ES/NQ 5-min alignment filter ---
        primary_trend = trend_direction_at(primary_5m, entry_ts)
        confirm_trend = trend_direction_at(confirm_5m, entry_ts)
        needed = "down" if swept_dir == "short" else "up"
        if primary_trend != needed or confirm_trend != needed:
            i = entry_idx + 1
            continue

        entry_price = c[entry_idx]

        # --- confluence 6: stop beyond swept level ---
        if swept_dir == "short":
            stop_price = swept_level + stop_ticks * tick_size
            risk_points = stop_price - entry_price
        else:
            stop_price = swept_level - stop_ticks * tick_size
            risk_points = entry_price - stop_price

        if risk_points <= 0:
            i = entry_idx + 1
            continue

        # --- confluence 5: target = nearest opposing 1H swing level,
        #     else cap at 3R ---
        target_pool = low_levels if swept_dir == "short" else high_levels
        candidate_targets = [p for ts, p in target_pool if ts < entry_ts]
        target_price = None
        if swept_dir == "short":
            below = [p for p in candidate_targets if p < entry_price]
            if below:
                target_price = max(below)  # nearest below
        else:
            above = [p for p in candidate_targets if p > entry_price]
            if above:
                target_price = min(above)  # nearest above

        max_target_dist = 10 * risk_points
        if target_price is None or abs(target_price - entry_price) > max_target_dist:
            if swept_dir == "short":
                target_price = entry_price - 3 * risk_points
            else:
                target_price = entry_price + 3 * risk_points

        reward_points = abs(target_price - entry_price)
        r_multiple_target = reward_points / risk_points

        # --- walk forward to see what actually happened ---
        outcome = None
        exit_price = None
        exit_idx = None
        for m in range(entry_idx + 1, n):
            if swept_dir == "short":
                if h[m] >= stop_price:
                    outcome, exit_price, exit_idx = "loss", stop_price, m
                    break
                if l[m] <= target_price:
                    outcome, exit_price, exit_idx = "win", target_price, m
                    break
            else:
                if l[m] <= stop_price:
                    outcome, exit_price, exit_idx = "loss", stop_price, m
                    break
                if h[m] >= target_price:
                    outcome, exit_price, exit_idx = "win", target_price, m
                    break

        if outcome is None:
            i = entry_idx + 1
            continue  # trade never resolved within available data

        r_realized = r_multiple_target if outcome == "win" else -1.0
        pnl_dollars = r_realized * risk_per_trade

        trades.append({
            "entry_time": entry_ts,
            "direction": "short" if swept_dir == "short" else "long",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "swept_level": swept_level,
            "outcome": outcome,
            "exit_time": idx[exit_idx],
            "exit_price": exit_price,
            "r_multiple": r_realized,
            "pnl_dollars": pnl_dollars,
        })

        i = exit_idx + 1  # no overlapping trades

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def summarize(trades_df, risk_per_trade):
    if trades_df.empty:
        print("No trades were generated by this rule set on the supplied data.")
        print("This itself is a result: either the setup is very rare, or a")
        print("threshold (sweep timeframe, impulsive-candle multiplier, etc.)")
        print("is too strict for this instrument/date range. Inspect manually")
        print("before concluding the strategy has no edge.")
        return

    n = len(trades_df)
    wins = (trades_df["outcome"] == "win").sum()
    losses = (trades_df["outcome"] == "loss").sum()
    win_rate = wins / n
    avg_r = trades_df["r_multiple"].mean()
    total_r = trades_df["r_multiple"].sum()
    total_pnl = trades_df["pnl_dollars"].sum()

    equity = trades_df["pnl_dollars"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    # longest losing streak
    streak = 0
    max_streak = 0
    for o in trades_df["outcome"]:
        if o == "loss":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    print("=" * 60)
    print("BACKTEST RESULTS -- Liquidity Sweep Strategy v1")
    print("=" * 60)
    print(f"Total trades:          {n}")
    print(f"Wins / Losses:         {wins} / {losses}")
    print(f"Win rate:              {win_rate:.1%}")
    print(f"Average R per trade:   {avg_r:+.2f}")
    print(f"Total R:               {total_r:+.2f}")
    print(f"Total P&L (${risk_per_trade:.0f} risk/trade): ${total_pnl:,.2f}")
    print(f"Max drawdown:          ${max_dd:,.2f}")
    print(f"Longest losing streak: {max_streak} trades")
    print(f"First trade:           {trades_df['entry_time'].min()}")
    print(f"Last trade:            {trades_df['entry_time'].max()}")
    print("=" * 60)
    print()
    print("READ THIS BEFORE TRUSTING THESE NUMBERS:")
    print("- Spread/slippage/commission are NOT modeled. Real fills will be worse.")
    print("- Assumes instant fill at FVG entry price -- unrealistic in fast moves.")
    print("- Trades are non-overlapping by construction; a live bot could face")
    print("  concurrent setups this backtest never had to resolve.")
    print("- Small trade counts (<30-50) are not statistically reliable.")
    print("- This is a v1 threshold set, not a validated final strategy.")


def main():
    p = argparse.ArgumentParser(description="Liquidity Sweep Strategy Backtester")
    p.add_argument("--primary", required=True, help="1-min CSV for the traded instrument (e.g. NQ)")
    p.add_argument("--confirm", required=True, help="1-min CSV for the alignment-check instrument (e.g. ES)")
    p.add_argument("--tick-size", type=float, default=0.25)
    p.add_argument("--tick-value", type=float, default=5.00)
    p.add_argument("--stop-ticks", type=float, default=4.0, help="Ticks beyond swept level for stop placement")
    p.add_argument("--risk-per-trade", type=float, default=100.0, help="Dollar risk per trade (fixed sizing)")
    p.add_argument("--out", default="/mnt/user-data/outputs/trades.csv", help="Where to save the trade log CSV")
    args = p.parse_args()

    primary_1m = load_1min(args.primary)
    confirm_1m = load_1min(args.confirm)

    trades_df = run_backtest(
        primary_1m, confirm_1m,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        stop_ticks=args.stop_ticks,
        risk_per_trade=args.risk_per_trade,
    )

    summarize(trades_df, args.risk_per_trade)

    if not trades_df.empty:
        trades_df.to_csv(args.out, index=False)
        print(f"\nFull trade log saved to: {args.out}")


if __name__ == "__main__":
    main()