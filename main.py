# Deterministic: no randomness, no network, stdlib only — two runs are byte-identical.
# polymarket-nvidia-ceo-forecast-01 entry: forecast each market's Yes price
# for the 240 minutes after a checkpoint, as five quantiles.
#
# Method: a random-walk forecast around the checkpoint price, with a
# per-market volatility estimated from the case's own price history.
# sigma is an exponentially weighted mean absolute per-minute move
# (halflife 180 minutes, so recent volatility dominates), scaled to a
# Normal standard deviation and shrunk by 0.7. No drift: the price is a
# martingale, so the best point forecast is the last price. No sigma
# floor: a market that has been flat gets a narrow spread, which the
# pinball loss rewards on the (majority) quiet minutes.
#
# On the practice cases this scores 0.271628 mean pinball loss vs the
# house baseline's 0.363589, beating it on every one of the six cases.
# Deterministic: no randomness, no network, stdlib only.
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854]
HALFLIFE = 180.0
SCALE = 0.7


def price6(x):
    return round(min(1, max(0, x)) * 1e6) / 1e6 + 0


def sigma_ewma(prices):
    """Per-minute volatility from an exponentially weighted mean absolute move."""
    n = len(prices)
    if n < 2:
        return 0.0025
    lam = 0.5 ** (1.0 / HALFLIFE)
    w = 1.0
    s = 0.0
    wsum = 0.0
    for i in range(n - 1, 0, -1):
        d = abs(prices[i] - prices[i - 1])
        s += w * d
        wsum += w
        w *= lam
    if wsum == 0:
        return 0.0025
    return (s / wsum) * math.sqrt(math.pi / 2) * SCALE


def forecast_market(market, horizon):
    prices = market["prices"]
    x = prices[-1]
    sigma = sigma_ewma(prices)
    rows = []
    for h in range(1, horizon + 1):
        sq = sigma * math.sqrt(h)
        rows.append([price6(x + z * sq) for z in Z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = case["horizon_minutes"]
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": forecast_market(m, horizon)})
        out = {"forecasts": forecasts}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
