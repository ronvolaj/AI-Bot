"""
News + trend trading signal bot.

What it does, for every symbol in your WATCHLIST:
  1. Downloads the latest news headlines about it.
  2. Reads them and scores each one from -1 (very bad news) to +1 (very good news).
     - If you set ANTHROPIC_API_KEY, Claude reads the headlines (much smarter).
     - Otherwise it uses VADER, a free word-based sentiment scorer (simpler).
  3. Checks the price trend (price vs 50-day average, last 20 days' momentum).
  4. Combines news + trend into one score and prints BUY / SELL / HOLD,
     ranked from strongest buy to strongest sell, with the reasons.
  5. Saves every signal to signals_log.csv. Run `python news_trading_bot.py check`
     later to see whether past signals were actually right.

IMPORTANT: these are mechanical signals from a simple formula, not financial advice.
Test them for weeks (use the `check` command) before acting on any of them.

SETUP
  pip install yfinance pandas vaderSentiment anthropic requests alpaca-py
  Optional (better news reading):  export ANTHROPIC_API_KEY="your_key"
                                    (get one at console.anthropic.com)
  Optional (signals on your phone): export TELEGRAM_TOKEN="..." and TELEGRAM_CHAT_ID="..."
  Optional (fake-money test trades): export ALPACA_KEY="..." and ALPACA_SECRET="..."
                                     (free paper account at alpaca.markets)

USAGE
  python news_trading_bot.py          -> scan watchlist and print signals
  python news_trading_bot.py check    -> grade past signals against what prices did
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

# ---------------- Settings you can change ----------------
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "SPY", "BTC-USD"]
NEWS_WEIGHT = 0.6        # how much the news counts (0-1); trend gets the rest
BUY_THRESHOLD = 0.25     # combined score above this -> BUY
SELL_THRESHOLD = -0.25   # combined score below this -> SELL
MAX_HEADLINES = 10       # headlines read per symbol
LOG_FILE = "signals_log.csv"
CLAUDE_MODEL = "claude-sonnet-5-5"
# ---------------------------------------------------------


# ---------- 1. News ----------
def get_headlines(symbol: str) -> list[str]:
    try:
        items = yf.Ticker(symbol).news or []
    except Exception as e:
        print(f"  [{symbol}] could not fetch news: {e}")
        return []
    headlines = []
    for item in items[:MAX_HEADLINES]:
        content = item.get("content", item)  # yfinance changed formats over time
        title = content.get("title") or ""
        summary = content.get("summary") or ""
        if title:
            headlines.append(f"{title}. {summary}".strip()[:400])
    return headlines


# ---------- 2. Sentiment ----------
def score_with_claude(symbol: str, headlines: list[str]) -> tuple[float, str]:
    import anthropic
    client = anthropic.Anthropic()
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = (
        f"You are a financial news analyst. Here are recent headlines about {symbol}:\n\n"
        f"{numbered}\n\n"
        f"How is this news likely to affect {symbol}'s price over the next few days? "
        "Ignore headlines that aren't really about it. Respond with JSON only, no other text: "
        '{"score": <number from -1 (very negative) to 1 (very positive)>, '
        '"reason": "<one short sentence>"}'
    )
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=200,
                                 messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    text = text[text.find("{"): text.rfind("}") + 1]
    data = json.loads(text)
    return max(-1.0, min(1.0, float(data["score"]))), data.get("reason", "")


_vader = None
def score_with_vader(headlines: list[str]) -> tuple[float, str]:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    scores = [_vader.polarity_scores(h)["compound"] for h in headlines]
    avg = sum(scores) / len(scores)
    pos = sum(s > 0.2 for s in scores)
    neg = sum(s < -0.2 for s in scores)
    return avg, f"{pos} positive / {neg} negative of {len(scores)} headlines (VADER)"


def news_score(symbol: str, headlines: list[str]) -> tuple[float, str]:
    if not headlines:
        return 0.0, "no recent news"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return score_with_claude(symbol, headlines)
        except Exception as e:
            print(f"  [{symbol}] Claude scoring failed ({e}); falling back to VADER")
    return score_with_vader(headlines)


# ---------- 3. Trend ----------
def trend_score(close: pd.Series) -> tuple[float, str]:
    if len(close) < 60:
        return 0.0, "not enough price history"
    price = close.iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    vs_ma = (price / ma50 - 1)                       # e.g. +0.05 = 5% above average
    momentum = (price / close.iloc[-21] - 1)         # last ~20 trading days
    # squash to -1..+1: being 10% above the average/up 10% counts as fully bullish
    score = max(-1.0, min(1.0, (vs_ma + momentum) / 0.2))
    return score, f"price {vs_ma:+.1%} vs 50-day avg, {momentum:+.1%} last 20 days"


def get_close(symbol: str, period: str = "1y") -> pd.Series:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


# ---------- 4. Combine ----------
def decide(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "BUY"
    if score <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def analyze(symbol: str) -> dict | None:
    close = get_close(symbol)
    if close.empty:
        print(f"  [{symbol}] no price data, skipped")
        return None
    headlines = get_headlines(symbol)
    n_score, n_reason = news_score(symbol, headlines)
    t_score, t_reason = trend_score(close)
    combined = NEWS_WEIGHT * n_score + (1 - NEWS_WEIGHT) * t_score
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "price": round(float(close.iloc[-1]), 2),
        "signal": decide(combined),
        "score": round(combined, 3),
        "news_score": round(n_score, 3),
        "trend_score": round(t_score, 3),
        "news_reason": n_reason,
        "trend_reason": t_reason,
        "top_headline": headlines[0][:120] if headlines else "",
    }


def scan() -> None:
    engine = "Claude" if os.getenv("ANTHROPIC_API_KEY") else "VADER (set ANTHROPIC_API_KEY for smarter reading)"
    print(f"Scanning {len(WATCHLIST)} symbols | news reader: {engine}\n")
    results = []
    for sym in WATCHLIST:
        print(f"  reading {sym} ...")
        r = analyze(sym)
        if r:
            results.append(r)
    if not results:
        sys.exit("No results.")

    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n" + "=" * 78)
    print(f"{'SIGNAL':<6} {'SYMBOL':<9} {'PRICE':>10} {'SCORE':>7} {'NEWS':>6} {'TREND':>6}")
    print("=" * 78)
    for r in results:
        print(f"{r['signal']:<6} {r['symbol']:<9} {r['price']:>10,.2f} {r['score']:>+7.2f} "
              f"{r['news_score']:>+6.2f} {r['trend_score']:>+6.2f}")
        print(f"       news:  {r['news_reason']}")
        print(f"       trend: {r['trend_reason']}")
        if r["top_headline"]:
            print(f"       top headline: {r['top_headline']}")
        print("-" * 78)

    pd.DataFrame(results).to_csv(LOG_FILE, mode="a", index=False,
                                 header=not os.path.exists(LOG_FILE))
    message = format_phone_message(results)
    if paper_enabled():
        try:
            actions = follow_signals(results)
            s = paper_summary()
            message += ("\n\n💰 Fake-money account\n" + "\n".join(actions) +
                        f"\nBalance ${s['equity']:,.0f} ({s['profit_pct']:+.2f}% since start)")
            print("\n".join(actions))
        except Exception as e:
            print(f"Paper trading failed: {e}")
    send_telegram(message)
    print(f"\nSaved to {LOG_FILE}. Run `python news_trading_bot.py check` in a few days to grade these.")
    print("Reminder: mechanical signals, not financial advice.")


# ---------- Fake-money testing on Alpaca paper trading ----------
PAPER_DOLLARS_PER_TRADE = 1000   # fake dollars spent on each BUY


def paper_enabled() -> bool:
    return bool(os.getenv("ALPACA_KEY") and os.getenv("ALPACA_SECRET"))


def alpaca_client():
    from alpaca.trading.client import TradingClient
    # paper=True -> Alpaca's fake-money environment ONLY. Never real money.
    return TradingClient(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"), paper=True)


def _is_crypto(sym: str) -> bool:
    return sym.upper().endswith("-USD")


def _alpaca_symbol(sym: str) -> str:
    return sym.upper().replace("-USD", "/USD") if _is_crypto(sym) else sym.upper()


def follow_signals(results: list[dict], dollars: float = PAPER_DOLLARS_PER_TRADE) -> list[str]:
    """BUY signal + not holding -> buy $dollars. SELL signal + holding -> sell all. No short selling."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = alpaca_client()
    held = {p.symbol.replace("/", ""): p for p in client.get_all_positions()}
    actions = []
    for r in results:
        sym = _alpaca_symbol(r["symbol"])
        key = sym.replace("/", "")
        try:
            if r["signal"] == "BUY" and key not in held:
                client.submit_order(order_data=MarketOrderRequest(
                    symbol=sym, notional=dollars, side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC if _is_crypto(r["symbol"]) else TimeInForce.DAY))
                actions.append(f"Bought ${dollars:,.0f} of {r['symbol']} (fake money)")
            elif r["signal"] == "SELL" and key in held:
                client.close_position(key)
                actions.append(f"Sold all {r['symbol']} (fake money)")
        except Exception as e:
            actions.append(f"{r['symbol']}: order failed ({e})")
    return actions or ["No trades needed today."]


def paper_summary(start_balance: float = 100_000) -> dict:
    client = alpaca_client()
    acct = client.get_account()
    equity = float(acct.equity)
    positions = [{
        "Symbol": p.symbol,
        "Value": round(float(p.market_value), 2),
        "Bought at": round(float(p.avg_entry_price), 2),
        "Now": round(float(p.current_price), 2),
        "Profit $": round(float(p.unrealized_pl), 2),
        "Profit %": f"{float(p.unrealized_plpc) * 100:+.2f}%",
    } for p in client.get_all_positions()]
    return {"equity": equity, "cash": float(acct.cash), "start": start_balance,
            "profit": equity - start_balance, "profit_pct": (equity / start_balance - 1) * 100,
            "positions": positions}


# ---------- Phone notifications (Telegram) ----------
def format_phone_message(results: list[dict]) -> str:
    icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
    lines = [f"📈 Trading signals {results[0]['time']}", ""]
    for r in results:
        lines.append(f"{icons[r['signal']]} {r['signal']} {r['symbol']}  ${r['price']:,.2f}  (score {r['score']:+.2f})")
        if r["signal"] != "HOLD":
            lines.append(f"   📰 {r['news_reason']}")
            if r["top_headline"]:
                lines.append(f"   “{r['top_headline'][:100]}”")
    lines += ["", "Mechanical signals, not financial advice."]
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token, chat_id = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return  # Telegram not set up; just print to screen
    import requests
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             data={"chat_id": chat_id, "text": text[:4000]}, timeout=20)
        print("Sent to your phone via Telegram." if resp.ok else f"Telegram error: {resp.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")


# ---------- 5. Grade past signals ----------
def check() -> None:
    if not os.path.exists(LOG_FILE):
        sys.exit("No signals logged yet. Run a scan first.")
    log = pd.read_csv(LOG_FILE)
    log = log[log["signal"] != "HOLD"]
    if log.empty:
        sys.exit("No BUY/SELL signals logged yet.")

    rows = []
    for sym, group in log.groupby("symbol"):
        now = get_close(sym, period="5d")
        if now.empty:
            continue
        current = float(now.iloc[-1])
        for _, r in group.iterrows():
            move = current / r["price"] - 1
            correct = (move > 0) if r["signal"] == "BUY" else (move < 0)
            rows.append({"time": r["time"], "symbol": sym, "signal": r["signal"],
                         "then": r["price"], "now": round(current, 2),
                         "move": f"{move:+.2%}", "correct": correct})
    graded = pd.DataFrame(rows)
    print(graded.to_string(index=False))
    hit = graded["correct"].mean() * 100
    print(f"\nHit rate: {hit:.0f}% of {len(graded)} BUY/SELL signals went the right way.")
    print("Around 50% = no better than a coin flip. You want weeks of data and clearly above 50%.")
    send_telegram(f"📊 Signal report card\nHit rate: {hit:.0f}% of {len(graded)} BUY/SELL signals "
                  "went the right way.\n(~50% = coin flip)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        scan()
