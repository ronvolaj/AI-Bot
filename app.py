"""
Phone app for the news trading signal bot.

Run on your computer:   streamlit run app.py
Put it on your phone:   deploy free at share.streamlit.io (see setup steps), open
                        the link on your phone, then "Add to Home Screen".

Uses the same logic as news_trading_bot.py (keep both files together).
Mechanical signals, not financial advice.
"""

import os

import pandas as pd
import streamlit as st

import news_trading_bot as bot

st.set_page_config(page_title="AI Signal Robot", page_icon="📈", layout="centered")

# Secrets set in Streamlit Cloud become environment variables for the bot
for key in ("ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        pass

COLORS = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#6b7280"}
ICONS = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
.sig {border-radius: 14px; padding: 14px 16px; margin: 8px 0; color: white;}
.sig h3 {margin: 0; font-size: 1.25rem; color: white;}
.sig .row {display: flex; justify-content: space-between; align-items: baseline;}
.sig .price {font-size: 1.1rem; font-weight: 600;}
.sig small {opacity: .9;}
div.stButton > button {width: 100%; height: 3.2rem; font-size: 1.1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Signal Robot")
engine = "Claude AI" if os.getenv("ANTHROPIC_API_KEY") else "basic word scoring (add an Anthropic key for Claude)"
st.caption(f"Reads the news + price trend → BUY / SELL / HOLD.  News reader: {engine}")

# ---------- Settings ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ", ".join(bot.WATCHLIST)

with st.expander("⚙️ Watchlist & settings"):
    st.session_state.watchlist = st.text_area(
        "Symbols (comma separated). Crypto like BTC-USD, ETH-USD works too.",
        st.session_state.watchlist)
    bot.NEWS_WEIGHT = st.slider("How much the news counts vs the trend", 0.0, 1.0, bot.NEWS_WEIGHT, 0.1)
    bot.BUY_THRESHOLD = st.slider("BUY when score is above", 0.05, 0.8, bot.BUY_THRESHOLD, 0.05)
    bot.SELL_THRESHOLD = -st.slider("SELL when score is below minus", 0.05, 0.8, -bot.SELL_THRESHOLD, 0.05)

symbols = [s.strip().upper() for s in st.session_state.watchlist.split(",") if s.strip()]


# ---------- Scan ----------
@st.cache_data(ttl=900, show_spinner=False)   # re-use results for 15 min to save API costs
def run_scan(symbols: tuple, news_w: float, buy_t: float, sell_t: float) -> list[dict]:
    bot.NEWS_WEIGHT, bot.BUY_THRESHOLD, bot.SELL_THRESHOLD = news_w, buy_t, sell_t
    out = []
    for sym in symbols:
        close = bot.get_close(sym)
        if close.empty:
            continue
        headlines = bot.get_headlines(sym)
        n_score, n_reason = bot.news_score(sym, headlines)
        t_score, t_reason = bot.trend_score(close)
        score = news_w * n_score + (1 - news_w) * t_score
        out.append({
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": sym, "price": round(float(close.iloc[-1]), 2),
            "signal": bot.decide(score), "score": round(score, 3),
            "news_score": round(n_score, 3), "trend_score": round(t_score, 3),
            "news_reason": n_reason, "trend_reason": t_reason,
            "top_headline": headlines[0][:120] if headlines else "",
            "headlines": headlines, "chart": close.tail(120),
        })
    return sorted(out, key=lambda r: r["score"], reverse=True)


if st.button("🔍 Scan the news now", type="primary"):
    if not symbols:
        st.warning("Add at least one symbol in the watchlist.")
    else:
        with st.spinner(f"Reading news for {len(symbols)} symbols..."):
            results = run_scan(tuple(symbols), bot.NEWS_WEIGHT, bot.BUY_THRESHOLD, bot.SELL_THRESHOLD)
        st.session_state.results = results
        if results:
            log_rows = [{k: v for k, v in r.items() if k not in ("headlines", "chart")} for r in results]
            pd.DataFrame(log_rows).to_csv(bot.LOG_FILE, mode="a", index=False,
                                          header=not os.path.exists(bot.LOG_FILE))

results = st.session_state.get("results")

# ---------- Results ----------
tab_sig, tab_hist = st.tabs(["Signals", "Track record"])

with tab_sig:
    if not results:
        st.info("Tap **Scan the news now** to get today's signals.")
    else:
        buys = sum(r["signal"] == "BUY" for r in results)
        sells = sum(r["signal"] == "SELL" for r in results)
        c1, c2, c3 = st.columns(3)
        c1.metric("BUY", buys)
        c2.metric("SELL", sells)
        c3.metric("HOLD", len(results) - buys - sells)

        for r in results:
            st.markdown(f"""
<div class="sig" style="background:{COLORS[r['signal']]}">
  <div class="row"><h3>{ICONS[r['signal']]} {r['signal']} · {r['symbol']}</h3>
  <span class="price">${r['price']:,.2f}</span></div>
  <small>Score {r['score']:+.2f} &nbsp;|&nbsp; News {r['news_score']:+.2f} &nbsp;|&nbsp; Trend {r['trend_score']:+.2f}</small>
</div>""", unsafe_allow_html=True)
            with st.expander(f"Why {r['symbol']}?"):
                st.write(f"**News:** {r['news_reason']}")
                st.write(f"**Trend:** {r['trend_reason']}")
                st.line_chart(r["chart"], height=180)
                if r["headlines"]:
                    st.write("**Latest headlines read:**")
                    for h in r["headlines"]:
                        st.write(f"• {h[:200]}")
        st.caption(f"Scanned {results[0]['time']}. Mechanical signals from a formula, not financial advice.")

with tab_hist:
    st.write("How often past BUY/SELL signals went the right way. Around 50% = coin flip.")
    if st.button("📊 Grade past signals"):
        if not os.path.exists(bot.LOG_FILE):
            st.info("No signals saved yet. Scan first, then come back in a few days.")
        else:
            log = pd.read_csv(bot.LOG_FILE)
            log = log[log["signal"] != "HOLD"]
            rows = []
            for sym, group in log.groupby("symbol"):
                now = bot.get_close(sym, period="5d")
                if now.empty:
                    continue
                cur = float(now.iloc[-1])
                for _, r in group.iterrows():
                    move = cur / r["price"] - 1
                    ok = move > 0 if r["signal"] == "BUY" else move < 0
                    rows.append({"When": r["time"], "Symbol": sym, "Signal": r["signal"],
                                 "Then": r["price"], "Now": round(cur, 2),
                                 "Move": f"{move:+.2%}", "Right?": "✅" if ok else "❌"})
            if rows:
                graded = pd.DataFrame(rows)
                hit = (graded["Right?"] == "✅").mean() * 100
                st.metric("Hit rate", f"{hit:.0f}%", help=f"{len(graded)} signals graded")
                st.dataframe(graded, hide_index=True, width="stretch")
            else:
                st.info("No BUY/SELL signals to grade yet.")
    st.caption("Note: on the free cloud, saved history can reset when the app restarts.")
