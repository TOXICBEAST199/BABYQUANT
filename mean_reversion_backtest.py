import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# 1. Data download function (Dynamic History)
# -----------------------------
@st.cache_data(ttl=3600)
def download_data(ticker: str, start_date, end_date: date) -> pd.DataFrame:
    # If "Start from Inception" is selected, start_date is None.
    # We use a very old date (1900-01-01); yfinance automatically clips to the actual first trading day.
    
    if start_date is None: 
        start_buffer = "1900-01-01"
    else:
        # Standard buffer for indicators (MA200 requires 200 days prior to simulation start)
        buffer_days = 365
        start_buffer = start_date - timedelta(days=buffer_days)
    
    try:
        df = yf.download(ticker, start=start_buffer, end=end_date, auto_adjust=False, progress=False)
    except Exception as e:
        raise ValueError(f"Error downloading {ticker}: {e}")
    
    if df.empty:
        raise ValueError(f"No data downloaded. Check ticker symbol.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

# -----------------------------
# 2. Core Backtest Engine
# -----------------------------
def run_backtest(
    data: pd.DataFrame,
    strategy_type: str,
    simulation_start_date: date,
    lookback: int,
    z_entry: float,
    z_exit: float,
    stop_loss_pct: float,
    max_hold_days: int,
    initial_capital: float,
    position_size_pct: float,
    commission_pct: float,
    use_long_only: bool,
    use_trend_filter: bool,
):
    df = data.copy()
    close = df["Close"]
    open_ = df["Open"]
    dates = df.index.to_list()

    cash = float(initial_capital)
    position_shares = 0
    entry_price = None
    holding_days = 0

    equity_curve = []
    equity_dates = []
    trades = []

    try:
        sim_start_ts = pd.Timestamp(simulation_start_date)
        start_idx = df.index.searchsorted(sim_start_ts) if sim_start_ts not in df.index else df.index.get_loc(sim_start_ts)
    except:
        start_idx = 0
    
    if start_idx >= len(df):
        return None

    # ---------------- MAIN LOOP ----------------
    for i in range(start_idx, len(df)):
        today_idx = i
        yesterday_idx = i - 1
        today = dates[today_idx]
        
        price_signal = float(close.iloc[yesterday_idx]) 
        open_today = float(open_.iloc[today_idx])       
        close_today = float(close.iloc[today_idx])      

        # Indicators
        window = close.iloc[yesterday_idx - lookback + 1 : yesterday_idx + 1]
        
        if len(window) < lookback:
            equity_curve.append(cash + (position_shares * close_today))
            equity_dates.append(today)
            continue
            
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        z = float((price_signal - mean) / std) if std > 0.0 else 0.0

        if yesterday_idx >= 200:
            sma_200 = float(close.iloc[yesterday_idx - 199 : yesterday_idx + 1].mean())
        else:
            sma_200 = 0.0

        is_uptrend = price_signal > sma_200
        is_downtrend = price_signal < sma_200

        # Exit Logic
        exit_flag = False
        exit_reason = None

        if position_shares != 0:
            holding_days += 1
            if std > 0.0:
                if strategy_type == "Mean Reversion":
                    if position_shares > 0 and z > -z_exit:
                        exit_flag, exit_reason = True, "Target (Reversion)"
                    elif position_shares < 0 and z < z_exit:
                        exit_flag, exit_reason = True, "Target (Reversion)"
                elif strategy_type == "Momentum":
                    if position_shares > 0 and z < z_exit:
                        exit_flag, exit_reason = True, "Trend Faded"
                    elif position_shares < 0 and z > -z_exit:
                        exit_flag, exit_reason = True, "Trend Faded"

            if entry_price is not None:
                pnl_pct = (price_signal - entry_price) / entry_price if position_shares > 0 else (entry_price - price_signal) / entry_price
                if pnl_pct <= -stop_loss_pct:
                    exit_flag, exit_reason = True, "Stop Loss"

            if holding_days >= max_hold_days:
                exit_flag, exit_reason = True, "Time Exit"

        # Execute Exit
        if exit_flag and position_shares != 0:
            trade_shares = -position_shares
            cost = abs(trade_shares * open_today)
            comm = commission_pct * cost
            cash -= (trade_shares * open_today + comm)
            
            trades.append({
                "date": today, "type": "EXIT", "shares": trade_shares, 
                "price": open_today, "reason": exit_reason, "equity_post": cash 
            })
            position_shares = 0
            entry_price = None
            holding_days = 0

        # Entry Logic
        if position_shares == 0 and std > 0.0:
            entry_flag = False
            direction = 0
            
            if strategy_type == "Mean Reversion":
                long_condition = z < -z_entry
                short_condition = z > z_entry
            elif strategy_type == "Momentum":
                long_condition = z > z_entry
                short_condition = z < -z_entry

            if use_trend_filter:
                if long_condition: long_condition = is_uptrend
                if short_condition: short_condition = is_downtrend
            
            if use_long_only: short_condition = False

            if long_condition: entry_flag, direction = True, 1
            elif short_condition: entry_flag, direction = True, -1

            if entry_flag:
                cap_use = min(cash * position_size_pct, cash)
                shares = math.floor(cap_use / open_today) if open_today > 0 else 0
                
                if shares > 0:
                    trade_shares = direction * shares
                    cost = abs(trade_shares * open_today)
                    comm = commission_pct * cost
                    cash -= (trade_shares * open_today + comm)
                    position_shares = trade_shares
                    entry_price = open_today
                    holding_days = 0
                    
                    trades.append({
                        "date": today, "type": "ENTRY", "shares": trade_shares, 
                        "price": open_today, "reason": f"{strategy_type} Entry", 
                        "equity_post": cash + (position_shares * open_today)
                    })

        equity_curve.append(cash + (position_shares * close_today))
        equity_dates.append(today)

    # ---------------- STATS COMPILATION ----------------
    if not equity_curve: return None
    
    equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_dates))
    total_return = (equity_series.iloc[-1] / initial_capital) - 1.0
    days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = ((equity_series.iloc[-1] / initial_capital) ** (365.25/days) - 1) if days > 0 and equity_series.iloc[-1] > 0 else 0
    
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    drawdown = (equity_series / equity_series.cummax()) - 1.0
    
    trades_df = pd.DataFrame(trades)
    win_rate, expectancy = 0.0, 0.0
    win_count, loss_count, total_trades_count = 0, 0, 0
    avg_pnl_per_trade_pct = 0.0
    avg_pnl_per_trade_amt = 0.0
    
    if not trades_df.empty:
        entries = trades_df[trades_df["type"] == "ENTRY"].reset_index(drop=True)
        exits = trades_df[trades_df["type"] == "EXIT"].reset_index(drop=True)
        
        pnls_pct = []
        
        # Calculate raw pnls
        for i in range(min(len(entries), len(exits))):
            en_price = entries.loc[i, "price"]
            ex_price = exits.loc[i, "price"]
            direction = 1 if entries.loc[i, "shares"] > 0 else -1
            pnl = (ex_price - en_price) / en_price * direction
            pnls_pct.append(pnl)
        
        if pnls_pct:
            wins = [p for p in pnls_pct if p > 0]
            losses = [p for p in pnls_pct if p <= 0]
            win_count = len(wins)
            loss_count = len(losses)
            total_trades_count = len(pnls_pct)
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            win_rate = len(wins) / len(pnls_pct)
            
            # Expectancy (Average Return per Trade in %)
            expectancy = (win_rate * avg_win) - ((1-win_rate) * avg_loss)
            avg_pnl_per_trade_pct = np.mean(pnls_pct)
            
            # Average PnL Amount (Currency)
            total_net_profit = equity_series.iloc[-1] - initial_capital
            avg_pnl_per_trade_amt = total_net_profit / total_trades_count

    return {
        "equity_curve": equity_series,
        "drawdown": drawdown,
        "trades": trades_df,
        "stats": {
            "final_equity": equity_series.iloc[-1], "total_return": total_return, "cagr": cagr,
            "max_drawdown": drawdown.min(), "sharpe": sharpe, "win_rate": win_rate, 
            "expectancy": expectancy, "win_count": win_count, "loss_count": loss_count,
            "total_trades": total_trades_count, "avg_pnl_pct": avg_pnl_per_trade_pct,
            "avg_pnl_amt": avg_pnl_per_trade_amt
        }
    }

# -----------------------------
# 3. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Pro Backtester", layout="wide", page_icon="📈")
    st.title("⚡ Baby Quant : Algo-Trading Backtester")

    # --- SIDEBAR ---
    st.sidebar.header("1. Market Data")
    ticker = st.sidebar.text_input("Ticker", value="^NSEBANK")

    # --- TIMELINE SECTION ---
    st.sidebar.subheader("Timeline")
    
    # Checkbox to use Max History (Inception)
    use_max_history = st.sidebar.checkbox(
        "Start from Inception (Max History)", 
        value=False,
        help="If checked, data will be downloaded from the very first day the stock was listed."
    )
    
    if use_max_history:
        # We pass None to signal 'Earliest Available' to the download function
        selected_start_date = None 
        st.sidebar.caption("📅 Start: Earliest Available Data")
    else:
        # Standard manual start date
        selected_start_date = st.sidebar.date_input(
            "Start Date", 
            value=date(2020, 1, 1)
        )

    # Dynamic End Date (Default to Yesterday)
    yesterday = date.today() - timedelta(days=1)
    end_date = st.sidebar.date_input(
        "End Date", 
        value=yesterday
    )

    st.sidebar.divider()
    st.sidebar.header("2. Strategy Logic")
    
    strategy_type = st.sidebar.radio(
        "Select Strategy Type", 
        ["Mean Reversion", "Momentum"],
        help="Mean Reversion: Buy when price is LOW (Oversold). Momentum: Buy when price is HIGH (Breakout)."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Filters:")
    
    use_long_only = st.sidebar.checkbox(
        "Long Only", 
        value=True, 
        help="Do you want to restrict the strategy to Long positions only?"
    )
    
    use_trend_filter = st.sidebar.checkbox(
        "Trend Filter (200 SMA)", 
        value=True, 
        help="Do you want to only trade when the price is above the 200-day Moving Average?"
    )

    lookback = st.sidebar.slider("Lookback (Days)", 10, 100, 20)
    z_entry = st.sidebar.slider("Entry Z-Score (Signal)", 1.0, 4.0, 2.0)
    z_exit = st.sidebar.slider("Exit Z-Score (Target)", -2.0, 2.0, 0.5)

    st.sidebar.divider()
    st.sidebar.header("3. Risk")
    capital = st.sidebar.number_input("Capital", value=500000.0)
    pos_pct = st.sidebar.slider("Size (%)", 10, 100, 50) / 100.0
    sl_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 3.0) / 100.0
    hold_days = st.sidebar.slider("Max Hold Days", 1, 50, 10)
    comm = st.sidebar.number_input("Commission (%)", value=0.05) / 100.0

    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running Simulation..."):
            try:
                # 1. Download Data
                df = download_data(ticker, selected_start_date, end_date)
                
                # 2. Determine Simulation Start Date
                # If using Max History, set simulation start to the first actual data point
                if selected_start_date is None:
                    sim_start = df.index[0].date()
                else:
                    sim_start = selected_start_date

                # 3. Run Strategy
                res = run_backtest(
                    df, strategy_type, sim_start, lookback, z_entry, z_exit, 
                    sl_pct, hold_days, capital, pos_pct, comm, use_long_only, use_trend_filter
                )
                
                if res:
                    stats = res["stats"]
                    
                    # 1. TOP METRICS
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Equity", f"₹ {stats['final_equity']:,.0f}", f"{stats['total_return']*100:.1f}%")
                    c2.metric("Total Trades", f"{stats['total_trades']}")
                    c3.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
                    c4.metric("Avg PnL % (Expectancy)", f"{stats['expectancy']*100:.2f}%")
                    c5.metric("Avg PnL (Amt)", f"₹ {stats['avg_pnl_amt']:,.0f}")
                    
                    st.divider()

                    # 2. GRAPHS
                    st.subheader("Strategy Performance")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=res["equity_curve"].index, 
                        y=res["equity_curve"], 
                        mode='lines', 
                        name='Equity',
                        line=dict(color='#00FF00', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.1)'
                    ))
                    fig_equity.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_equity, use_container_width=True)

                    st.subheader(f"Price Analysis: {ticker}")
                    
                    # Ensure plot starts from simulation start to avoid clutter from buffer data
                    plot_start_ts = pd.Timestamp(sim_start)
                    plot_df = df[df.index >= plot_start_ts]
                    plot_df['MA200_Plot'] = plot_df['Close'].rolling(200).mean()

                    fig_price = go.Figure()
                    fig_price.add_trace(go.Candlestick(
                        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                        low=plot_df['Low'], close=plot_df['Close'], name='Price'
                    ))
                    if use_trend_filter:
                        fig_price.add_trace(go.Scatter(
                            x=plot_df.index, y=plot_df['MA200_Plot'], 
                            opacity=0.7, line=dict(color='orange', width=2), name='MA 200'
                        ))
                    fig_price.update_layout(
                        xaxis_rangeslider_visible=False, height=400, template="plotly_dark",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

                    # 3. OUTCOME DISTRIBUTION
                    st.markdown("---")
                    st.header("Outcome Distribution & Trades")
                    
                    col_pie, col_log = st.columns([1, 2])
                    
                    with col_pie:
                        st.subheader("Wins vs Losses")
                        if stats['win_count'] + stats['loss_count'] > 0:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=['Wins', 'Losses'],
                                values=[stats['win_count'], stats['loss_count']],
                                hole=.4,
                                marker=dict(colors=['#00FF00', '#FF0000']),
                                textinfo='label+percent+value'
                            )])
                            fig_pie.update_layout(
                                template="plotly_dark", 
                                height=300, 
                                margin=dict(l=20, r=20, t=0, b=20),
                                showlegend=False
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.warning("No completed trades.")

                        # Exit Reasons Pie Chart
                        st.subheader("Exit Reasons")
                        trades_df_exits = res["trades"][res["trades"]["type"] == "EXIT"]
                        if not trades_df_exits.empty:
                            reason_counts = trades_df_exits["reason"].value_counts()
                            fig_reason = go.Figure(data=[go.Pie(
                                labels=reason_counts.index,
                                values=reason_counts.values,
                                hole=.4,
                                textinfo='label+percent'
                            )])
                            fig_reason.update_layout(
                                template="plotly_dark",
                                height=300,
                                margin=dict(l=20, r=20, t=0, b=20),
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                            )
                            st.plotly_chart(fig_reason, use_container_width=True)
                            
                    with col_log:
                        st.subheader("Trade Log")
                        if not res["trades"].empty:
                            trade_log_display = res["trades"].copy()
                            trade_log_display['date'] = trade_log_display['date'].dt.strftime('%Y-%m-%d')
                            trade_log_display['price'] = trade_log_display['price'].apply(lambda x: f"{x:.2f}")
                            trade_log_display['equity_post'] = trade_log_display['equity_post'].apply(lambda x: f"{x:,.0f}")
                            st.dataframe(trade_log_display, use_container_width=True, height=700) 
                        else:
                            st.info("No trades executed.")

                    # 4. FINAL VERDICT
                    st.markdown("---")
                    st.header("🧠 Final Verdict")
                    
                    profit_color = "green" if stats['total_return'] > 0 else "red"
                    
                    st.markdown(f"""
                    **Financial Outcome**
                    
                    Over this period, the model identified **{stats['total_trades']}** high-probability setups.
                    
                    * **Starting Capital:** {capital:,.0f}
                    * **Ending Capital:** {stats['final_equity']:,.0f}
                    * **Net Profit:** {stats['final_equity'] - capital:,.0f}
                    
                    **Bottom Line:** The strategy generated a net return on investment (ROI) of :{profit_color}[**{stats['total_return']*100:.1f}%**].
                    """)

                else:
                    st.error("No trades generated or data error.")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()