import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import ccxt
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

# ======================
# Configurações padrão
# ======================
DEFAULT_TIMEFRAME = "4h"   # conforme solicitado
DEFAULT_SINCE = "2021-01-01T00:00:00Z"
DEFAULT_SYMBOLS = [
    "ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "DOGE/USDT","LINK/USDT","MATIC/USDT","TRX/USDT","AVAX/USDT"
]
BENCH_SYMBOL = "BTC/USDT"
DEFAULT_PRED_THRESHOLD = 0.10
DEFAULT_MIN_TRAIN_MONTHS = 12

# ======================
# Utilidades
# ======================
def ts_to_ms(ts_str: str) -> int:
    return int(pd.Timestamp(ts_str).timestamp() * 1000)

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ohlcv_binance(symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    all_rows = []
    fetch_since = since_ms
    while True:
        try:
            ohlcvs = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        except Exception:
            time.sleep(2)
            continue
        if not ohlcvs:
            break
        all_rows.extend(ohlcvs)
        last_ts = ohlcvs[-1][0]
        if fetch_since is not None and last_ts <= fetch_since:
            break
        fetch_since = last_ts + 1
        time.sleep(0.2)
        if len(ohlcvs) < limit:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    return df.sort_index()

def make_features(df: pd.DataFrame, btc_ret: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["fut_ret_1"] = out["ret_1"].shift(-1)  # alvo: retorno no próximo candle (4h)

    # Momentos
    for n in [3, 6, 12]:
        out[f"mom_{n}"] = out["close"].pct_change(n)

    # Volatilidade (desvio padrão de retornos)
    out["vol_12"] = out["ret_1"].rolling(12).std()
    out["vol_24"] = out["ret_1"].rolling(24).std()

    # RSI
    out["rsi_14"] = compute_rsi(out["close"], 14)

    # Volume z-score
    vol_mean_24 = out["volume"].rolling(24).mean()
    vol_std_24 = out["volume"].rolling(24).std()
    out["vol_z_24"] = (out["volume"] - vol_mean_24) / (vol_std_24.replace(0, np.nan))

    # Correlação com BTC (24 janelas)
    out = out.join(btc_ret.rename("btc_ret"), how="left")
    out["corr_btc_24"] = out["ret_1"].rolling(24).corr(out["btc_ret"])

    # Remove NaNs críticos e a última linha por causa do shift do alvo
    out = out.dropna(subset=["fut_ret_1", "mom_3","mom_6","mom_12","vol_12","vol_24","rsi_14","vol_z_24","corr_btc_24"])
    return out

def time_month_splits(index: pd.DatetimeIndex, min_train_months: int):
    months = pd.PeriodIndex(index, freq="M")
    uniq_months = pd.Index(months.unique().sort_values())
    for i in range(min_train_months, len(uniq_months)):
        train_end_month = uniq_months[i-1]
        test_month = uniq_months[i]
        train_mask = months <= train_end_month
        test_mask = months == test_month
        yield train_mask, test_mask, str(test_month)

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

@dataclass
class BacktestResult:
    symbol: str
    total_trades: int
    strategy_return: float
    buyhold_return: float
    mdd_strategy: float
    mdd_buyhold: float

def run_backtest(symbols, timeframe, since_str, pred_threshold, min_train_months):
    since_ms = ts_to_ms(since_str)

    st.info("Baixando BTC/USDT para benchmark...")
    btc_df = fetch_ohlcv_binance(BENCH_SYMBOL, timeframe, since_ms)
    if btc_df.empty:
        st.error("Falha ao baixar BTC/USDT.")
        return None

    btc_df = btc_df[["close"]].rename(columns={"close":"btc_close"})
    btc_df["btc_ret"] = btc_df["btc_close"].pct_change(1)

    results = []
    all_equity = []
    all_bh_equity = []

    progress = st.progress(0.0, text="Processando ativos...")
    for i, symbol in enumerate(symbols, start=1):
        st.write(f"Processando {symbol}...")
        df = fetch_ohlcv_binance(symbol, timeframe, since_ms)
        if df.empty or len(df) < 300:
            st.warning(f"Dados insuficientes para {symbol}. Pulando.")
            progress.progress(i/len(symbols), text=f"Processando ativos... {i}/{len(symbols)}")
            continue

        merged = df.join(btc_df[["btc_ret"]], how="left")
        features_df = make_features(merged, merged["btc_ret"])

        feature_cols = ["ret_1","mom_3","mom_6","mom_12","vol_12","vol_24","rsi_14","vol_z_24","corr_btc_24"]
        X = features_df[feature_cols].copy()
        y = features_df["fut_ret_1"].copy()
        idx = features_df.index

        model = GradientBoostingRegressor(random_state=42)

        symbol_equity = pd.Series(index=idx, dtype=float)
        buyhold_equity = pd.Series(index=idx, dtype=float)
        total_trades = 0

        for train_mask, test_mask, test_month in time_month_splits(idx, min_train_months):
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            if len(X_train) < 200 or len(X_test) == 0:
                continue

            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)

            signal = (y_pred >= pred_threshold).astype(int)
            period_ret = signal * y_test  # custo = 0 por solicitação
            eq = (1 + period_ret).cumprod()
            symbol_equity.loc[eq.index] = eq

            bh = (1 + y_test).cumprod()
            buyhold_equity.loc[bh.index] = bh

            trades = int(signal.sum())
            total_trades += trades

        symbol_equity = symbol_equity.ffill().fillna(1.0)
        buyhold_equity = buyhold_equity.ffill().fillna(1.0)

        strat_ret = float(symbol_equity.iloc[-1] - 1.0) if len(symbol_equity) else 0.0
        bh_ret = float(buyhold_equity.iloc[-1] - 1.0) if len(buyhold_equity) else 0.0
        mdd_s = max_drawdown(symbol_equity)
        mdd_bh = max_drawdown(buyhold_equity)

        results.append(BacktestResult(
            symbol=symbol,
            total_trades=total_trades,
            strategy_return=strat_ret,
            buyhold_return=bh_ret,
            mdd_strategy=mdd_s,
            mdd_buyhold=mdd_bh,
        ))
        all_equity.append(symbol_equity.rename(symbol))
        all_bh_equity.append(buyhold_equity.rename(symbol))

        progress.progress(i/len(symbols), text=f"Processando ativos... {i}/{len(symbols)}")

    progress.empty()

    out = {}
    if results:
        res_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("symbol")
        out["results_df"] = res_df

    if all_equity:
        eq_df = pd.concat(all_equity, axis=1).sort_index().dropna(how="all")
        eq_df = eq_df.fillna(method="ffill").fillna(1.0)
        out["equity_df"] = eq_df
        eq_mean = eq_df.mean(axis=1)
        out["equity_mean"] = eq_mean

    if all_bh_equity:
        bh_df = pd.concat(all_bh_equity, axis=1).sort_index().dropna(how="all")
        bh_df = bh_df.fillna(method="ffill").fillna(1.0)
        out["bh_equity_df"] = bh_df
        out["bh_equity_mean"] = bh_df.mean(axis=1)

    return out

# ======================
# UI Streamlit
# ======================
st.set_page_config(page_title="Previsão e Backtest (4h) - Altcoins Binance", layout="wide")

st.title("Previsão de movimentos de altcoins (4h) - Binance")
st.caption("Pesquisa educacional. Não é recomendação de investimento. Custo de transação assumido como 0.")

with st.sidebar:
    st.header("Parâmetros")
    symbols = st.multiselect(
        "Pares USDT",
        DEFAULT_SYMBOLS,
        default=DEFAULT_SYMBOLS
    )
    timeframe = st.selectbox("Timeframe", ["4h"], index=0)
    since_str = st.text_input("Desde (UTC, ISO 8601)", value=DEFAULT_SINCE, help="Ex.: 2021-01-01T00:00:00Z")
    pred_threshold = st.number_input("Limiar de previsão para abrir compra", min_value=0.0, max_value=1.0, step=0.01, value=DEFAULT_PRED_THRESHOLD, help="Ex.: 0.10 = 10% em 4h")
    min_train_months = st.number_input("Meses mínimos de treino", min_value=3, max_value=36, step=1, value=DEFAULT_MIN_TRAIN_MONTHS)
    run_btn = st.button("Rodar backtest", type="primary")

if run_btn:
    if not symbols:
        st.warning("Selecione ao menos um par.")
    else:
        with st.spinner("Executando backtest..."):
            results = run_backtest(symbols, timeframe, since_str, pred_threshold, min_train_months)

        if not results:
            st.error("Sem resultados. Verifique parâmetros ou dados.")
        else:
            res_df = results.get("results_df")
            if res_df is not None:
                st.subheader("Resumo por ativo")
                st.dataframe(res_df, use_container_width=True)

                csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button("Baixar resumo (CSV)", data=csv, file_name="resumo_backtest.csv", mime="text/csv")

            eq_mean = results.get("equity_mean")
            bh_mean = results.get("bh_equity_mean")
            eq_df = results.get("equity_df")

            if eq_mean is not None:
                st.subheader("Equity médio (estratégia vs. buy&hold)")
                comp_df = pd.DataFrame({
                    "Estratégia (média)": eq_mean,
                    "Buy&Hold (média)": bh_mean if bh_mean is not None else None
                }).dropna(how="all")
                st.line_chart(comp_df)

            if eq_df is not None:
                st.subheader("Equity por ativo (estratégia)")
                st.line_chart(eq_df)

else:
    st.info("Defina os parâmetros na barra lateral e clique em Rodar backtest.")

st.markdown(
    "Observações: 1) Limiar de 10% em 4h tende a gerar poucos trades. 2) Este app não considera custos, slippage ou funding. 3) Resultados passados não garantem desempenho futuro."
)
