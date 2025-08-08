import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Função para buscar dados OHLCV
# -----------------------------
def fetch_ohlcv_binance(symbol: str, timeframe: str = "4h", since: int = None, limit: int = 500) -> pd.DataFrame:
    """
    Busca dados OHLCV da Binance para um símbolo e timeframe especificados.
    Se "since" não for informado, busca dados dos últimos 30 dias.
    """
    ex = ccxt.binance({
        'enableRateLimit': True,
    })
    if since is None:
        # Últimos 30 dias
        since = ex.milliseconds() - 30 * 24 * 60 * 60 * 1000

    all_data = []
    while True:
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"Erro ao buscar dados: {e}")
            break
        if not data:
            break
        all_data.extend(data)
        last_ts = data[-1][0]
        # Se os dados retornados forem menos do que o limite, encerra
        if len(data) < limit:
            break
        since = last_ts + 1
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.sort_index()

# -----------------------------
# Função para cálculo de indicadores técnicos
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Retorno percentual do candle
    df['ret'] = df['close'].pct_change()

    # Cálculo do RSI (14 períodos)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Cálculo do z-score do volume (janela de 24 candles)
    df['vol_mean'] = df['volume'].rolling(window=24, min_periods=24).mean()
    df['vol_std'] = df['volume'].rolling(window=24, min_periods=24).std()
    df['vol_z'] = (df['volume'] - df['vol_mean']) / (df['vol_std'].replace(0, np.nan))
    return df

# -----------------------------
# Função para detectar eventos extremos
# -----------------------------
def detect_extreme_events(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Detecta os candles cujo retorno seja maior ou igual ao threshold.
    Por padrão, identifica movimentos de ganhos iguais ou maiores que 80%.
    """
    df = df.copy()
    df['extreme_event'] = (df['ret'] >= threshold).astype(int)
    events = df[df['extreme_event'] == 1]
    return events

# -----------------------------
# Função para clusterizar os eventos extremos
# -----------------------------
def cluster_events(events: pd.DataFrame):
    """
    Utiliza KMeans para clusterizar os eventos com base em alguns indicadores.
    São utilizados os indicadores: retorno, RSI e volume z-score.
    """
    if events.empty:
        return None, None
    # Seleciona as colunas de interesse (garantindo que não existam NaNs)
    features = events[['ret', 'rsi_14', 'vol_z']].dropna()
    if features.empty:
        return None, None
    # Padroniza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # Definindo um número arbitrário de clusters (ex.: 2)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    features['cluster'] = clusters
    return features, kmeans

# -----------------------------
# Bloco Principal
# -----------------------------
if __name__ == "__main__":
    # Exemplo com um par de altcoin (pode trocar por "ETH/USDT", "SOL/USDT", etc.)
    symbol = "SOL/USDT"
    timeframe = "4h"
    # Define a data inicial (em milissegundos); por exemplo: 1º de janeiro de 2022
    since_str = "2022-01-01T00:00:00Z"
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)

    print(f"Buscando dados para {symbol}...")
    df = fetch_ohlcv_binance(symbol, timeframe, since_ms)
    if df.empty:
        print("Nenhum dado foi retornado. Verifique o símbolo ou a conexão.")
        exit()

    print("Calculando indicadores...")
    df = add_indicators(df)

    print("Detectando eventos extremos (movimentos >= 80%)...")
    extreme_events = detect_extreme_events(df, threshold=0.80)
    print(f"Eventos extremos encontrados: {len(extreme_events)}")
    if extreme_events.empty:
        print("Nenhum evento extremo detectado.")
    else:
        print(extreme_events[['close', 'ret', 'rsi_14', 'vol_z']])

        print("Clusterizando os eventos extremos...")
        clustered_events, kmeans_model = cluster_events(extreme_events)
        if clustered_events is not None:
            print("Resultados de Clusterização:")
            print(clustered_events)

            # Plot do resultado da clusterização: RSI vs Volume z-Score
            plt.figure(figsize=(8, 6))
            for clust in clustered_events['cluster'].unique():
                subset = clustered_events[clustered_events['cluster'] == clust]
                plt.scatter(subset['rsi_14'], subset['vol_z'], label=f"Cluster {clust}", s=100, alpha=0.7)
            plt.xlabel("RSI (14 períodos)")
            plt.ylabel("Volume Z-Score (24 períodos)")
            plt.title("Clusterização de Eventos Extremos")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Plot do preço com marcação dos eventos extremos
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label="Preço")
    if not extreme_events.empty:
        plt.scatter(extreme_events.index, extreme_events['close'], color="red", label="Eventos extremos", zorder=5)
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.title(f"{symbol} - Preço com Eventos Extremos")
    plt.legend()
    plt.grid(True)
    plt.show()
