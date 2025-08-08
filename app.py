import time
import ccxt
import pandas as pd
import streamlit as st

# ---------------------------
# Configurações padrão
# ---------------------------
DEFAULT_SYMBOLS = [
    "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "LINK/USDT", "MATIC/USDT", "TRX/USDT", "AVAX/USDT"
]

# ---------------------------
# Navegação via Sidebar
# ---------------------------
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione a página", ["Backtest", "Movimentos em Tempo Real"])

# ---------------------------
# Página de Backtest (placeholder)
# ---------------------------
if page == "Backtest":
    st.title("Backtest de Estratégia")
    st.write("Aqui ficaria o código do backtest de estratégia. Utilize a estrutura já existente ou adapte conforme necessário.")
    
# ---------------------------
# Página de Movimentos em Tempo Real
# ---------------------------
else:
    st.title("Movimentos em Tempo Real dos Altcoins da Binance")
    
    # Inicializa o flag de conexão no session_state
    if 'realtime_on' not in st.session_state:
        st.session_state.realtime_on = False

    # Botões para conectar e desconectar
    col1, col2 = st.columns(2)
    if col1.button("Conectar"):
        st.session_state.realtime_on = True
    if col2.button("Desconectar"):
        st.session_state.realtime_on = False

    # Se a conexão estiver ativa, atualiza a cada 2 segundos
    if st.session_state.realtime_on:
        st.write("Conectado. Atualizando dados em tempo real...")
        ex = ccxt.binance({
            "enableRateLimit": True,
        })

        try:
            dados = []
            # Para cada altcoin na lista de símbolos, busca informações do ticker
            for sym in DEFAULT_SYMBOLS:
                ticker = ex.fetch_ticker(sym)
                dados.append({
                    "Símbolo": sym,
                    "Preço": ticker["last"],
                    "Variação (%)": ticker.get("percentage", None)
                })
            df = pd.DataFrame(dados)
            st.table(df)
        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
        
        # Aguarda 2 segundos e reexecuta a página para atualização contínua
        time.sleep(2)
        st.experimental_rerun()
    else:
        st.write("Clique em 'Conectar' para iniciar a conexão com os dados em tempo real.")
