import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff

# Carregar dados
df = pd.read_csv("./dataset/dataset_clean.csv")

# Identificar a coluna correta
coluna_valor = [col for col in df.columns if "valor_mercado" in col][0]

# Converter os valores para float
df['valor_mercado'] = df[coluna_valor].str.replace('R$', '', regex=False)\
                                    .str.replace('.', '', regex=False)\
                                    .str.replace(',', '.', regex=False)\
                                    .astype(float)

# Slider para faixa total
min_valor = int(df['valor_mercado'].min())
max_valor = int(df['valor_mercado'].max())
step = 5_000_000

# SIDEBAR
# Using object notation
# Sidebar para seleção de posição
position_labels = {'G': 'Goleiro', 'D': 'Zagueiro', 'M': 'Meia', 'F': 'Atacante'}
#selected_position = st.sidebar.selectbox("Selecione a posição:", list(position_labels.keys()), format_func=lambda x: position_labels[x])
# Sidebar para escolha da posição
selected_position = st.sidebar.selectbox(
    "Selecione a posição:",
    list(position_labels.keys()),
    format_func=lambda x: position_labels[x],
    key="position_selectbox"
)

#values = st.sidebar.slider("Selecione a faixa de preço dos jogadores", 0.0, 100.0, (25.0, 75.0))
intervalo = st.sidebar.slider("Escolha o intervalo de valor", min_valor, max_valor, (min_valor, max_valor), step=step, format="R$ %d")

st.markdown("""
## Função para criar o time

Esta seção utiliza um modelo de aprendizado de máquina para sugerir a formação ideal de um time com base nos atributos dos jogadores e no valor de mercado. A ideia é montar uma equipe equilibrada, maximizando o desempenho dentro de um orçamento definido.
""")

# parte introdutoria
st.title('Dashboard - Statistics Players')
st.markdown("""
## 📊 Dashboard Interativo – Análise de Dados

Este dashboard foi desenvolvido com **Streamlit** como parte do Trabalho 1 da disciplina **Ciência de Dados (DCA3501)**. Ele apresenta, de forma interativa e visual, os principais resultados da análise exploratória realizada no notebook original, facilitando a interpretação dos dados e das métricas estatísticas geradas durante o estudo.
""")

# lendo o dataframe do valor de mercado
#df_prices = pd.read_csv(r"C:\Users\carlos.medeiros\carlos\uf\data_science_dca3501\uni2\dataset\market_value.csv")
#print(df_prices.shape)

# lendo o dataframe das estatisticas do jogador
#df = pd.read_csv(r"C:\Users\carlos.medeiros\carlos\uf\data_science_dca3501\uni2\dataset\statistics_player.csv")
#print(df.shape)

# Primeira linha: gráfico 1 e gráfico 2
#col1, col2 = st.columns(2)

# Filtrar o DataFrame com base na posição e no intervalo de valores
df_filtrado = df[
    (df['valor_mercado'] >= intervalo[0]) &
    (df['valor_mercado'] <= intervalo[1]) &
    (df['position'] == selected_position)  # <--- ajuste aqui conforme o nome exato da coluna
]

# GRAFICO 1
st.subheader("Gráfico 1")
st.caption("Mostrar os jogadores na faixa de preço (barra)")

# Gerar bins
bins = list(range(intervalo[0], intervalo[1] + step, step))
# Gerar bins
bins = list(range(intervalo[0], intervalo[1] + step, step))
df_filtrado['faixa_preco'] = pd.cut(df_filtrado['valor_mercado'], bins=bins)

# Contar jogadores por faixa
faixa_counts = df_filtrado['faixa_preco'].value_counts().sort_index()

# Criar rótulos legíveis
labels = [f"{int(b.left / 1e6)}M–{int(b.right / 1e6)}M" for b in faixa_counts.index]
faixa_counts.index = labels

# Exibir gráfico
st.bar_chart(faixa_counts)






# GRAFICO 2
st.subheader("Gráfico 2")
st.caption("Atributos físicos")

# Labels específicas por posição
position_metrics = {
    'G': ["rating", "saves", "savedShotsFromInsideTheBox", "goodHighClaim", "height", "goalsPrevented"],
    'D': ["duelWon", "totalClearance", "blockedScoringAttempt", "interceptionWon", "height", "accuratePass"],
    'M': ["accuratePass", "rating", "keyPass", "expectedGoals", "expectedAssists", "goalAssist"],
    'F': ["goals", "expectedGoals", "onTargetScoringAttempt", "bigChanceCreated", "expectedAssists", "goalAssist"]
}


# Colunas a serem usadas no gráfico
selected_labels = position_metrics[selected_position]

# Normalizar globalmente os atributos selecionados
df_normalized = df.copy()
for col in selected_labels:
    if col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        df_normalized[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

# Filtrar os 5 melhores jogadores da posição
top_players = df_normalized[df['position'] == selected_position].sort_values(by='rating', ascending=False).head(5)

# Ângulos para o gráfico
angles = np.linspace(0, 2 * np.pi, len(selected_labels), endpoint=False).tolist()
angles += angles[:1]

# Título e descrição
st.subheader(f"Radar - Top 5 {position_labels[selected_position]}s")
st.caption("Atributos normalizados com base em todo o conjunto de dados")

# Criação do gráfico
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors = plt.cm.viridis(np.linspace(0, 1, len(top_players)))

for idx, (i, player) in enumerate(top_players.iterrows()):
    values = player[selected_labels].values.tolist()
    values += values[:1]
    nome = player['name'] if 'name' in player else f"Jogador {idx+1}"
    ax.plot(angles, values, label=nome, color=colors[idx], linewidth=2)
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# Configurações visuais
ax.set_xticks(angles[:-1])
ax.set_xticklabels(selected_labels, fontsize=9)
ax.set_title(f"Top 5 jogadores - {position_labels[selected_position]}")
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Mostrar gráfico
st.pyplot(fig)







#with col1:
#with col2:
    
# Segunda linha: gráfico 3 e modelo de ML
col3, col4 = st.columns(2)

with col3:
    st.subheader("Gráfico 3")
    st.caption("Correlação das variáveis com o preço")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    st.scatter_chart(chart_data)    

with col4:
    st.subheader("Modelo de ML")
    st.caption("Predição de preços com base nos atributos")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    st.area_chart(chart_data)


