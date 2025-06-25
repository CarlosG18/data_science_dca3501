import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

# Carregar dados
df = pd.read_csv("./dataset/melhores_jogadores.csv")

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

# Dicionário de métricas por posição
position_metrics = {
    'G': ["rating", "saves", "savedShotsFromInsideTheBox", "goodHighClaim", "height", "goalsPrevented"],
    'D': ["duelWon", "totalClearance", "blockedScoringAttempt", "interceptionWon", "height", "accuratePass"],
    'M': ["accuratePass", "rating", "keyPass", "expectedGoals", "expectedAssists", "goalAssist"],
    'F': ["goals", "expectedGoals", "onTargetScoringAttempt", "bigChanceCreated", "expectedAssists", "goalAssist"]
}

# Exemplo de labels de posição (mapeamento tipo 'G' → 'Goleiro')
position_labels = {
    'G': 'Goleiro',
    'D': 'Defensor',
    'M': 'Meio-campo',
    'F': 'Atacante'
}

# Suponha que selected_position seja definido anteriormente, ex: selected_position = 'D'
# Carregue seu DataFrame df aqui (já limpo)
# df = pd.read_csv("...")

selected_labels = position_metrics[selected_position]

# Normalização
df_normalized = df.copy()
for col in selected_labels:
    if col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        df_normalized[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

# Top 5 jogadores da posição
top_players = df_normalized[df['position'] == selected_position].sort_values(
    by='score_normalizado', ascending=False
).head(5)

# Radar com Plotly
fig = go.Figure()

for _, row in top_players.iterrows():
    values = row[selected_labels].tolist()
    values += values[:1]  # Fecha o círculo

    labels = selected_labels + [selected_labels[0]]
    nome = row['name'] if 'name' in row else "Jogador"

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=nome,
        line=dict(width=2)
    ))

# Layout do gráfico
fig.update_layout(
    title=f"Radar - Top 5 {position_labels[selected_position]}s",
    width=800,     # largura do gráfico em pixels
    height=700,    # altura do gráfico em pixels
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    legend=dict(x=1.05, y=1)
)


# Mostrar no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Descrição
st.caption("Atributos normalizados com base em todo o conjunto de dados")







# GRAFICO 3
# Gráfico de correlação
st.subheader("Gráfico 3")
st.caption("Correlação das variáveis com o preço")




st.subheader("Modelo de ML")
st.caption("Predição de preços com base nos atributos")

