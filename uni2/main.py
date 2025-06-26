import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os

# Carregar dados
df = pd.read_csv("uni2\dataset\melhores_jogadores.csv")

#path = os.path.join("dataset", "melhores_jogadores.csv")
#if not os.path.exists(path):
    #raise FileNotFoundError(f"Arquivo não encontrado: {path}")

#df = pd.read_csv(path)

# Separar colunas numéricas e não numéricas (excluindo name e team_name)
numericas = df.select_dtypes(include='number').columns.tolist()
nao_numericas = df.select_dtypes(exclude='number').columns.difference(['name', 'team_name']).tolist()

# Agrupar por nome e time
df_agrupado = df.groupby(['name', 'team_name'], as_index=False).agg(
    {col: 'mean' for col in numericas} | {col: 'first' for col in nao_numericas}
)

df = df_agrupado
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
position_labels = {'G': 'Goleiro', 'D': 'Zagueiro', 'M': 'Meia', 'F': 'Atacante'}
# Sidebar para escolha da posição
selected_position = st.sidebar.selectbox(
    "Selecione a posição:",
    list(position_labels.keys()),
    format_func=lambda x: position_labels[x],
    key="position_selectbox"
)

#values = st.sidebar.slider("Selecione a faixa de preço dos jogadores", 0.0, 100.0, (25.0, 75.0))
intervalo = st.sidebar.slider("Escolha o intervalo de valor", min_valor, max_valor, (min_valor, max_valor), step=step, format="R$ %d")


# parte introdutoria
st.title('Dashboard - Statistics Players')
st.markdown("""
## 📊 Dashboard Interativo – Análise de Dados

Este dashboard foi desenvolvido com **Streamlit** como parte do Trabalho 1 da disciplina **Ciência de Dados (DCA3501)**. Ele apresenta, de forma interativa e visual, os principais resultados da análise exploratória realizada no notebook original, facilitando a interpretação dos dados e das métricas estatísticas geradas durante o estudo.
""")

# Primeira linha: gráfico 1 e gráfico 2
# col1, col2 = st.columns(2)

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
st.subheader(f"Radar - Top 5 {position_labels[selected_position]}s")

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
    width=600,     # largura do gráfico em pixels
    height=500,    # altura do gráfico em pixels
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    legend=dict(x=1.05, y=1)
)

# Mostrar no Streamlit
st.plotly_chart(fig, use_container_width=True)

# GRAFICO 3
# Gráfico de correlação
st.subheader("Gráfico 3")

def correlacao(df):
    st.markdown("## Heamap de correlação Interativo")

    tipo = st.selectbox("Escolha o tipo de jogador:", ["Todos", "Goleiros", "Jogadores de linha"])

    # 1. Filtrar por tipo de jogador
    if tipo == "Goleiros":
        variaveis = [
        'rating',
        'saves',
        'savedShotsFromInsideTheBox',
        'goodHighClaim',
        'accuratePass',
        'height',
        'goalsPrevented',
        'age',
        'penaltyConceded',
        'valor_mercado'
        ]
        df_filtrado = df[df['position'] == 'G']
        # Junta todas as colunas
        df_filtrado = df_filtrado[variaveis]
        
    elif tipo == "Jogadores de linha":
        df_filtrado = df[df['position'] != 'G']
    else:
        df_filtrado = df.copy()

    if df_filtrado.shape[0] < 2:
        st.warning("⚠️ Dados insuficientes para calcular a correlação.")
        return

    # 2. Copiar e calcular correlações
    df_temp = df_filtrado.copy()
    correlacoes = df_temp.corr(numeric_only=True)

    if 'valor_mercado' not in correlacoes.columns:
        st.warning("⚠️ 'valor_mercado' não está entre as colunas numéricas.")
        return

    # 3. Selecionar as 10 variáveis com maior correlação absoluta com 'valor_mercado'
    top10_vars = correlacoes['valor_mercado'].drop('valor_mercado').abs().sort_values(ascending=False).head(10).index.tolist()
    colunas_plot = top10_vars + ['valor_mercado']

    # 4. Criar nova matriz com essas colunas
    matriz_reduzida = df_temp[colunas_plot].corr()

    # 5. Exibir com Plotly
    fig = px.imshow(
        matriz_reduzida,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"Top 10 correlações com valor_mercado - {tipo}"
    )

    st.plotly_chart(fig, use_container_width=True)

correlacao(df.drop(columns=['score_total','score_normalizado']))


# GRAFICO 4
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus

def get_esquema_tatico(esquema):
    """Define o esquema tático com base na formação escolhida."""
    if esquema == "4-3-3":
        return {'G': 1, 'D': 4, 'M': 3, 'F': 3}
    elif esquema == "3-5-2":
        return {'G': 1, 'D': 3, 'M': 5, 'F': 2}
    elif esquema == "4-5-1":
        return {'G': 1, 'D': 4, 'M': 5, 'F': 1}
    elif esquema == "4-4-2":
        return {'G': 1, 'D': 4, 'M': 4, 'F': 2}
    elif esquema == "3-4-3":
        return {'G': 1, 'D': 3, 'M': 4, 'F': 3}
    elif esquema == "4-6-0":
        return {'G': 1, 'D': 4, 'M': 6, 'F': 0}
    elif esquema == "5-3-2":
        return {'G': 1, 'D': 5, 'M': 3, 'F': 2}
    else:
        raise ValueError("Esquema tático desconhecido.")


def formatar_com_pontos(valor):
    return f"{valor:,}".replace(",", ".")


def montar_time_ideal(df, budget, position_labels):
  st.subheader("🔧 Montagem Otimizada do Time Ideal")
  st.caption("Seleciona automaticamente os 11 melhores jogadores respeitando posições e orçamento.")

  orcamento = st.slider("💰 Selecione o orçamento máximo para montar seu time:", 
                        10_000_000, 700_000_000, budget, step=5_000_000, format="R$ %d")

  if 'score_normalizado' not in df.columns:
      df['score_normalizado'] = np.random.rand(len(df))  # substituir com métrica real

  # Definir seletor de esquema tático
  st.markdown("### ⚽ Escolha o esquema tático:")
  st.caption("Selecione o esquema tático desejado para a formação do time.")
  global widget_esquema_tatico
  widget_esquema_tatico = st.selectbox(
      "Selecione o esquema tático:",
      options=["4-3-3", "3-5-2", "4-5-1", "4-4-2", "3-4-3", "5-3-2"],
      index=0,
      key="esquema_tatico"
  )

  # Definir esquema tático

  esquema_tatico = get_esquema_tatico(widget_esquema_tatico)

  # Criação do modelo de otimização
  modelo = LpProblem("Selecao_Time_Ideal", LpMaximize)

  # Variáveis de decisão
  jogadores_vars = {
      i: LpVariable(f"jogador_{i}", cat=LpBinary)
      for i in df.index
  }

  # Função objetivo: maximizar score total
  modelo += lpSum(jogadores_vars[i] * df.loc[i, 'score_normalizado'] for i in df.index)

  # Restrição de orçamento
  modelo += lpSum(jogadores_vars[i] * df.loc[i, 'valor_mercado'] for i in df.index) <= orcamento

  # Restrição de número de jogadores por posição
  for pos, qtd in esquema_tatico.items():
      modelo += lpSum(jogadores_vars[i] for i in df[df['position'] == pos].index) == qtd

  # Resolver o modelo
  modelo.solve()

  if LpStatus[modelo.status] == 'Optimal':
    selecionados = [i for i in df.index if jogadores_vars[i].varValue == 1]
    df_time = df.loc[selecionados]
    total_valor = df_time['valor_mercado'].sum()
    st.success(f"✅ Time montado com sucesso! Total gasto: R$ {int(total_valor):,}".replace(",", "."))

    # Criar uma cópia para formatar só para exibição
    df_exibir = df_time.copy()

    # Formatar a coluna valor_mercado para string com pontos como separador de milhar
    df_exibir['valor_mercado'] = df_exibir['valor_mercado'].apply(lambda x: f"{int(x):,}".replace(",", "."))

    # Mostrar no Streamlit com as colunas renomeadas
    st.dataframe(df_exibir[['name', 'position', 'valor_mercado', 'score_normalizado']].rename(columns={
        'name': 'Nome',
        'position': 'Posição',
        'valor_mercado': 'Valor de Mercado (R$)',
        'score_normalizado': 'Pontuação'
    }), use_container_width=True)

    return df_time
  else:
    st.warning("⚠️ Não foi possível montar um time com as restrições definidas (orçamento muito baixo).")


def exibir_time_em_campo(df_time, position_labels):
    st.markdown(f"## 🟢 Time em Campo (Formação {widget_esquema_tatico})")
    st.markdown("Visualização dos jogadores como se estivessem dispostos em um campo de futebol.")

    # Agrupar jogadores por posição
    posicoes = {'G': [], 'D': [], 'M': [], 'F': []}
    for _, row in df_time.iterrows():
        posicoes[row['position']].append(row)

    def format_player(jogador):
        valor_formatado = f"{int(jogador['valor_mercado']):,}".replace(",", ".")
        return f"""<div style="background:#cce7ff; border-radius:12px; padding:10px; text-align:center; box-shadow:2px 2px 6px #00000033; margin:5px;">
            <strong>{jogador['name']}</strong><br>
            <span style="font-size:12px;">💰 R$ {valor_formatado}</span><br>
            <span style="font-size:12px;">⭐ {jogador['score_normalizado']:.2f}</span>
    </div>"""
    
    # Goleiro
    st.markdown("### 🧤 Goleiro", unsafe_allow_html=True)
    col = st.columns(1)
    if posicoes['G']:
        col[0].markdown(format_player(posicoes['G'][0]), unsafe_allow_html=True)

    # Defensores
    st.markdown("### 🛡️ Defesa", unsafe_allow_html=True)
    cols = st.columns(len(posicoes['D']))
    for i, jogador in enumerate(posicoes['D']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    # Meio-campistas
    st.markdown("### 🎯 Meio-Campo", unsafe_allow_html=True)
    cols = st.columns(len(posicoes['M']))
    for i, jogador in enumerate(posicoes['M']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    # Atacantes
    st.markdown("### 🎯 Ataque", unsafe_allow_html=True)
    cols = st.columns(len(posicoes['F']))
    for i, jogador in enumerate(posicoes['F']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

df_time_ideal = montar_time_ideal(df, budget=100_000_000, position_labels=position_labels)
if df_time_ideal is not None:
  # Exibir o time em campo
  exibir_time_em_campo(pd.DataFrame(df_time_ideal), position_labels)