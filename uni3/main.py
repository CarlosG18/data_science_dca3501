import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

# Carregar dados
df = pd.read_csv("../uni2/dataset/melhores_jogadores.csv")

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

# GRAFICO 3
# Gráfico de correlação
st.subheader("Gráfico 3")
st.caption("Correlação das variáveis com o preço")

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

def montar_time_ideal(df, budget, position_labels):
  st.subheader("🔧 Montagem Otimizada do Time Ideal")
  st.caption("Seleciona automaticamente os 11 melhores jogadores respeitando posições e orçamento.")

  orcamento = st.slider("💰 Selecione o orçamento máximo para montar seu time:", 
                        10_000_000, 200_000_000, budget, step=5_000_000, format="R$ %d")

  if 'score_normalizado' not in df.columns:
      df['score_normalizado'] = np.random.rand(len(df))  # substituir com métrica real

  # Definir seletor de esquema tático
  st.markdown("### ⚽ Escolha o esquema tático:")
  st.caption("Selecione o esquema tático desejado para a formação do time.")
  global widget_esquema_tatico
  widget_esquema_tatico = st.selectbox(
      "Selecione o esquema tático:",
      options=["4-3-3", "3-5-2", "4-5-1", "4-4-2", "3-4-3", "4-6-0", "5-3-2"],
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
      st.dataframe(df_time[['name', 'position', 'valor_mercado', 'score_normalizado']].rename(columns={
          'name': 'Nome',
          'position': 'Posição',
          'valor_mercado': 'Valor de Mercado',
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
        return f"""<div style="background:#000; border-radius:12px; padding:10px; text-align:center; box-shadow:2px 2px 6px #00000033; margin:5px;">
            <strong>{jogador['name']}</strong><br>
            <span style="font-size:12px;">💰 R$ {int(jogador['valor_mercado']):,}</span><br>
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


# UNIDADE 3 - Machine Learning

st.markdown("""
## Machine Learning - predição do valor de mercado dos jogadores

Esta seção utiliza um modelo de aprendizado de máquina para sugerir a formação ideal de um time com base nos atributos dos jogadores e no valor de mercado. A ideia é montar uma equipe equilibrada, maximizando o desempenho dentro de um orçamento definido.
""")

X = pd.read_csv('./dataset/X.csv')
y = pd.read_csv('./dataset/y.csv')

# Suponha que X e y já estejam carregados

# 1. Combinar X e y em um único DataFrame temporário
df_temp = X.drop(columns=['name'])
df_temp['y'] = y

# 2. Calcular a correlação de todas as colunas com y
correlacoes = df_temp.corr(numeric_only=True)['y'].drop('y')

# 3. Selecionar os 10 maiores em valor absoluto
top_10 = correlacoes.abs().sort_values(ascending=False).head(10).index

# 4. Criar matriz de correlação com as 10 + y
colunas_top = list(top_10) + ['y']
matriz_correlacao = df_temp[colunas_top].corr()

# 5. Visualizar com heatmap no Streamlit
st.title("Heatmap das 10 maiores correlações com y")

fig, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
st.pyplot(fig)

# Exibir as 10 maiores correlações em tabela
st.subheader("Top 10 correlações com y (valor absoluto)")
st.dataframe(correlacoes.loc[top_10].sort_values(key=abs, ascending=False).to_frame(name="correlação"))

# --- LazyPredict ---
df_temp = X.copy()
df_temp['y'] = y
top_10 = correlacoes.abs().sort_values(ascending=False).head(10).index.tolist()
X_sel = df_temp[top_10]
y_sel = df_temp['y']

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y_sel, test_size=0.2, random_state=42
)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

st.subheader("📉 Top 10 Modelos com Maior R²")

# Resetar índice e renomear para 'Model'
models_reset = models.reset_index().rename(columns={'index': 'Model'})

# Selecionar top 10 pelo R-Squared
top_10_models = models_reset.nlargest(10, 'R-Squared')

# Ordenar para gráfico horizontal
models_sorted = top_10_models.sort_values(by='R-Squared', ascending=True)

# Plotar gráfico de barras horizontal
fig = px.bar(
    models_sorted,
    x='R-Squared',
    y='Model',   # Agora é a coluna correta
    orientation='h',
    text='R-Squared',
    color='R-Squared',
    color_continuous_scale='Blues',
    title="Top 10 Modelos por R²",
    labels={'R-Squared': 'Coeficiente de Determinação (R²)', 'Model': 'Modelo'}
)

fig.update_traces(texttemplate='%{text:.3f}', textposition='inside')
fig.update_layout(
    xaxis_title='R²',
    yaxis_title='',
    yaxis=dict(categoryorder='total ascending'),
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)


# --- Previsões com o melhor modelo ---
melhor_modelo_nome = models['R-Squared'].idxmax()
melhor_modelo = reg.models[melhor_modelo_nome]
y_pred = melhor_modelo.predict(X_sel)
residuos = y_sel - y_pred
desvio_padrao = np.std(residuos)

df_resultados = X_sel.copy()
df_resultados['name'] = X['name']
df_resultados['Preço Real'] = y_sel
df_resultados['Preço Previsto'] = y_pred

meio = int(df_resultados.shape[0]/2)

# Ordena do mais barato para o mais caro
df_ord = df_resultados.sort_values(by='Preço Real', ascending=True)

# 'meio' deve estar definido antes — índice do meio (ex: meio = len(df_ord)//2)
top_jogadores = df_ord.iloc[meio+20:meio+30, :]

# Formata os valores como moeda brasileira (R$)
formatar_moeda = lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

top_jogadores = df_resultados.sort_values(by='Preço Real', ascending=False).head(10)

top_jogadores_fmt = top_jogadores[['name', 'Preço Real', 'Preço Previsto']].copy()
top_jogadores_fmt['Preço Real'] = top_jogadores_fmt['Preço Real'].apply(formatar_moeda)
top_jogadores_fmt['Preço Previsto'] = top_jogadores_fmt['Preço Previsto'].apply(formatar_moeda)

st.subheader("⚽ Top 10 Jogadores com Maior Preço de Mercado Real")
st.markdown(f"**Modelo utilizado:** {melhor_modelo_nome}")
st.table(top_jogadores_fmt.set_index('name'))

# Ordena do mais barato para o mais caro
df_ord = df_resultados.sort_values(by='Preço Real', ascending=True)

# 'meio' deve estar definido antes — índice do meio (ex: meio = len(df_ord)//2)
top_jogadores = df_ord.iloc[meio+20:meio+30, :]

# Formata os valores como moeda brasileira (R$)
formatar_moeda = lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

top_jogadores_fmt = top_jogadores[['name', 'Preço Real', 'Preço Previsto']].copy()
top_jogadores_fmt['Preço Real'] = top_jogadores_fmt['Preço Real'].apply(formatar_moeda)
top_jogadores_fmt['Preço Previsto'] = top_jogadores_fmt['Preço Previsto'].apply(formatar_moeda)

# Exibição no Streamlit
st.subheader("⚽ Top 10 Jogadores da Meiuca dos Preços")
st.table(top_jogadores_fmt.set_index('name'))

def formatar_moeda(valor):
    return f"R$ {valor:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

# Histograma do valor real
fig_real = px.histogram(
    y, 
    nbins=30, 
    title="Distribuição do Valor de Mercado Real",
    labels={'value': 'Valor de Mercado Real'},
    color_discrete_sequence=['blue']
)
fig_real.update_layout(bargap=0.1)
st.plotly_chart(fig_real, use_container_width=True)

# Histograma do valor previsto
fig_pred = px.histogram(
    y_pred, 
    nbins=30, 
    title="Distribuição do Valor de Mercado Predição",
    labels={'value': 'Valor de Mercado Predição'},
    color_discrete_sequence=['orange']
)
fig_pred.update_layout(bargap=0.1)
st.plotly_chart(fig_pred, use_container_width=True)
