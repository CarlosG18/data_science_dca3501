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

def montar_time_ideal(df, budget, position_labels):
    st.subheader("🔧 Montagem Interativa do Time Ideal")
    st.caption("Selecione o orçamento máximo e veja a seleção sugerida com os 11 melhores jogadores dentro do limite.")

    # Entrada do usuário
    orcamento = st.slider("💰 Selecione o orçamento máximo para montar seu time:", 10_000_000, 200_000_000, budget, step=5_000_000, format="R$ %d")
    
    # Garantir que a coluna de pontuação está presente
    if 'score_normalizado' not in df.columns:
        df['score_normalizado'] = np.random.rand(len(df))  # substituir por uma métrica real se houver

    # Definir posições e quantidades
    esquema_tatico = {'G': 1, 'D': 4, 'M': 3, 'F': 3}
    global jogadores_selecionados
    jogadores_selecionados = []
    total_valor = 0

    for posicao, qtd in esquema_tatico.items():
        candidatos = df[df['position'] == posicao].sort_values(by='score_normalizado', ascending=False)
        selecionados = []

        for _, jogador in candidatos.iterrows():
            if len(selecionados) < qtd and total_valor + jogador['valor_mercado'] <= orcamento:
                selecionados.append(jogador)
                total_valor += jogador['valor_mercado']

        jogadores_selecionados.extend(selecionados)

    # Mostrar resultado
    if jogadores_selecionados:
        df_time = pd.DataFrame(jogadores_selecionados)
        st.success(f"✅ Time montado com sucesso! Total gasto: R$ {int(total_valor):,}".replace(",", "."))
        
        # Exibir como tabela
        st.dataframe(df_time[['name', 'position', 'valor_mercado', 'score_normalizado']].rename(columns={
            'name': 'Nome',
            'position': 'Posição',
            'valor_mercado': 'Valor de Mercado',
            'score_normalizado': 'Pontuação'
        }), use_container_width=True)
    else:
        st.warning("⚠️ Não foi possível montar um time com o orçamento definido.")

def exibir_time_em_campo(df_time, position_labels):
    st.markdown("## 🟢 Time em Campo (Formação 1-4-3-3)")
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
    cols = st.columns(4)
    for i, jogador in enumerate(posicoes['D']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    # Meio-campistas
    st.markdown("### 🎯 Meio-Campo", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, jogador in enumerate(posicoes['M']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    # Atacantes
    st.markdown("### 🎯 Ataque", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, jogador in enumerate(posicoes['F']):
        cols[i].markdown(format_player(jogador), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

montar_time_ideal(df, budget=100_000_000, position_labels=position_labels)
exibir_time_em_campo(pd.DataFrame(jogadores_selecionados), position_labels)

st.set_page_config(layout="wide")

# HTML e CSS para o campo de futebol
html_code = """
<style>
  .field-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh; /* Ajuste a altura conforme necessário */
    overflow: hidden; /* Para garantir que o campo não transborde */
  }

  .soccer-field {
    width: 90%;
    max-width: 1000px; /* Largura máxima para o campo */
    aspect-ratio: 100 / 60; /* Proporção aproximada de um campo de futebol */
    background-color: #588f27; /* Cor verde do campo */
    border: 5px solid white; /* Borda externa */
    position: relative;
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
    overflow: hidden; /* Garante que os elementos fora da borda não sejam visíveis, exceto os gols */
  }

  /* Círculo central */
  .center-circle {
    width: 20%;
    height: 33.33%; /* Ajuste com base na proporção desejada do círculo */
    border: 3px solid white;
    border-radius: 50%;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  /* Linha central */
  .center-line {
    width: 3px;
    height: 100%;
    background-color: white;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
  }

  /* Áreas de grande penalidade */
  .penalty-box-left, .penalty-box-right {
    width: 15%; /* Ajuste da largura */
    height: 44%; /* Ajuste da altura */
    border: 3px solid white;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    box-sizing: border-box;
  }

  .penalty-box-left {
    left: 0;
    border-left: none;
  }

  .penalty-box-right {
    right: 0;
    border-right: none;
  }

  /* Áreas de baliza */
  .goal-area-left, .goal-area-right {
    width: 7%; /* Ajuste da largura */
    height: 20%; /* Ajuste da altura */
    border: 3px solid white;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    box-sizing: border-box;
  }

  .goal-area-left {
    left: 0;
    border-left: none;
  }

  .goal-area-right {
    right: 0;
    border-right: none;
  }

  /* Balizas */
  .goal-left, .goal-right {
    width: 2%; /* Largura da baliza */
    height: 10%; /* Altura da baliza */
    background-color: lightgray;
    border: 2px solid darkgray;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    z-index: 10;
  }

  .goal-left {
    left: -2%; /* Posição fora do campo */
  }

  .goal-right {
    right: -2%; /* Posição fora do campo */
  }

  /* Marcas de grande penalidade */
  .penalty-spot-left, .penalty-spot-right {
    width: 1%;
    height: 1.66%; /* Relativo à altura do campo */
    background-color: white;
    border-radius: 50%;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
  }

  .penalty-spot-left {
    left: 10%; /* Ajuste da posição */
  }

  .penalty-spot-right {
    right: 10%; /* Ajuste da posição */
  }

  /* Arco penal (meia-lua) */
  .penalty-arc-left, .penalty-arc-right {
    width: 15%; /* Largura do contêiner do arco (mesma da área para facilitar o posicionamento) */
    height: 25%; /* Altura do contêiner do arco */
    border: 3px solid white;
    border-radius: 50%; /* Faz um círculo completo */
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
  }

  .penalty-arc-left {
    left: calc(15% - 7.5%); /* Posiciona o centro do círculo do arco na linha da área */
    border-top-color: transparent;
    border-bottom-color: transparent;
    border-left-color: transparent; /* Esconde a parte interna do círculo */
  }

  .penalty-arc-right {
    right: calc(15% - 7.5%); /* Posiciona o centro do círculo do arco na linha da área */
    border-top-color: transparent;
    border-bottom-color: transparent;
    border-right-color: transparent; /* Esconde a parte interna do círculo */
  }

  /* Estilo dos jogadores (MAIOR) */
  .player {
    position: absolute;
    width: 6%; /* Tamanho do jogador - AUMENTADO */
    height: 10%; /* Tamanho do jogador (proporcional à altura do campo) - AUMENTADO */
    border-radius: 50%; /* Formato de botão */
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 0.9em; /* Aumentado */
    font-weight: bold;
    color: white;
    text-align: center;
    flex-direction: column; /* Nome abaixo do botão */
    line-height: 1.2;
    z-index: 5; /* Garante que os jogadores fiquem acima do campo */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3); /* Sombra para dar profundidade */
  }

  .player-name {
    font-size: 0.7em; /* Tamanho da fonte do nome - Aumentado */
    color: white;
    margin-top: 0.3em; /* Espaçamento entre o botão e o nome */
    white-space: nowrap; /* Evita que o nome quebre a linha */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7); /* Sombra no texto para melhor legibilidade */
  }

  .team-a {
    background-color: #007bff; /* Azul */
    border: 2px solid #0056b3; /* Borda mais grossa */
  }

  /* Posições dos jogadores (agora só um time) - Formação 4-3-3 adaptada */
  /* Time A (Azul) */
  .player-a-gk { left: 5%; top: 50%; transform: translate(-50%, -50%); }
  .player-a-cb1 { left: 20%; top: 30%; transform: translate(-50%, -50%); }
  .player-a-cb2 { left: 20%; top: 70%; transform: translate(-50%, -50%); }
  .player-a-lb { left: 15%; top: 15%; transform: translate(-50%, -50%); }
  .player-a-rb { left: 15%; top: 85%; transform: translate(-50%, -50%); }
  .player-a-cm1 { left: 40%; top: 25%; transform: translate(-50%, -50%); }
  .player-a-cm2 { left: 40%; top: 50%; transform: translate(-50%, -50%); }
  .player-a-cm3 { left: 40%; top: 75%; transform: translate(-50%, -50%); }
  .player-a-lw { left: 60%; top: 15%; transform: translate(-50%, -50%); }
  .player-a-rw { left: 60%; top: 85%; transform: translate(-50%, -50%); }
  .player-a-st { left: 75%; top: 50%; transform: translate(-50%, -50%); }

</style>

<div class="field-container">
  <div class="soccer-field">
    <div class="center-circle"></div>
    <div class="center-line"></div>
    <div class="penalty-box-left"></div>
    <div class="penalty-box-right"></div>
    <div class="goal-area-left"></div>
    <div class="goal-area-right"></div>
    <div class="goal-left"></div>
    <div class="goal-right"></div>
    <div class="penalty-spot-left"></div>
    <div class="penalty-spot-right"></div>
    <div class="penalty-arc-left"></div>
    <div class="penalty-arc-right"></div>

    <div class="player team-a player-a-gk">GK<span class="player-name">Alisson</span></div>
    <div class="player team-a player-a-cb1">ZAG<span class="player-name">Marquinhos</span></div>
    <div class="player team-a player-a-cb2">ZAG<span class="player-name">Thiago Silva</span></div>
    <div class="player team-a player-a-lb">LE<span class="player-name">Alex Telles</span></div>
    <div class="player team-a player-a-rb">LD<span class="player-name">Danilo</span></div>
    <div class="player team-a player-a-cm1">MC<span class="player-name">Casemiro</span></div>
    <div class="player team-a player-a-cm2">MC<span class="player-name">Paquetá</span></div>
    <div class="player team-a player-a-cm3">MC<span class="player-name">Bruno G.</span></div>
    <div class="player team-a player-a-lw">PE<span class="player-name">Vini Jr.</span></div>
    <div class="player team-a player-a-rw">PD<span class="player-name">Raphinha</span></div>
    <div class="player team-a player-a-st">ATA<span class="player-name">Richarlison</span></div>

  </div>
</div>
"""

st.components.v1.html(html_code, height=600) # Ajuste a altura conforme necessário