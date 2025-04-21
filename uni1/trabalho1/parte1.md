# 📊 Trabalho 1 - Parte I - Projeto de Análise de Dados

> Projeto de Análise de Dados — Talentos Subvalorizados no Futebol

### 📁 1. Apresentação do Dataset
Utilizamos um conjunto de dados disponível no [Kaggle](https://www.kaggle.com/datasets/felipesembay/sofascore-and-transfermarkt-football-data), com foco em informações estatísticas de jogadores e seus respectivos valores de mercado.

#### Arquivos disponíveis:
- `statistics_player.csv`: Estatísticas por partida de cada jogador.
- `market_value.csv`: Valores de mercado históricos dos jogadores.

---

### 🔍 2. Variáveis e Dados Faltantes

#### 2.1 `statistics_player.csv`
Contém variáveis como:
- Gols
- Assistências
- Passes certos
- Finalizações
- Defesas
- Faltas sofridas/cometidas

**Observações sobre dados faltantes:**
- Algumas estatísticas não se aplicam a certas posições (ex: goleiros não fazem gols).
- Quando **todas as estatísticas são nulas**, o jogador não entrou em campo (foi apenas reserva) — esses registros serão **descartados**.

#### 2.2 `market_value.csv`
Variáveis importantes:
- `x`: timestamp da data (será convertido para `dd/mm/yyyy`).
- `mw`: valor de mercado com símbolo (ex: `€500k`).
- `y`: valor numérico equivalente (float).
- `verein`: clube do jogador.
- `age`: idade.
- `player_id`, `player_name`: identificação dos jogadores.

> Este arquivo **não possui dados faltantes relevantes** e será essencial para cruzamento com estatísticas de desempenho.

---

### 🛠️ 3. Pré-processamento dos Dados

As seguintes transformações serão realizadas:

- Conversão de valores monetários com símbolos (ex: `€750k`, `R$1.2m`) para números inteiros (ex: `750000`, `1200000`).
- Conversão de timestamps para **datas legíveis** (`dd/mm/yyyy`).
- Substituição de **valores nulos parciais** por `0`.
- Remoção de registros com **todas as estatísticas nulas** (jogadores não utilizados).
- Conversão de variáveis categóricas (ex: clubes, posições) com **One-Hot Encoding** ou **Label Encoding** (se necessário para modelos preditivos).

---

### 📈 4. Objetivo das Análises

Nosso objetivo é **identificar jogadores com alto desempenho estatístico, mas com valores de mercado baixos**, ou seja, **potenciais contratações com ótimo custo-benefício**.

Essa análise pode ser muito útil para clubes que buscam **reforços eficientes e acessíveis financeiramente**.

### 🎥 Link do video

O video da apresentação do dataset esta [aqui](https://www.youtube.com/watch?v=LYOqcKhHylM)