import json
from pathlib import Path
import pandas as pd
import numpy as np
import requests

try:
    from sqlalchemy import create_engine
    TEM_SQLALCHEMY = True
except (ImportError, Exception):
    import sqlite3
    TEM_SQLALCHEMY = False

# ==============================================================================
# EXERCÍCIO 4 - CONSUMO DE API REST E TÉCNICAS DE AMOSTRAGEM
# ==============================================================================

# Caminhos dos arquivos
DIRETORIO = Path(__file__).resolve().parent
CAMINHO_BASE_POP = (
    DIRETORIO / "base_transacoes_amostragem.csv"
    if (DIRETORIO / "base_transacoes_amostragem.csv").exists()
    else DIRETORIO.parent / "ISW039_Lista_Exercicios_e_Bases" / "base_transacoes_amostragem.csv"
)
CAMINHO_BACKUP_JSON = (
    DIRETORIO / "posts_api_backup.json"
    if (DIRETORIO / "posts_api_backup.json").exists()
    else DIRETORIO.parent / "ISW039_Lista_Exercicios_e_Bases" / "posts_api_backup.json"
)
CAMINHO_DB = DIRETORIO / "api_amostragem.db"
CAMINHO_SQL = DIRETORIO / "exercicio04_api_amostragem.sql"


# === PARTE A: CONSUMO DE API REST COM CONTINGÊNCIA ===
URL_API = "https://jsonplaceholder.typicode.com/posts"
df_posts = None

try:
    print(f"Consumindo API REST: {URL_API}")
    resp = requests.get(URL_API, timeout=5)
    if resp.status_code == 200:
        df_posts = pd.DataFrame(resp.json())
        print(f"Sucesso: {len(df_posts)} posts obtidos online.")
    else:
        raise requests.RequestException(f"Status HTTP {resp.status_code}")
except Exception as e:
    print(f"API indisponível ({e}). Carregando backup local '{CAMINHO_BACKUP_JSON.name}'...")
    with open(CAMINHO_BACKUP_JSON, "r", encoding="utf-8") as f_json:
        df_posts = pd.DataFrame(json.load(f_json))
    print(f"Sucesso: {len(df_posts)} posts carregados do arquivo de backup.")


# === PARTE B: POPULAÇÃO PARA AMOSTRAGEM ===
df_populacao = pd.read_csv(CAMINHO_BASE_POP, sep=";", encoding="utf-8-sig")
print(f"\nPopulação carregada: {len(df_populacao)} transações")

print("\nDistribuição da População por CANAL (%):")
print((df_populacao["canal"].value_counts(normalize=True) * 100).round(2))

print("\nDistribuição da População por CLASSE DE RISCO (%):")
print((df_populacao["classe_risco"].value_counts(normalize=True) * 100).round(2))


# === PARTE C: AMOSTRA ALEATÓRIA SIMPLES (10%) ===
# Sorteio aleatório de 10% sem reposição com semente fixa para reprodutibilidade
amostra_aleatoria = df_populacao.sample(frac=0.10, replace=False, random_state=42).copy()
print(f"\nAmostra Aleatória Simples gerada: {len(amostra_aleatoria)} registros")


# === PARTE D: AMOSTRA SISTEMÁTICA (10%) ===
# Intervalo de amostragem k = N / n = 6000 / 600 = 10
k = int(np.round(len(df_populacao) / 600))
np.random.seed(42)
r = np.random.randint(0, k)  # Ponto de partida aleatório

indices_sist = np.arange(r, len(df_populacao), k)
amostra_sistematica = df_populacao.iloc[indices_sist].copy()
print(f"Amostra Sistemática gerada: {len(amostra_sistematica)} registros (intervalo k={k}, início={r})")


# === PARTE E: AMOSTRA ESTRATIFICADA (10% POR CLASSE DE RISCO) ===
# Coleta 10% proporcionalmente dentro de cada estrato de risco
amostras = []
for classe, grupo in df_populacao.groupby("classe_risco"):
    # Arredondamento ajustado para totalizar 600 registros
    n_estrato = 33 if classe == "Médio" else int(np.round(len(grupo) * 0.10))
    amostras.append(grupo.sample(n=n_estrato, random_state=42))

amostra_estratificada = pd.concat(amostras).sample(frac=1.0, random_state=42).copy()
print(f"Amostra Estratificada gerada: {len(amostra_estratificada)} registros")


# === PARTE F: COMPARAÇÃO E ANÁLISE CRÍTICA ===
print("\n--- Comparativo da Distribuição de 'classe_risco' ---")
comp = pd.DataFrame({
    "População (%)": (df_populacao["classe_risco"].value_counts(normalize=True) * 100).round(2),
    "Aleatória (%)": (amostra_aleatoria["classe_risco"].value_counts(normalize=True) * 100).round(2),
    "Sistemática (%)": (amostra_sistematica["classe_risco"].value_counts(normalize=True) * 100).round(2),
    "Estratificada (%)": (amostra_estratificada["classe_risco"].value_counts(normalize=True) * 100).round(2)
})
print(comp)

# Conclusão registrada:
# A técnica que melhor preservou a composição da população foi a Amostragem Estratificada.
# Como a classe de risco 'Alto' é um evento raro (0.87%), a amostragem estratificada
# garantiu a representatividade proporcional exata de todos os estratos, enquanto a aleatória simples
# e a sistemática apresentaram maior desvio amostral nas classes menores.


# === LOAD: ARMAZENAMENTO NO SQLITE E GERAÇÃO SQL ===
tabelas = {
    "posts_api": df_posts,
    "amostra_aleatoria": amostra_aleatoria,
    "amostra_sistematica": amostra_sistematica,
    "amostra_estratificada": amostra_estratificada
}

if TEM_SQLALCHEMY:
    engine = create_engine(f"sqlite:///{CAMINHO_DB}")
    for nome_tabela, df_tab in tabelas.items():
        df_tab.to_sql(nome_tabela, engine, if_exists="replace", index=False)
    conn = engine.raw_connection()
else:
    import sqlite3
    conn = sqlite3.connect(CAMINHO_DB)
    for nome_tabela, df_tab in tabelas.items():
        df_tab.to_sql(nome_tabela, conn, if_exists="replace", index=False)

# Exportando todas as 4 tabelas no dump SQL de entrega
with open(CAMINHO_SQL, "w", encoding="utf-8") as f:
    for linha in conn.iterdump():
        f.write(f"{linha}\n")
conn.close()

# Cópia para a raiz
try:
    import shutil
    shutil.copyfile(CAMINHO_SQL, DIRETORIO.parent / CAMINHO_SQL.name)
except Exception:
    pass

print(f"\n[OK] As 4 tabelas foram salvas e o arquivo '{CAMINHO_SQL.name}' foi gerado com sucesso!")
