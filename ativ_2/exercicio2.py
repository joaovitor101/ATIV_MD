from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sqlalchemy import create_engine
    TEM_SQLALCHEMY = True
except (ImportError, Exception):
    import sqlite3
    TEM_SQLALCHEMY = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    TEM_SKLEARN = True
except (ImportError, Exception):
    TEM_SKLEARN = False

# ==============================================================================
# EXERCÍCIO 2 - LIMPEZA, TRANSFORMAÇÃO E REDUÇÃO DE DIMENSIONALIDADE (PCA)
# ==============================================================================

# Caminhos dos arquivos
DIRETORIO = Path(__file__).resolve().parent
CAMINHO_CSV = (
    DIRETORIO / "base_clientes_varejo.csv"
    if (DIRETORIO / "base_clientes_varejo.csv").exists()
    else DIRETORIO.parent / "ISW039_Lista_Exercicios_e_Bases" / "base_clientes_varejo.csv"
)
CAMINHO_DB = DIRETORIO / "clientes_preprocessados.db"
CAMINHO_SQL = DIRETORIO / "exercicio02_clientes_preprocessados.sql"


# === EXTRACT: EXTRAÇÃO E DIAGNÓSTICO ===
# 1. Carrega os dados brutos e avalia tipos e nulos
df = pd.read_csv(CAMINHO_CSV, sep=';', encoding='utf-8-sig')

print("--- 1. Diagnóstico Inicial da Base ---")
print(f"Total de registros: {df.shape[0]} linhas | Atributos: {df.shape[1]} colunas\n")
print("Tipos de dados identificados:")
print(df.dtypes)

print("\nValores ausentes por coluna:")
print(df.isna().sum()[df.isna().sum() > 0])

# Identificação de duplicidades
print(f"\nLinhas totalmente duplicadas: {df.duplicated().sum()}")
print(f"Duplicidades por id_cliente: {df.duplicated(subset=['id_cliente']).sum()}")


# === TRANSFORM: LIMPEZA DE DADOS ===
# 2. Remoção de duplicidades pelo id do cliente
df = df.drop_duplicates(subset=['id_cliente'], keep='first').copy()
print(f"\nBase após remover duplicados: {len(df)} clientes únicos.")

# Padronização de campos de texto (espaços e capitalização)
df['nome'] = df['nome'].str.strip().str.title()
df['uf'] = df['uf'].str.strip().str.upper()

# Padronização e correção de grafias de cidades
def normalizar_cidade(nome):
    if pd.isna(nome):
        return nome
    c = str(nome).strip().title()
    correcoes = {
        'Sao Paulo': 'São Paulo',
        'Florianopolis': 'Florianópolis',
        'Brasilia': 'Brasília',
        'Goiania': 'Goiânia',
        'Rio De Janeiro': 'Rio de Janeiro'
    }
    return correcoes.get(c, c)

df['cidade'] = df['cidade'].apply(normalizar_cidade)

# Tratamento de valores ausentes utilizando mediana para numéricos e moda para categóricos
df['idade'] = df['idade'].fillna(df['idade'].median())
df['renda_mensal'] = df['renda_mensal'].fillna(df['renda_mensal'].median())
df['ticket_medio'] = df['ticket_medio'].fillna(df['ticket_medio'].median())
df['categoria_cliente'] = df['categoria_cliente'].fillna(df['categoria_cliente'].mode()[0])

# Investigação de outliers
print("\n--- Investigação de Outliers ---")
for col in ['idade', 'renda_mensal', 'ticket_medio']:
    z = (df[col] - df[col].mean()) / df[col].std()
    qtd_outliers = (np.abs(z) > 3).sum()
    print(f"Atributo '{col}': {qtd_outliers} valores com |Z| > 3 (faixa de {df[col].min():,.2f} a {df[col].max():,.2f})")

# Decisão: Manter os registros, pois representam clientes reais da rede (idosos e VIPs)
# que trazem variabilidade legítima aos dados.


# === TRANSFORM: CODIFICAÇÃO E REDUÇÃO DE DIMENSIONALIDADE (PCA) ===
# Codificação ordinal da categoria do cliente
mapa_categoria = {'Bronze': 1, 'Prata': 2, 'Ouro': 3, 'Diamante': 4}
df['categoria_cliente_cod'] = df['categoria_cliente'].map(mapa_categoria).astype(int)

# Seleção dos atributos numéricos para padronização e PCA
cols_num = [
    'idade', 'renda_mensal', 'tempo_cliente_meses',
    'compras_12m', 'ticket_medio', 'dias_desde_ultima_compra', 'categoria_cliente_cod'
]

# Padronização (StandardScaler)
if TEM_SKLEARN:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols_num])
    
    # Aplicação do PCA com 2 componentes
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled)
    variancia = pca.explained_variance_ratio_
else:
    X_vals = df[cols_num].values
    X_scaled = (X_vals - np.mean(X_vals, axis=0)) / np.std(X_vals, axis=0)
    cov_mat = np.cov(X_scaled, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(cov_mat)
    idx = np.argsort(autovalores)[::-1]
    autovalores, autovetores = autovalores[idx], autovetores[:, idx]
    componentes = np.dot(X_scaled, autovetores[:, :2])
    variancia = autovalores[:2] / np.sum(autovalores)

# Inclusão dos componentes no DataFrame final
df['componente_1'] = componentes[:, 0].round(4)
df['componente_2'] = componentes[:, 1].round(4)

print("\n--- Variância Explicada pelo PCA ---")
print(f"Componente 1: {variancia[0] * 100:.2f}% | Componente 2: {variancia[1] * 100:.2f}%")
print(f"Variância Total Acumulada: {np.sum(variancia) * 100:.2f}%\n")
print("Amostra final dos dados com PCA:")
print(df[['id_cliente', 'nome', 'cidade', 'renda_mensal', 'categoria_cliente', 'componente_1', 'componente_2']].head())


# === LOAD: ARMAZENAMENTO E GERAÇÃO SQL ===
if TEM_SQLALCHEMY:
    engine = create_engine(f'sqlite:///{CAMINHO_DB}')
    df.to_sql('clientes_preprocessados', engine, if_exists='replace', index=False)
    conn = engine.raw_connection()
else:
    import sqlite3
    conn = sqlite3.connect(CAMINHO_DB)
    df.to_sql('clientes_preprocessados', conn, if_exists='replace', index=False)

# Exportando o dump SQL contendo estrutura e dados
with open(CAMINHO_SQL, 'w', encoding='utf-8') as f:
    for linha in conn.iterdump():
        f.write(f"{linha}\n")
conn.close()

# Cópia para a raiz
try:
    import shutil
    shutil.copyfile(CAMINHO_SQL, DIRETORIO.parent / CAMINHO_SQL.name)
except Exception:
    pass

print(f"\n[OK] Tabela 'clientes_preprocessados' salva e arquivo '{CAMINHO_SQL.name}' gerado com sucesso!")
