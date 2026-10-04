from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sqlalchemy import create_engine
    TEM_SQLALCHEMY = True
except (ImportError, Exception):
    import sqlite3
    TEM_SQLALCHEMY = False

# ==============================================================================
# EXERCÍCIO 1 - ANÁLISE EXPLORATÓRIA DE VENDAS
# ==============================================================================

# Caminhos dos arquivos
DIRETORIO = Path(__file__).resolve().parent
CAMINHO_CSV = (
    DIRETORIO / "base_vendas_ecommerce.csv"
    if (DIRETORIO / "base_vendas_ecommerce.csv").exists()
    else DIRETORIO.parent / "ISW039_Lista_Exercicios_e_Bases" / "base_vendas_ecommerce.csv"
)
CAMINHO_DB = DIRETORIO / "vendas_analisadas.db"
CAMINHO_SQL = DIRETORIO / "exercicio01_vendas_analisadas.sql"


# === EXTRACT ===
# 1. Carrega os dados brutos e visualiza o início da base
df = pd.read_csv(CAMINHO_CSV, sep=';', encoding='utf-8-sig')

print("--- 1. Primeiras 5 linhas da base ---")
print(df.head())

# 2. Dimensões e tipos de dados identificados
print("\n--- 2. Dimensões e tipos de dados ---")
print(f"Total de registros: {df.shape[0]} linhas | Atributos: {df.shape[1]} colunas\n")
print(df.dtypes)

# 3. Quantidade de valores ausentes em cada coluna
print("\n--- 3. Valores ausentes por coluna ---")
print(df.isna().sum())


# === TRANSFORM ===
# 4. Conversão correta dos tipos numéricos e datas
df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce')
df['preco_unitario'] = pd.to_numeric(df['preco_unitario'], errors='coerce')
df['desconto'] = pd.to_numeric(df['desconto'], errors='coerce')
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# 5. Estatísticas descritivas dos atributos numéricos
print("\n--- 5. Estatísticas descritivas (média, mediana, mín, máx, desvio padrão) ---")
cols_num = ['quantidade', 'preco_unitario', 'desconto']
stats = pd.DataFrame({
    'Média': df[cols_num].mean(),
    'Mediana': df[cols_num].median(),
    'Mínimo': df[cols_num].min(),
    'Máximo': df[cols_num].max(),
    'Desvio Padrão': df[cols_num].std()
})
print(stats)

# 6. Criando o atributo valor_bruto (quantidade x preco_unitario)
df['valor_bruto'] = (df['quantidade'] * df['preco_unitario']).round(2)

# 7. Criando o atributo valor_liquido considerando o desconto informado
# Caso o desconto seja nulo, consideramos 0% para manter o valor da venda
df['valor_liquido'] = (df['valor_bruto'] * (1 - df['desconto'].fillna(0.0))).round(2)

print("\n--- 6 e 7. Amostra com valor bruto e líquido calculados ---")
print(df[['id_venda', 'produto', 'quantidade', 'preco_unitario', 'valor_bruto', 'desconto', 'valor_liquido']].head())

# 8. Identificando transações atípicas (outliers) utilizando Z-Score
# Transações com valor líquido a mais de 3 desvios padrão da média
z = (df['valor_liquido'] - df['valor_liquido'].mean()) / df['valor_liquido'].std()
outliers = df[np.abs(z) > 3]

print("\n--- 8. Identificação de Outliers (|Z| > 3) ---")
print(f"Média: R$ {df['valor_liquido'].mean():,.2f} | Desvio Padrão: R$ {df['valor_liquido'].std():,.2f}")
print(f"Total de transações atípicas identificadas: {len(outliers)}")
print("Exemplos de transações com maior valor líquido:")
print(outliers[['id_venda', 'produto', 'quantidade', 'preco_unitario', 'valor_liquido']].sort_values(by='valor_liquido', ascending=False).head(5))

# 9. Faturamento líquido total por região e categoria
print("\n--- 9. Faturamento Líquido Total ---")
print("\nPor Região:")
print(df.groupby('regiao', dropna=False)['valor_liquido'].sum())

print("\nPor Categoria:")
print(df.groupby('categoria', dropna=False)['valor_liquido'].sum())


# === LOAD ===
# 10. Armazena no SQLite e exporta o arquivo SQL de entrega
df_salvar = df.copy()
df_salvar['data'] = df_salvar['data'].dt.strftime('%Y-%m-%d')

if TEM_SQLALCHEMY:
    engine = create_engine(f'sqlite:///{CAMINHO_DB}')
    df_salvar.to_sql('vendas_analisadas', engine, if_exists='replace', index=False)
    conn = engine.raw_connection()
else:
    import sqlite3
    conn = sqlite3.connect(CAMINHO_DB)
    df_salvar.to_sql('vendas_analisadas', conn, if_exists='replace', index=False)

# Exportando o dump SQL contendo a estrutura e os registros
with open(CAMINHO_SQL, 'w', encoding='utf-8') as f:
    for linha in conn.iterdump():
        f.write(f"{linha}\n")
conn.close()

# Cópia para a raiz para facilitar a visualização
try:
    import shutil
    shutil.copyfile(CAMINHO_SQL, DIRETORIO.parent / CAMINHO_SQL.name)
except Exception:
    pass

print(f"\n[OK] Tabela 'vendas_analisadas' salva e arquivo '{CAMINHO_SQL.name}' gerado com sucesso!")
