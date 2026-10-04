import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

try:
    from sqlalchemy import create_engine
    TEM_SQLALCHEMY = True
except (ImportError, Exception):
    import sqlite3
    TEM_SQLALCHEMY = False

# ==============================================================================
# EXERCÍCIO 3 - WEB SCRAPING E TRATAMENTO DE PRODUTOS
# ==============================================================================

# Caminhos dos arquivos
DIRETORIO = Path(__file__).resolve().parent
CAMINHO_HTML = (
    DIRETORIO / "catalogo_produtos.html"
    if (DIRETORIO / "catalogo_produtos.html").exists()
    else DIRETORIO.parent / "ISW039_Lista_Exercicios_e_Bases" / "catalogo_produtos.html"
)
CAMINHO_DB = DIRETORIO / "produtos_raspados.db"
CAMINHO_SQL = DIRETORIO / "exercicio03_produtos_raspados.sql"


# === EXTRACT: LEITURA E PARSING HTML ===
print("Lendo catálogo HTML...")
with open(CAMINHO_HTML, "r", encoding="utf-8") as arq:
    soup = BeautifulSoup(arq.read(), "html.parser")

# Localiza todos os cartões de produtos
cards = soup.find_all("article", class_="produto-card")
print(f"Total de cartões encontrados no HTML: {len(cards)}")

# Extração dos dados de cada produto (com verificação para evitar quebras por tags ausentes)
produtos = []
for card in cards:
    tag_nome = card.find("h2", class_="produto-nome")
    tag_cat = card.find("span", class_="produto-categoria")
    tag_preco = card.find("span", class_="produto-preco")
    tag_est = card.find("span", class_="produto-estoque")
    tag_av = card.find("span", class_="produto-avaliacao")

    produtos.append({
        "id_produto": card.get("data-id"),
        "nome": tag_nome.get_text() if tag_nome else None,
        "categoria": tag_cat.get_text() if tag_cat else None,
        "preco_raw": tag_preco.get_text() if tag_preco else None,
        "estoque_raw": tag_est.get_text() if tag_est else None,
        "avaliacao_raw": tag_av.get_text() if tag_av else None
    })

df = pd.DataFrame(produtos)


# === TRANSFORM: LIMPEZA E PADRONIZAÇÃO ===
# Limpeza de texto: remoção de espaços invisíveis (\xa0) e espaços repetidos
def limpar_texto(t):
    if not t or pd.isna(t):
        return None
    return re.sub(r"\s+", " ", str(t).replace("\xa0", " ")).strip()

df["id_produto"] = df["id_produto"].apply(limpar_texto)
df["nome"] = df["nome"].apply(limpar_texto)
df["categoria"] = df["categoria"].apply(limpar_texto)

# Conversão do preço (R$ 1.234,56 -> float 1234.56)
def converter_preco(val):
    if not val or pd.isna(val):
        return np.nan
    limpo = re.sub(r"[^\d,]", "", str(val)).replace(",", ".")
    try:
        return float(limpo)
    except ValueError:
        return np.nan

df["preco"] = df["preco_raw"].apply(converter_preco)

# Extração do estoque (número inteiro)
def extrair_estoque(val):
    if not val or pd.isna(val):
        return np.nan
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else np.nan

df["estoque"] = df["estoque_raw"].apply(extrair_estoque)

# Extração da nota de avaliação (número decimal)
def extrair_avaliacao(val):
    if not val or pd.isna(val):
        return np.nan
    m = re.search(r"(\d+[\.,]\d+|\d+)", str(val))
    return float(m.group().replace(",", ".")) if m else np.nan

df["avaliacao"] = df["avaliacao_raw"].apply(extrair_avaliacao)

# Remove colunas brutas intermediárias
df = df.drop(columns=["preco_raw", "estoque_raw", "avaliacao_raw"])

# Remoção de duplicidades pelo ID do produto
qtd_duplicados = df.duplicated(subset=["id_produto"]).sum()
df = df.drop_duplicates(subset=["id_produto"], keep="first").copy()
print(f"Duplicidades removidas: {qtd_duplicados} | Total consolidado: {len(df)} produtos")

# Adiciona a data de execução da coleta
df["data_coleta"] = datetime.now().strftime("%Y-%m-%d")

print("\nAmostra dos produtos tratados:")
print(df.head())


# === LOAD: GRAVAÇÃO EM SQLITE E EXPORTAÇÃO SQL ===
if TEM_SQLALCHEMY:
    engine = create_engine(f"sqlite:///{CAMINHO_DB}")
    df.to_sql("produtos_raspados", engine, if_exists="replace", index=False)
    conn = engine.raw_connection()
else:
    import sqlite3
    conn = sqlite3.connect(CAMINHO_DB)
    df.to_sql("produtos_raspados", conn, if_exists="replace", index=False)

# Exportando o dump SQL contendo estrutura e dados
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

print(f"\n[OK] Tabela 'produtos_raspados' salva e arquivo '{CAMINHO_SQL.name}' gerado com sucesso!")
