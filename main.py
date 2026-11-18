import pandas as pd
from pathlib import Path

resultados = Path(__file__).parent / 'resultados'
resultados.mkdir(exist_ok=True)

avaliacoes = pd.read_csv('data/avaliacoes.csv')

quantidade = avaliacoes['sentimento'].value_counts()

palavras = avaliacoes['texto'].str.split().str.len()
avaliacoes['numero_palavras'] = palavras

media_por_classe = avaliacoes.groupby('sentimento')['numero_palavras'].mean()

positivos = avaliacoes[avaliacoes['sentimento']=="positivo"].head(3)
negativos = avaliacoes[avaliacoes['sentimento']=="negativo"].head(3)

with open(resultados / 'exploracao.txt', 'w', encoding='utf-8') as f:
    f.write("Quantidade de avaliações por classe:\n")
    f.write(str(quantidade))
    f.write("\n\nMédia do número de palavras por classe:\n")
    f.write(str(media_por_classe))
    f.write("\n\nTotal de avaliações:\n")
    f.write(str(len(avaliacoes)))
    f.write("\n\nExemplos de avaliações positivas:\n")
    f.write(positivos[['texto', 'numero_palavras']].to_string(index=False))
    f.write("\n\nExemplos de avaliações negativas:\n")
    f.write(negativos[['texto', 'numero_palavras']].to_string(index=False))
