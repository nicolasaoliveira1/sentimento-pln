import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from src.exploracao import exploracao
from src.avaliacao import avaliar, palavras_influentes, casos_dificeis
from src.modelos import modelo_a, comparar_modelos

resultados = Path(__file__).parent / 'resultados'
resultados.mkdir(exist_ok=True)

avaliacoes = pd.read_csv('data/avaliacoes.csv')
exploracao(avaliacoes, resultados)

X = avaliacoes['texto']
y = avaliacoes['sentimento']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Quantidade de avaliações no conjunto de treino: {len(X_treino)}")
print(f"Quantidade de avaliações no conjunto de teste: {len(X_teste)}")

comparar_modelos(X_treino, y_treino, resultados)

avaliar(modelo_a, X_teste, y_teste, resultados)

palavras_influentes(modelo_a, resultados)

dados_dificeis = pd.read_csv('data/casos_dificeis.csv')
casos_dificeis(modelo_a, dados_dificeis, resultados)