import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessamento import prep
from functools import partial

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
    
X = avaliacoes['texto']
y = avaliacoes['sentimento']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Quantidade de avaliações no conjunto de treino: {len(X_treino)}")
print(f"Quantidade de avaliações no conjunto de teste: {len(X_teste)}")

prep_sem_remocao = partial(prep, stopwords=False)
prep_com_remocao = partial(prep, stopwords=True)

modelo_a =Pipeline([
    ('vetorizador', CountVectorizer(preprocessor=prep_sem_remocao)),
    ('classificador', LogisticRegression(max_iter=1000))])

modelo_b = Pipeline([
    ('vetorizador', TfidfVectorizer(preprocessor=prep_sem_remocao)),
    ('classificador', LogisticRegression(max_iter=1000))])

modelo_C = Pipeline([
    ('vetorizador', CountVectorizer(preprocessor=prep_com_remocao)),
    ('classificador', LogisticRegression(max_iter=1000))])

modelo_D = Pipeline([
    ('vetorizador', TfidfVectorizer(preprocessor=prep_com_remocao)),
    ('classificador', LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results_A = cross_validate(modelo_a, X_treino, y_treino, cv=cv, scoring=['accuracy','f1_macro'])

print(f"Validação cruzada Modelo A (sem remoção de stop words):")
print(f"Acurácia média: {cross_val_results_A['test_accuracy'].mean():.4f}")
print(f"Desvio padrão Acurácia: {cross_val_results_A['test_accuracy'].std():.4f}")
print(f"Média F1 macro: {cross_val_results_A['test_f1_macro'].mean():.4f}")
print(f"Desvio padrão F1 macro: {cross_val_results_A['test_f1_macro'].std():.4f}")
    
cross_val_results_B = cross_validate(modelo_b, X_treino, y_treino, cv=cv, scoring=['accuracy','f1_macro'])

print(f"Validação cruzada Modelo B (sem remoção de stop words):")
print(f"Acurácia média: {cross_val_results_B['test_accuracy'].mean():.4f}")
print(f"Desvio padrão Acurácia: {cross_val_results_B['test_accuracy'].std():.4f}")
print(f"Média F1 macro: {cross_val_results_B['test_f1_macro'].mean():.4f}")
print(f"Desvio padrão F1 macro: {cross_val_results_B['test_f1_macro'].std():.4f}")

cross_val_results_C = cross_validate(modelo_C, X_treino, y_treino, cv=cv, scoring=['accuracy','f1_macro'])
print(f"Validação cruzada Modelo C (com remoção de stop words):")
print(f"Acurácia média: {cross_val_results_C['test_accuracy'].mean():.4f}")
print(f"Desvio padrão Acurácia: {cross_val_results_C['test_accuracy'].std():.4f}")
print(f"Média F1 macro: {cross_val_results_C['test_f1_macro'].mean():.4f}")
print(f"Desvio padrão F1 macro: {cross_val_results_C['test_f1_macro'].std():.4f}")

cross_val_results_D = cross_validate(modelo_D, X_treino, y_treino, cv=cv, scoring=['accuracy','f1_macro'])
print(f"Validação cruzada Modelo D (com remoção de stop words):")
print(f"Acurácia média: {cross_val_results_D['test_accuracy'].mean():.4f}")
print(f"Desvio padrão Acurácia: {cross_val_results_D['test_accuracy'].std():.4f}")
print(f"Média F1 macro: {cross_val_results_D['test_f1_macro'].mean():.4f}")
print(f"Desvio padrão F1 macro: {cross_val_results_D['test_f1_macro'].std():.4f}")

modelo_a.fit(X_treino, y_treino)
vocabulario_a = len(modelo_a.named_steps['vetorizador'].vocabulary_)
print(f"Vocabulário A: {vocabulario_a}")

modelo_b.fit(X_treino, y_treino)
vocabulario_b = len(modelo_b.named_steps['vetorizador'].vocabulary_)
print(f"Vocabulário B: {vocabulario_b}")

modelo_C.fit(X_treino, y_treino)
vocabulario_c = len(modelo_C.named_steps['vetorizador'].vocabulary_)
print(f"Vocabulário C: {vocabulario_c}")

modelo_D.fit(X_treino, y_treino)
vocabulario_d = len(modelo_D.named_steps['vetorizador'].vocabulary_)
print(f"Vocabulário D: {vocabulario_d}")

comparacao = [
    {
        "config": "A",
        "stopwords": "mantidas",
        "vetorizador": "BoW",
        "vocabulario": vocabulario_a,
        "acc_cv_media": cross_val_results_A['test_accuracy'].mean(),
        "acc_cv_desvio": cross_val_results_A['test_accuracy'].std(),
        "f1_cv_media": cross_val_results_A['test_f1_macro'].mean(),
        "f1_cv_desvio": cross_val_results_A['test_f1_macro'].std()
    },
    {
        "config": "B",
        "stopwords": "mantidas",
        "vetorizador": "TF-IDF",
        "vocabulario": vocabulario_b,
        "acc_cv_media": cross_val_results_B['test_accuracy'].mean(),
        "acc_cv_desvio": cross_val_results_B['test_accuracy'].std(),
        "f1_cv_media": cross_val_results_B['test_f1_macro'].mean(),
        "f1_cv_desvio": cross_val_results_B['test_f1_macro'].std()
    },
    {
        "config": "C",
        "stopwords": "removidas",
        "vetorizador": "BoW",
        "vocabulario": vocabulario_c,
        "acc_cv_media": cross_val_results_C['test_accuracy'].mean(),
        "acc_cv_desvio": cross_val_results_C['test_accuracy'].std(),
        "f1_cv_media": cross_val_results_C['test_f1_macro'].mean(),
        "f1_cv_desvio": cross_val_results_C['test_f1_macro'].std()
    },
    {
        "config": "D",
        "stopwords": "removidas",
        "vetorizador": "TF-IDF",
        "vocabulario": vocabulario_d,
        "acc_cv_media": cross_val_results_D['test_accuracy'].mean(),
        "acc_cv_desvio": cross_val_results_D['test_accuracy'].std(),
        "f1_cv_media": cross_val_results_D['test_f1_macro'].mean(),
        "f1_cv_desvio": cross_val_results_D['test_f1_macro'].std()
    }
]

pd.DataFrame(comparacao).to_csv(resultados / 'comparacao.csv', index=False)

previsoes = modelo_a.predict(X_teste)
matriz_confusao = confusion_matrix(y_teste, previsoes, labels=["positivo", "negativo"])
print(f"Matriz de confusão Modelo A:\n{matriz_confusao}")

pd.DataFrame(matriz_confusao, index=["positivo", "negativo"], columns=["positivo", "negativo"]).to_csv(resultados / 'matriz_confusao.csv', index=True)

classificacao = classification_report(y_teste, previsoes, labels=["positivo", "negativo"])
with open(resultados / 'relatorio_classificacao.txt', 'w', encoding='utf-8') as f:
    f.write(classificacao)

mascara_erros = y_teste != previsoes
print("Quantidade de erros:", mascara_erros.sum())
print("Ordem das classes:", modelo_a.classes_)


X_teste_convertido = X_teste.to_numpy()
y_teste_convertido = y_teste.to_numpy()

tabela_teste = pd.DataFrame({
    "texto": X_teste_convertido,
    "real": y_teste_convertido,
    "previsto": previsoes,
    "probabilidade": modelo_a.predict_proba(X_teste).max(axis=1)
})

erros = tabela_teste[mascara_erros.to_numpy()]
erros.to_csv(resultados / 'erros.csv', index=False)

nomes_palavras = modelo_a.named_steps['vetorizador'].get_feature_names_out()