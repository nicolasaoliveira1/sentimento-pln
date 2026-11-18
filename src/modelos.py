import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from src.preprocessamento import prep
from functools import partial

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


def comparar_modelos(X_treino, y_treino, resultados):
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
