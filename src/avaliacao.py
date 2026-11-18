import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def avaliar(modelo, X_teste, y_teste, resultados):
    previsoes = modelo.predict(X_teste)
    matriz_confusao = confusion_matrix(y_teste, previsoes, labels=["positivo", "negativo"])
    print(f"Matriz de confusão do Modelo:\n{matriz_confusao}")

    pd.DataFrame(matriz_confusao, index=["positivo", "negativo"], columns=["positivo", "negativo"]).to_csv(resultados / 'matriz_confusao.csv', index=True)

    classificacao = classification_report(y_teste, previsoes, labels=["positivo", "negativo"])
    with open(resultados / 'relatorio_classificacao.txt', 'w', encoding='utf-8') as f:
        f.write(classificacao)

    mascara_erros = y_teste != previsoes
    print("Quantidade de erros:", mascara_erros.sum())
    print("Ordem das classes:", modelo.classes_)


    X_teste_convertido = X_teste.to_numpy()
    y_teste_convertido = y_teste.to_numpy()

    tabela_teste = pd.DataFrame({
        "texto": X_teste_convertido,
        "real": y_teste_convertido,
        "previsto": previsoes,
        "probabilidade": modelo.predict_proba(X_teste).max(axis=1)
    })

    erros = tabela_teste[mascara_erros.to_numpy()]
    erros.to_csv(resultados / 'erros.csv', index=False)


def palavras_influentes(modelo, resultados):
    nomes_palavras = modelo.named_steps['vetorizador'].get_feature_names_out()
    coeficientes = modelo.named_steps['classificador'].coef_[0]
    print(len(nomes_palavras), len(coeficientes))

    tabela_palavras = pd.DataFrame({
        "palavra": nomes_palavras,
        "coeficiente": coeficientes
    })

    palavras_ordenadas = tabela_palavras.sort_values(by="coeficiente", ascending=False)
    print("Top 10 palavras mais associadas à classe positiva:")
    print(palavras_ordenadas.head(10))
    print("\nTop 10 palavras mais associadas à classe negativa:")
    print(palavras_ordenadas.tail(10))

    pd.concat([palavras_ordenadas.head(10), palavras_ordenadas.tail(10)]).to_csv(resultados / 'palavras_influentes.csv', index=False)