# Análise de sentimento em avaliações em português

Classifica avaliações em português como positivas ou negativas e compara quatro configurações (Regressão Logística):

| Config | Stop words | Vetorizador |
|--------|-----------|-------------|
| A | mantidas  | BoW |
| B | mantidas  | TF-IDF |
| C | removidas | BoW |
| D | removidas | TF-IDF |

## Requisito

Python 3.10 ou superior.

## Como executar (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Resultados

Os arquivos gerados (métricas, matriz de confusão, erros, palavras influentes etc.) ficam em `resultados/`.

## Configuração escolhida

A **configuração A** (stop words mantidas + BoW), escolhida pela **validação cruzada** no conjunto de treino. Ela teve a maior acurácia e F1 macro médios (≈ 0,856). Os valores estão em `resultados/comparacao.csv`.
