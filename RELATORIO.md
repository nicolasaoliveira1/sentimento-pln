# RELATORIO — Análise de Sentimento com PLN

**Nome:Nicolas de Almeida Oliveira** 
**Repositório:https://github.com/nicolasaoliveira1/sentimento-pln** 
**Data:18/11/2026** 

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420 | 0,8564 ± 0,0926 | 0,8563 ± 0,0926 |
| B | mantidas | TF-IDF | 420 | 0,8462 ± 0,0843 | 0,8459 ± 0,0844 |
| C | removidas | BoW | 408 | 0,8513 ± 0,0894 | 0,8512 ± 0,0894 |
| D | removidas | TF-IDF | 408 | 0,8462 ± 0,0725 | 0,8458 ± 0,0727 |

**Configuração escolhida:** A **Acurácia no teste:** 0,8154 **F1 no teste:** 0,8150 (macro)

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*

A função prep faz três coisas, transforma em minúsculas, tira os acentos, pontuação e números. A pequena lista de stopwords foi adaptada de uma das TDEs anteriores. Cai o vocabulario de 420 pra 408.
O "Não" foi mantido, já que carrega negacao das frases (se prova sendo o coeficiente mais negativo do modelo escolhido). Remover stopwords nao mudou tanto o vocabulario nem resultado.

## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*

A diferença entre as configs é ~0,01 na acuracia, ~0,07 no desvio padrão. Não sao muito relevantes. Escolhi a A pois tem a maior media, e é a mais simples. Porém é quase um empate.
BoW conta quantas vezes as palavras aparecem, enquanto TF-IDF da menos peso a palavras comuns e mais pra palavras raras e distintas naquele contexto.

## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*

Matriz de confusao tem 28 positivos e 4 negativos como positivo/negativo, e 8 e 25 no contrário.
Tem mais negativos classificados como positivos (8) do que o inverso (4). O erro mais grave aqui é classificar uma reclamação como elogio. Como o dataset é balanceado, a acurácia é menos enganosa (130 cada classe).

## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*

Com 260 textos e 65 no teste, uma divisao depende de que textos cairam em cada lado, validação cruzada usa 5 divisoes e obtemos a média e desvio. Escolher pelo teste faria a nota final ficar otimista. Vazamento de dados seria o vocabulário ou TF-IDF ver os textos do teste, ou escolher a config olhando o teste. pra evitar tem o split antes de tudo (treino e teste), e usei a sugestão da IA ao Pipeline, uma esteira, que junta os passos em uma coisa só, assim o vetorizador aprende o vocabulario com os dados de treino, e nao do teste, nada vaza.

## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*

Positiva, classificou negativa:
    "O sabor é maravilhoso e as porções são generosas": "maravilhoso" e "generosas" não apareceram provavelmente.
Negativa, classificou positivo:
    "Camiseta encolheu e desbotou depois da primeira lavagem": o sentimento tá nas palavras "encolheu", "desbotou", que o modelo não conhece.
    "Cheguei animado, mas me arrependi, tudo estava sujo": mistura "animado" com "arrependi" e "sujo".
Palavras Influentes que nao carregam sentimento:
    Positivas: "bastante" (maior coeficiente 2,34) e "bem".
    Negativas: "mais", "toda" e "dinheiro".
Isso mostra que o modelo aprendeu correlações do dataset pequeno, e nao o sentimento.

## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

Negação: acertou "Não gostei de nada" e "Não é ruim, gostei bastante", mas errou "Não tenho nada a reclamar" e "Sem defeitos" (preveu negativo)
Ironia: "Que ótimo, o produto chegou quebrado" pegou positivo 0,83. As outras duas acertaram, provavelmente por acaso (probabilidade fraca).
Frases mistas: Modelo viu como positiva com alta confiança, não sabe o que é "Misto".
Neutro: "Chegou ontem", foi positiva (0,51), modelo só conhece duas classes.

Esses vetorizadores tratam o texto como um saco de palavras, nao ordena. Algumas tecnicas pra melhorar isso poderiam ser bigramas ("não gostei"), word embeddings (palavras como vetores), ou até modelos transformers, como o BERT da Google.

Esse dataset tem só duas classes, 260 textos, frases curtas. Em produção o modelo erra em ironia, frases neutras/mistas, muito comuns na linguagem natural.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: reorganização do código (separação em módulos), orientação do caminho a seguir em trechos do código e agilização da entrega.
