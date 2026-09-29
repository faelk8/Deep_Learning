# Experimentos de tradução com T5

[Voltar ao README principal](../../../README.md)

Notebooks de inferência com `T5ForConditionalGeneration` e `T5Tokenizer`. A classe `MultilingualTranslator` monta um prefixo de tarefa, tokeniza a entrada e gera traduções com um checkpoint pré-treinado. O código seleciona CUDA quando disponível, com CPU como alternativa.

| Notebook | Escopo |
| --- | --- |
| [01.01-t5.ipynb](01.01-t5.ipynb) | Exemplos inglês → francês, inglês → alemão e espanhol → inglês |
| [01.02-t5_alternativa.ipynb](01.02-t5_alternativa.ipynb) | Geração de múltiplos candidatos com `t5-base`, scores de sequência e cálculo de BLEU contra uma referência em francês |

## Preparação

Siga o [guia de execução](../../../docs/EXECUCAO.md). Os notebooks usam `torch`, `transformers` e tokenização T5 com `sentencepiece`; o alternativo também importa `sacrebleu`. Execute as células em ordem, com acesso ao modelo externo ou cache preparado.

As versões ainda não estão fixadas. A configuração de geração com grupos de beams precisa ser verificada no ambiente escolhido. Entradas são truncadas em até 512 tokens na implementação alternativa; documentos maiores exigem uma estratégia explícita de segmentação.

## Avaliação e escopo

A lista de idiomas aceita pela classe inclui português, mas isso apenas valida um argumento. Os exemplos não estabelecem qualidade para todos os pares aceitos, e não há treinamento de um tradutor português ↔ inglês nestes notebooks.

Scores de sequência ajudam a ordenar candidatos e não equivalem à probabilidade de uma tradução estar correta. O BLEU calculado sobre um único trecho demonstra o uso da métrica; uma avaliação de qualidade requer um conjunto maior de referências, pares de idiomas definidos e análise de erros.

O primeiro notebook registra a referência [Implementing Multilingual Translation with T5 and Transformers](https://machinelearningmastery.com/implementing-multilingual-translation-with-t5-and-transformers/).
