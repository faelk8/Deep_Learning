# Revisão técnica e plano de evolução

[Voltar ao README](../README.md)

Revisão estática de scripts, células de notebooks e documentação. Não foram executados treinamentos, downloads de modelos ou benchmarks. As prioridades abaixo são propostas de trabalho, não funcionalidades já implementadas.

## Prioridade 1 — validade da avaliação e execução

| Evidência no código | Consequência | Ação e critério de conclusão |
| --- | --- | --- |
| `0002.01-Energia_Prophet_LSTM_GRU.ipynb` cria um novo `MinMaxScaler` e faz `fit` no teste | Usa estatísticas do teste e coloca treino/teste em escalas distintas | Ajustar apenas no treino, reutilizar a transformação e conferir a inversão de escala antes de publicar métricas |
| `Keras/008-LSTM-TimeSeries.py` normaliza a série antes da divisão; usa `trainPredic[:0]` no RMSE | Pré-processamento acessa o período de teste e a métrica recebe um vetor vazio | Separar primeiro, corrigir shapes e calcular RMSE com pares alinhados; rever a loss logarítmica para valores padronizados |
| `Keras/004-LogisticRegression.py` sobrescreve `y_one` do treino com rótulos do teste | O treinamento recebe rótulos que não correspondem às amostras | Manter rótulos separados e verificar correspondência entre amostras e classes |
| `Keras/006-CNN-MNIST.py` usa `matrices` em `compile`, layout `(N, 1, 28, 28)` sem configuração explícita e normalização sem os parênteses esperados | Há bloqueios de execução e inconsistências no processamento | Corrigir argumento, layout, normalização e adequação da loss à classificação; executar um lote completo |
| Exemplos Keras e notebook de sarcasmo passam o conjunto de teste como `validation_data` | Se usado para escolher configurações, o teste deixa de ser uma avaliação independente | Criar validação separada e reservar o teste para a configuração final |
| Áudio divide amostras aleatoriamente, sem agrupamento explícito por ator | Pode medir desempenho em vozes já vistas, sem demonstrar generalização a novos atores | Separar por ator e verificar ausência de sobreposição; publicar métricas por classe |

## Prioridade 2 — reprodução

- **Ambientes por trilha:** registrar versões após execução real. Há APIs legadas como `keras.layers.core`, `np_utils`, `predict_classes`, `fit_generator`, `fbprophet` e `statsmodels.tsa.arima_model`; definir migração ou preservação do ambiente para cada estudo.
- **Dados:** corrigir o caminho do Superstore; documentar aquisição de energia e áudio; identificar `sp500.csv`. Critério: uma pessoa consegue preparar os dados a partir do guia, com esquema e versão verificáveis.
- **Aleatoriedade:** registrar seeds das bibliotecas usadas e condições de execução. No resumo extrativo, explicitar o modo de avaliação do modelo, como já ocorre nos scripts de vetores.
- **Verificação automatizada:** após estabilizar um experimento, adicionar uma execução curta que confira formatos, valores finitos e ausência de sobreposição entre partições. Evitar treinar todos os modelos a cada alteração documental.

## Prioridade 3 — evidências para o portfólio

Escolher um caso principal, como previsão de consumo, e documentá-lo de ponta a ponta antes de ampliar a coleção:

1. Formular a pergunta, a variável prevista, a granularidade e o horizonte. Diferenciar a motivação de uma empresa de energia da abrangência da base doméstica utilizada.
2. Definir baseline e protocolo temporal. Comparar Prophet, LSTM e GRU nos mesmos períodos e na mesma escala.
3. Publicar MAE/RMSE, custo de execução e análise dos períodos com maior erro, acompanhados da configuração que produziu os resultados.
4. Explicar quando a complexidade adicional se justifica e quais limitações impedem extrapolar o resultado.

Para NLP, os scores de geração do T5 não são probabilidades calibradas de tradução correta. O BLEU de um único trecho no notebook alternativo é uma demonstração da métrica. Expandir o conjunto de avaliação e registrar pares de idiomas antes de afirmar qualidade de tradução. Similaridade de vetores e mapas de atenção também precisam de interpretação delimitada ao experimento.

## Relação com o rafael_io

A cópia local consultada do portfólio organiza relatos em contexto, contribuição, arquitetura e resultados. Este repositório pode fornecer os links para código e evidências desses relatos. Descrição sugerida, compatível com o conteúdo atual:

> Laboratório de aprendizado de máquina e deep learning com estudos de séries temporais, NLP, visão computacional e áudio. Reúne experimentos com TensorFlow/Keras, PyTorch e Transformers, acompanhados de documentação de execução, limitações metodológicas e um plano de evolução para avaliações reproduzíveis.

Na apresentação de cada caso, distinguir material de estudo baseado em referências, adaptações realizadas e resultados efetivamente medidos. A revisão documental não altera o site nem comprova ganhos de negócio, desempenho em produção ou superioridade de modelos.

## Melhorias documentais realizadas

- README principal com escopo, navegação e vínculo com o portfólio.
- Guia de execução com dados disponíveis, dependências e impedimentos.
- READMEs de Keras, vetores e tradução alinhados às implementações.
- Modelo de relato técnico para futuras execuções.

As correções de código e avaliações descritas acima permanecem pendentes.
