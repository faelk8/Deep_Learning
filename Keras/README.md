# Redes neurais com Keras

[Voltar ao README principal](../README.md)

Exemplos didáticos de classificação e previsão temporal. Os scripts preservam APIs e decisões de estudos anteriores; precisam de revisão antes de servir como referência de execução ou avaliação.

| Arquivo | Conteúdo |
| --- | --- |
| [001-CIFAR10.py](001-CIFAR10.py) | CNN com `tensorflow.keras`, normalização e curvas de acurácia |
| [002-CIFAR-Keras.py](002-CIFAR-Keras.py) | Rede densa com CIFAR-10 e gravação de modelo/pesos |
| [003-MLP_Iris.py](003-MLP_Iris.py) | MLP para classificação de Iris; inclui ajuste de regressão linear |
| [004-LogisticRegression.py](004-LogisticRegression.py) | Rede densa em Iris; o trecho de regressão logística está comentado |
| [005-MNIST-Regression.py](005-MNIST-Regression.py) | Classificação de dígitos por rede densa, apesar do nome histórico |
| [006-CNN-MNIST.py](006-CNN-MNIST.py) | Exemplo de CNN para MNIST, com correções pendentes |
| [007-CIFAR-10.py](007-CIFAR-10.py) | CNN com aumento de dados e persistência do modelo |
| [008-LSTM-TimeSeries.py](008-LSTM-TimeSeries.py) | LSTM para uma série lida de `sp500.csv`, arquivo não incluído |

## Execução e limitações

Consulte o [guia de ambientes e dados](../docs/EXECUCAO.md). Os scripts executam treinamento diretamente, alguns por 100 a 1.000 épocas; revise a configuração antes de iniciá-los.

Há chamadas legadas, uso do teste como validação e problemas específicos de código. Em particular, `004` sobrescreve os rótulos de treino, `006` contém o argumento `matrices` em `compile`, e `008` calcula RMSE com um slice vazio. A [revisão técnica](../docs/REVISAO_TECNICA.md) descreve as correções prioritárias.

A comparação entre redes densas e convolucionais deve usar o mesmo protocolo de avaliação e um teste reservado. As saídas desses scripts ainda não formam um benchmark comparável.
