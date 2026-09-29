# Guia de execução e dados

[Voltar ao README](../README.md)

## Estado de reprodução

O repositório reúne experimentos de épocas diferentes, sem `requirements.txt`, lockfile ou matriz de versões validada. Os pacotes abaixo foram identificados nos imports; a lista orienta a preparação, mas não representa um ambiente testado. Use ambientes separados para as trilhas Keras legada e Transformers.

## Primeiro experimento: vetores com BERT

Na raiz do repositório, com o ambiente virtual ativado:

```bash
python -m pip install torch transformers numpy
python PyTorch/vetor/01-comparando_vetores.py
```

O script baixa `bert-base-uncased`, extrai representações da palavra `bank` em duas frases e imprime a similaridade de cosseno. Não há valor numérico de referência validado nesta revisão. Os demais scripts de visualização também usam `matplotlib` e, no mapa de atenção, `seaborn`.

Esse fluxo foi identificado por leitura do código; o download e a inferência não foram executados na revisão documental. Após uma execução bem-sucedida, registre versões, revisão do modelo e saída observada.

## Dependências por trilha

| Trilha | Pacotes identificados / necessários à leitura dos dados | Observações |
| --- | --- | --- |
| Keras e sarcasmo | `tensorflow`, `keras`, `numpy`, `matplotlib`, `scikit-learn`; `pandas` no exemplo temporal | Há imports e chamadas legadas; instalar versões atuais não basta para garantir execução |
| Vendas e energia | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `keras`; `xlrd` para `.xls` | Notebooks usam `pandas_profiling` e/ou `fbprophet`; precisam de validação ou migração |
| Áudio | `librosa`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib` | Requer os áudios externos |
| PyTorch básico e atenção | `torch`, `torchvision` no notebook básico | Os exemplos variam entre treino e operações com tensores |
| Grafos | `torch`, `torch-geometric`, `networkx`, `matplotlib` | A combinação de binários deve ser compatível com o ambiente escolhido |
| BERT, DistilBERT e tokenização | `torch`, `transformers`, `numpy`; `datasets` no ajuste | Treinamento com `Trainer` pode exigir dependências adicionais, como `accelerate`; tokenização também contém exemplo TensorFlow |
| Tradução | `torch`, `transformers`, `sentencepiece`, `sacrebleu` | O segundo notebook usa BLEU; parâmetros de geração precisam ser compatíveis com a versão escolhida |

Não há requisito de hardware medido. Tradução seleciona CUDA quando disponível; outros exemplos podem executar em CPU. Treinamento e downloads têm custos de tempo, memória e disco que ainda não foram registrados.

## Dados e diretórios de execução

| Experimento | Disponibilidade | Preparação |
| --- | --- | --- |
| Vendas | [Data/SampleSuperstore.xls](../Data/SampleSuperstore.xls) versionado | O notebook lê `SampleSuperstore.xls` sem `Data/`; ajustar a célula para `Data/SampleSuperstore.xls` ao executar na raiz |
| Energia | `household_power_consumption.txt` ausente | Obter a base indicada no notebook em [Kaggle](https://www.kaggle.com/uciml/electric-power-consumption-data-set) e ajustar o caminho |
| Áudio | Diretório `dados/` ausente | Consultar a referência [RAVDESS no Zenodo](https://zenodo.org/record/1188976); o código espera `dados/Actor_01/*.wav` até `Actor_24` |
| Sarcasmo | [JSON de manchetes](../Tensorflow/DetectandoSarcasmo-main/dataset/Sarcasm_Headlines_Dataset.json) versionado | Executar com diretório de trabalho `Tensorflow/DetectandoSarcasmo-main`, pois a célula usa `dataset/...` |
| LSTM em Keras | `sp500.csv` ausente | Identificar a origem e o esquema da série antes de tentar reproduzir o script |
| Iris, MNIST e CIFAR-10 | Carregadores das bibliotecas | Iris é carregado pelo scikit-learn; MNIST e CIFAR-10 podem exigir download |
| Transformers | Modelos externos; SQuAD no notebook de ajuste | Requer acesso à fonte na primeira execução ou cache previamente preenchido |

A procedência e as condições de redistribuição dos arquivos locais ainda precisam ser documentadas. Não trate arquivos com o mesmo nome como versões equivalentes de um dataset.

## Registro de uma execução

1. Escolha um único experimento e confira seus imports, caminhos e limitações.
2. Registre versão do Python, bibliotecas, sistema operacional e CPU/GPU.
3. Identifique a versão dos dados e o checkpoint/revisão dos modelos externos.
4. Execute o notebook desde um kernel reiniciado, em ordem, e registre qualquer adaptação.
5. Salve configuração, métricas e limitações usando o [modelo de experimento](MODELO_EXPERIMENTO.md).

A existência de gráficos e saídas salvas não substitui essa execução completa.
