# Deep Learning · Estudos e experimentos aplicados

Coleção de experimentos em **séries temporais, processamento de linguagem natural, visão computacional e análise de áudio**, com Python, TensorFlow/Keras e PyTorch. O repositório reúne exploração de dados, treinamento e uso de modelos pré-treinados, além de estudos dos mecanismos de atenção e representações contextuais.

Integra o [portfólio](https://faelk8.github.io/rafael_io/), mantido no [rafael_io](https://github.com/faelk8/rafael_io). Aqui estão os notebooks e scripts que permitem examinar as implementações e os limites dos experimentos.

**Estágio atual:** laboratório de estudos, com exemplos didáticos e código legado. Não há ambiente de dependências fixado, execução automatizada ou benchmark consolidado. As saídas salvas nos notebooks são registros de execuções anteriores; não constituem uma avaliação reproduzida nesta revisão.

## Por onde começar

| Interesse | Ponto de entrada | O que observar |
| --- | --- | --- |
| Representações de linguagem | [Vetores contextuais com BERT](PyTorch/vetor/README.md) | Similaridade, mudanças entre camadas, desambiguação e atenção |
| Previsão de séries temporais | [Consumo de energia](0002.01-Energia_Prophet_LSTM_GRU.ipynb) | Exploração de sazonalidade e abordagens Prophet, LSTM e GRU |
| Perguntas e respostas | [Experimentos com DistilBERT](PyTorch/DistilBERT/) | Inferência, comparação com BERT e notebook de ajuste com SQuAD |
| Geração de texto | [Tradução com T5](PyTorch/vetor/tradutor/README.md) | Candidatos de tradução, configuração de geração e exemplo de BLEU |
| Redes neurais clássicas | [Exemplos com Keras](Keras/README.md) | MLP, CNN e LSTM em datasets didáticos |

## Mapa dos experimentos

| Área | Implementação | Escopo e dados |
| --- | --- | --- |
| Áudio | [Análise de voz](0001.00-AnaliseDeSentimentoComVoz_MachineLearning.ipynb) | Extração de atributos com librosa e classificação de rótulos de gênero e emoção de atores; referência RAVDESS |
| Vendas | [Séries temporais de vendas](0002.00-Energia_Prophet_LSTM_GRU.ipynb) | Sample Superstore, agregação temporal, exploração e LSTM; o nome histórico do arquivo menciona energia |
| Energia | [Previsão de consumo](0002.01-Energia_Prophet_LSTM_GRU.ipynb) | Household Power Consumption, decomposição, Prophet e redes recorrentes |
| Visão computacional | [Keras](Keras/) e [fundamentos de PyTorch](PyTorch/base.ipynb) | Exemplos de classificação; em Keras, Iris, MNIST e CIFAR-10 |
| Classificação de texto | [Detecção de sarcasmo](Tensorflow/DetectandoSarcasmo-main/Detectando-Sarcasmo.ipynb) | Tokenização, sequências e treinamento com manchetes rotuladas |
| Atenção | [Notebooks de atenção](PyTorch/atencao/) | Implementações didáticas com tensores e módulos PyTorch |
| Grafos | [Rede convolucional em grafos](PyTorch/graph/01.01-graph.ipynb) | Exemplo com PyTorch Geometric e visualização com NetworkX |
| NLP com Transformers | [Tokenização](PyTorch/token/), [DistilBERT](PyTorch/DistilBERT/) e [T5](PyTorch/vetor/tradutor/) | Classes de modelos, inferência e experimentos de ajuste/geração |
| Representações contextuais | [Vetores](PyTorch/vetor/) e [palavras-chave e resumo](PyTorch/palavras-chave/) | BERT, similaridade de cosseno e seleção extrativa de conteúdo |

A organização por framework é histórica: alguns notebooks combinam bibliotecas. A pasta não define um ambiente isolado de execução.

## Executar e reproduzir

Comece pelo [guia de execução](docs/EXECUCAO.md), que lista dependências por trilha, localização dos dados e impedimentos conhecidos.

```bash
git clone https://github.com/faelk8/Deep_Learning.git
cd Deep_Learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab
python -m jupyter lab
```

No Windows, a ativação no PowerShell é `.venv\Scripts\Activate.ps1`. Esses comandos preparam apenas o Jupyter; instale as bibliotecas do experimento escolhido antes de executar suas células. A compatibilidade entre versões ainda precisa ser validada por trilha.

Para uma entrada de menor escopo, o script [comparando vetores](PyTorch/vetor/01-comparando_vetores.py) usa frases embutidas e um BERT pré-treinado, sem etapa de treinamento. A primeira execução requer download do modelo.

## Leitura técnica dos resultados

O valor dos experimentos está em permitir discutir a escolha do modelo, o tratamento dos dados e o desenho da avaliação. A comparação entre abordagens exige condições equivalentes:

- **Séries temporais:** preservar a ordem temporal, ajustar transformações apenas no treino e comparar modelos no mesmo horizonte com um baseline ingênuo ou sazonal.
- **Classificação:** separar treino, validação e teste; avaliar erros por classe, além da acurácia agregada.
- **Áudio:** avaliar a separação por ator para medir generalização a pessoas ausentes do treino.
- **NLP:** distinguir scores de geração, métricas contra referências e avaliação qualitativa. Um exemplo isolado não comprova qualidade em um idioma ou domínio.

Esses são critérios para a evolução do trabalho. Ainda há implementações que precisam ser adequadas: normalização independente de treino e teste no notebook de energia, uso do teste como validação em exemplos Keras e dependências antigas. A [revisão técnica e o plano de evolução](docs/REVISAO_TECNICA.md) registram evidências e critérios de conclusão.

## Documentação e evolução

- [Guia de execução e dados](docs/EXECUCAO.md)
- [Revisão técnica com prioridades](docs/REVISAO_TECNICA.md)
- [Modelo para documentar um experimento](docs/MODELO_EXPERIMENTO.md)

O próximo marco é tornar um experimento reproduzível de ponta a ponta: ambiente registrado, dados identificados, baseline, avaliação independente e análise de erros. Isso permitirá sustentar os relatos do portfólio com evidências verificáveis.

## Referências e atribuição

Alguns estudos indicam tutoriais e fontes nas próprias células ou scripts. Consulte essas referências para distinguir implementações didáticas de contribuições específicas deste repositório. O [guia de vetores](PyTorch/vetor/README.md) preserva a referência original do estudo de contexto com Transformers.

O repositório ainda não contém um arquivo de licença. As condições de uso dos datasets e dos modelos devem ser consultadas nas respectivas fontes.
