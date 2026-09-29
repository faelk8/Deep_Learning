# Representações contextuais com BERT

[Voltar ao README principal](../../README.md)

Quatro scripts exploram como `bert-base-uncased` representa tokens em função do contexto. Usam inferência com `model.eval()` e `torch.no_grad()`, sem ajuste dos pesos.

| Script | Pergunta explorada | Saída |
| --- | --- | --- |
| [01-comparando_vetores.py](01-comparando_vetores.py) | Como muda o vetor de `bank` entre banco e margem de rio? | Tokens e similaridade de cosseno |
| [02-visualizando_contexto.py](02-visualizando_contexto.py) | Como a representação de `fox` muda entre camadas? | Gráfico de similaridade entre camadas consecutivas |
| [03-desambiguacao.py](03-desambiguacao.py) | Um agrupamento por similaridade separa contextos de `bank`? | Grupos com limiar heurístico de 0,60 |
| [04-atencao.py](04-atencao.py) | Como visualizar pesos de atenção de uma camada/cabeça? | Mapas de calor por cabeça e média entre cabeças |

## Executar

Na raiz do repositório, com ambiente virtual ativo:

```bash
python -m pip install torch transformers numpy matplotlib seaborn
python PyTorch/vetor/01-comparando_vetores.py
```

Troque o nome do script para explorar os demais. O primeiro uso requer download do modelo; os gráficos precisam de um ambiente que permita exibição. Não há combinação de versões validada ou saídas numéricas reproduzidas nesta revisão.

## Limites de interpretação

As frases estão embutidas nos scripts. O agrupamento usa um limiar fixo, sem avaliação com sentidos rotulados. As buscas por palavras usam igualdade entre tokens e texto, o que exige revisão para palavras divididas em subpalavras. Os mapas mostram pesos internos de atenção; isoladamente, não demonstram qualidade em uma tarefa final.

## Referência do estudo

O código referencia [Generating and Visualizing Context Vectors in Transformers](https://machinelearningmastery.com/generating-and-visualizing-context-vectors-in-transformers/). Os exemplos devem ser apresentados como estudos apoiados nessa referência, com adaptações próprias identificadas quando documentadas.
