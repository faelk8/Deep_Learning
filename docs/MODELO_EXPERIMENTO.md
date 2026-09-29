# Modelo de documentação de experimento

[Voltar ao README](../README.md)

Copie esta estrutura ao documentar um estudo. Preencha apenas informações verificadas e marque campos ainda não medidos como pendentes.

## Problema e escopo

- Pergunta que o experimento busca responder e uso pretendido.
- Entrada, saída e limites de aplicação.
- Papel do autor: referência utilizada, implementação e adaptações próprias.

## Dados e protocolo

- Origem, licença, versão, tamanho e unidade de observação.
- Variável alvo, tratamento de ausências e transformações.
- Divisão de treino/validação/teste e prevenção de vazamento.
- Baseline e justificativa das métricas.

## Decisões técnicas

| Decisão | Alternativa considerada | Justificativa | Limitação ou custo |
| --- | --- | --- | --- |
| Preencher | Preencher | Evidência ou hipótese a testar | Preencher |

## Reprodução

- Commit, arquivo de entrada e diretório de execução.
- Python, dependências, sistema e CPU/GPU.
- Comandos de preparação e execução.
- Seeds, hiperparâmetros, versão dos dados e revisão do checkpoint.
- Tempo, memória e artefatos produzidos, quando medidos.

## Resultados

| Modelo / baseline | Partição e período | Métrica e unidade | Resultado | Evidência |
| --- | --- | --- | --- | --- |
| Preencher | Preencher | Preencher | Pendente de medição | Link para execução ou artefato |

Descreva erros representativos, variação entre execuções e condições em que o resultado não se sustenta. Diferencie observações medidas de hipóteses.

## Conclusão e próximos passos

Registre o que os dados permitem concluir, a decisão decorrente do experimento e o próximo teste necessário. Só associe impacto de negócio a uma medição que o sustente.
