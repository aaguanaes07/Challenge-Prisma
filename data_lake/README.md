# PRISMA Data Lake Simulado

Estrutura local criada para simular um bucket AWS S3 com zonas `raw` e `trusted`, conforme a arquitetura proposta do PRISMA.

## Estrutura

- `raw/boletos`
  Armazena a base original de boletos recebida no challenge, sem tratamento.

- `raw/auxiliar`
  Armazena a base auxiliar original de enriquecimento, sem tratamento.

- `raw/documentacao`
  Armazena documentos de apoio, como o dicionario de dados.

- `trusted/base_modelagem`
  Destino previsto para a base consolidada apos merge, padronizacao e validacao.

- `trusted/features`
  Destino previsto para datasets derivados e atributos prontos para modelagem.

- `trusted/carteira_scoreada`
  Destino previsto para exportacoes de scoring e classificacao da carteira.

- `trusted/monitoramento`
  Destino previsto para snapshots operacionais, watchlists e alertas de risco.

- `artifacts/models`
  Armazena os modelos serializados gerados no projeto.

- `artifacts/reports`
  Destino previsto para relatorios de validacao, metricas e evidencias da sprint.

## Leitura Arquitetural

- `raw`
  Replica a chegada dos dados brutos no lake.

- `trusted`
  Replica a camada curada pronta para consumo analitico e operacional.

- `artifacts`
  Centraliza saidas do pipeline de ciencia de dados e governanca do modelo.
