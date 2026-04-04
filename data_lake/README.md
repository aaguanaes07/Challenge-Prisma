# PRISMA Data Lake Simulado

Estrutura local criada para simular um bucket AWS S3 com arquitetura em camadas para ingestao, curadoria, modelagem e monitoramento do PRISMA.

## Estrutura

- `raw/boletos`
  Armazena a base original de boletos recebida no challenge, sem tratamento.

- `raw/auxiliar`
  Armazena a base auxiliar original de enriquecimento, sem tratamento.

- `raw/documentacao`
  Armazena documentos de apoio, como o dicionario de dados.

- `raw/receita_federal`
  Destino sugerido para consultas cadastrais de CNPJ e dados publicos de empresa.

- `raw/bureaus_credito`
  Destino sugerido para eventos de protesto, negativacao, score e consultas recentes.

- `raw/juridico_compliance`
  Destino sugerido para recuperacao judicial, falencia, CEIS e processos relevantes.

- `raw/macroeconomia`
  Destino sugerido para snapshots de Selic, IPCA e indicadores de cenario.

- `trusted/base_modelagem`
  Destino previsto para a base consolidada e padronizada, equivalente a uma camada silver.

- `trusted/features`
  Destino previsto para datasets derivados e atributos prontos para modelagem, equivalente a uma camada gold.

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
  Replica a chegada dos dados brutos internos e externos no lake.

- `trusted`
  Replica as camadas curadas e analiticas prontas para consumo do modelo e da operacao.

- `artifacts`
  Centraliza saidas do pipeline de ciencia de dados e governanca do modelo.

## Evolucao recomendada

Leitura por camadas:

- `raw`
  Dados brutos por fonte, preservando payload original e trilha de auditoria.

- `trusted/base_modelagem`
  Entidades consolidadas de titulo, sacado, cedente e eventos externos.

- `trusted/features`
  Feature store para treino, scoring online e monitoramento.

- `artifacts`
  Modelos, metricas, relatorios, scorecards e evidencias de validacao.

Documento de referencia:

- `docs/arquitetura_dados_modelo.md`
  Proposta detalhada de arquitetura de dados por camadas e novas features para o modelo PRISMA.
