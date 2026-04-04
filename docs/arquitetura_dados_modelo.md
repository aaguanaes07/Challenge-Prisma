# Arquitetura de Dados e Novas Features do Modelo PRISMA

## Objetivo

Evoluir o motor atual de risco do PRISMA, hoje fortemente concentrado em comportamento interno de boletos e liquidez, para uma arquitetura em camadas que una:

- comportamento interno do titulo;
- historico do sacado e do cedente;
- sinais externos cadastrais, juridicos e de credito;
- saida operacional para decisao, monitoramento e re-treinamento.

## Leitura do estado atual

O modelo atual usa principalmente atributos internos definidos em [prisma_core.py](/C:/Users/aagua/Downloads/PRISMA%20-%20Challenge%20Fiap/src/prisma_core.py), com foco em:

- valor nominal;
- atraso medio;
- scores de materialidade e quantidade;
- indicadores de liquidez;
- UF e tipo de especie.

Essa base funciona bem para captar deterioracao historica, mas ainda enxerga pouco do ambiente externo do CNPJ.

## Arquitetura em camadas

### 1. Landing / Raw

Camada de chegada dos dados sem transformacao estrutural relevante.

Objetivo:
- preservar o dado original;
- permitir reprocessamento;
- manter trilha de auditoria.

Fontes sugeridas:
- `raw/boletos`: recebimento de titulos e eventos de pagamento;
- `raw/auxiliar`: base auxiliar interna usada hoje;
- `raw/receita_federal`: JSON ou CSV de consultas de CNPJ;
- `raw/bureaus_credito`: protestos, negativacoes, consultas recentes e eventos cadastrais;
- `raw/juridico_compliance`: RJ, falencia, processos, CEIS e certidoes;
- `raw/macroeconomia`: Selic, IPCA e curvas de referencia;
- `raw/documentacao`: dicionarios, contratos de API e layouts.

Chaves esperadas:
- `cnpj_consultado`
- `data_consulta`
- `fonte`
- `payload_bruto`
- `request_id` ou `id_consulta`

### 2. Bronze / Padronizacao

Camada de normalizacao minima, ainda proxima da origem, mas com schema controlado.

Objetivo:
- padronizar nomes de colunas;
- corrigir tipos;
- deduplicar;
- carimbar data de ingestao e versao da fonte.

Tabelas sugeridas:
- `bronze_boleto_evento`
- `bronze_cnpj_receita`
- `bronze_bureau_evento_credito`
- `bronze_juridico_evento`
- `bronze_macro_snapshot`

Transformacoes tipicas:
- padronizacao de CNPJ;
- parse de datas;
- parse de valores monetarios;
- flatten de JSON das APIs;
- remocao de duplicidades por `cnpj + data_evento + fonte`.

### 3. Silver / Entidades Curadas

Camada analitica por entidade de negocio.

Objetivo:
- consolidar a "verdade operacional" por boleto, sacado, cedente e tempo;
- preparar joins estaveis;
- separar historico observavel de sinais derivados.

Entidades sugeridas:
- `silver_titulo`
- `silver_sacado`
- `silver_cedente`
- `silver_cnpj_externo`
- `silver_calendario_macro`
- `silver_evento_juridico`

Exemplos de colunas:
- `id_boleto`
- `id_pagador`
- `id_beneficiario`
- `dt_emissao`
- `dt_vencimento`
- `dt_pagamento`
- `dias_atraso_real`
- `status_cadastral`
- `capital_social`
- `idade_empresa_dias`
- `qtd_protestos_12m`
- `flag_recuperacao_judicial`
- `qtd_consultas_credito_30d`
- `selic_aa`
- `ipca_12m`

### 4. Gold / Feature Store

Camada de atributos prontos para treinamento, scoring online e explicabilidade.

Objetivo:
- concentrar features por observacao;
- garantir consistencia entre treino e producao;
- permitir versionamento de features.

Datasets sugeridos:
- `gold_training_titulos`
- `gold_scoring_titulos`
- `gold_monitoramento_carteira`

Boas praticas:
- cada linha deve representar um titulo em uma data de decisao;
- features devem usar apenas informacao disponivel ate aquela data;
- guardar `feature_timestamp`, `snapshot_date` e `model_version`.

### 5. Artifacts / Decisioning / Monitoring

Camada final de consumo operacional e governanca.

Objetivo:
- servir score para API e dashboard;
- registrar decisoes;
- acompanhar drift, default e fraude.

Saidas sugeridas:
- `artifacts/models`
- `artifacts/reports`
- `trusted/carteira_scoreada`
- `trusted/monitoramento`
- `trusted/decisoes_credito`

Indicadores recomendados:
- taxa de aprovacao;
- inadimplencia observada;
- perda esperada;
- PSI por feature critica;
- lift por fonte externa;
- fraude evitada por regra de bloqueio.

## Fluxo de dados recomendado

1. Ingerir dados internos de boletos e base auxiliar.
2. Enriquecer CNPJs de sacado e cedente com Receita Federal.
3. Consultar sinais de bureau para janelas recentes.
4. Anexar eventos juridicos e de compliance.
5. Anexar snapshot macroeconomico da data da operacao.
6. Gerar entidades curadas por titulo, sacado e cedente.
7. Materializar features para treino e para scoring.
8. Aplicar regras de bloqueio antes do modelo.
9. Rodar score supervisionado.
10. Publicar decisao, motivos, watchlist e monitoramento.

## Regras antes do modelo

Antes do score probabilistico, vale ter uma camada de bloqueio deterministica.

Bloqueios imediatos sugeridos:
- CNPJ `baixado`, `inapto` ou `suspenso`;
- flag ativa de recuperacao judicial;
- evidencia forte de fraude cadastral;
- incompatibilidade extrema entre capital social e volume operado;
- empresa recem-aberta acima do apetite de risco definido;
- presenca em cadastro de inidoneidade ou restricao critica.

Essas regras blindam o fundo em casos nos quais o modelo nao deveria "negociar" risco.

## Novas features para o modelo

### Grupo A. Receita Federal

Melhor uso:
- antifraude cadastral;
- robustez estrutural minima;
- coerencia entre porte e operacao.

Features recomendadas:
- `idade_empresa_dias`
- `idade_empresa_faixa`
- `capital_social`
- `log_capital_social`
- `capital_social_por_valor_titulo`
- `flag_capital_social_incompativel`
- `porte_empresa`
- `natureza_juridica`
- `cnae_principal`
- `flag_cnae_sensivel`
- `qtd_cnaes_secundarios`
- `status_cadastral`
- `flag_status_cadastral_critico`
- `uf_receita`
- `divergencia_uf_operacao_vs_receita`
- `qtd_socios`
- `flag_mei`

### Grupo B. Bureaus de Credito

Melhor uso:
- antecipacao de inadimplencia;
- deteccao de estresse de liquidez;
- identificacao de comportamento oportunista.

Features recomendadas:
- `qtd_protestos_3m`
- `qtd_protestos_12m`
- `vl_total_protestos_12m`
- `dias_desde_ultimo_protesto`
- `flag_cheque_sem_fundo`
- `qtd_ocorrencias_negativas_6m`
- `qtd_consultas_credito_7d`
- `qtd_consultas_credito_30d`
- `intensidade_busca_credito_recente`
- `flag_spike_consulta_credito`
- `rating_bureau`
- `score_bureau_normalizado`
- `flag_renegociacao_recente`
- `qtd_instituicoes_consultantes_30d`

### Grupo C. Juridico e Compliance

Melhor uso:
- eventos fatais;
- degradacao reputacional;
- pressao de caixa.

Features recomendadas:
- `flag_recuperacao_judicial`
- `dias_desde_evento_rj`
- `flag_falencia`
- `qtd_processos_trabalhistas_12m`
- `qtd_processos_fiscais_12m`
- `qtd_execucoes_12m`
- `valor_causa_total_12m`
- `flag_ceis`
- `flag_ceaf`
- `qtd_eventos_juridicos_criticos_12m`

### Grupo D. Macroeconomia e Setor

Melhor uso:
- risco de cenario;
- ajuste de apetite por setor e momento do ciclo.

Features recomendadas:
- `selic_aa_data_operacao`
- `ipca_12m_data_operacao`
- `variacao_selic_3m`
- `variacao_ipca_3m`
- `cnae_x_selic_sensibilidade`
- `risco_setorial_cnae`
- `inadimplencia_media_setor_3m`
- `inadimplencia_media_setor_12m`
- `ticket_vs_media_setorial`

### Grupo E. Features de interacao

Essas costumam gerar mais ganho do que variaveis externas puras.

Features recomendadas:
- `atraso_sacado_x_qtd_protestos_12m`
- `liquidez_sacado_x_consultas_credito_30d`
- `volume_titulo_x_capital_social`
- `idade_empresa_x_materialidade_sacado`
- `flag_rj_x_valor_titulo`
- `setor_x_macro_stress`
- `score_interno_x_score_bureau`
- `share_inad_curta_x_protesto_recente`

## Priorizacao de implementacao

### Fase 1. MVP forte

Objetivo:
- elevar protecao antifraude e criar ganho rapido de negocio.

Entradas:
- Receita Federal;
- Bureaus basicos;
- regras de bloqueio.

Features prioritarias:
- `idade_empresa_dias`
- `capital_social_por_valor_titulo`
- `flag_status_cadastral_critico`
- `qtd_protestos_12m`
- `dias_desde_ultimo_protesto`
- `qtd_consultas_credito_30d`
- `flag_spike_consulta_credito`
- `score_interno_x_score_bureau`

### Fase 2. Blindagem juridica

Entradas:
- recuperacao judicial;
- falencia;
- CEIS;
- processos criticos.

Features prioritarias:
- `flag_recuperacao_judicial`
- `dias_desde_evento_rj`
- `flag_ceis`
- `qtd_execucoes_12m`

### Fase 3. Contexto de cenario

Entradas:
- Selic;
- IPCA;
- risco setorial.

Features prioritarias:
- `selic_aa_data_operacao`
- `variacao_selic_3m`
- `risco_setorial_cnae`
- `setor_x_macro_stress`

## Mapeamento para o modelo atual

Hoje o projeto usa as features numericas e categoricas abaixo em [prisma_core.py](/C:/Users/aagua/Downloads/PRISMA%20-%20Challenge%20Fiap/src/prisma_core.py):

- `vlr_nominal`
- `media_atraso_dias_sacado`
- `score_materialidade_v2_sacado`
- `score_quantidade_v2_sacado`
- `sacado_indice_liquidez_1m_sacado`
- `share_vl_inad_pag_bol_6_a_15d_sacado`
- `media_atraso_dias_cedente`
- `score_materialidade_v2_cedente`
- `score_quantidade_v2_cedente`
- `cedente_indice_liquidez_1m_cedente`
- `indicador_liquidez_quantitativo_3m_cedente`
- `tipo_especie`
- `uf_sacado`
- `uf_cedente`

Proposta de expansao imediata do schema:

- `idade_empresa_dias_sacado`
- `capital_social_sacado`
- `status_cadastral_sacado`
- `qtd_protestos_12m_sacado`
- `dias_desde_ultimo_protesto_sacado`
- `qtd_consultas_credito_30d_sacado`
- `score_bureau_sacado`
- `flag_recuperacao_judicial_sacado`
- `idade_empresa_dias_cedente`
- `capital_social_cedente`
- `status_cadastral_cedente`
- `qtd_protestos_12m_cedente`
- `qtd_consultas_credito_30d_cedente`
- `score_bureau_cedente`
- `selic_aa_data_operacao`
- `ipca_12m_data_operacao`

## Desenho minimo de tabelas

### `silver_cnpj_externo`

Chave:
- `cnpj`
- `data_referencia`

Colunas:
- `status_cadastral`
- `capital_social`
- `porte_empresa`
- `natureza_juridica`
- `cnae_principal`
- `idade_empresa_dias`
- `qtd_socios`
- `fonte_receita`

### `silver_evento_credito`

Chave:
- `cnpj`
- `data_referencia`

Colunas:
- `qtd_protestos_3m`
- `qtd_protestos_12m`
- `vl_total_protestos_12m`
- `dias_desde_ultimo_protesto`
- `qtd_consultas_credito_7d`
- `qtd_consultas_credito_30d`
- `score_bureau_normalizado`

### `silver_evento_juridico`

Chave:
- `cnpj`
- `data_referencia`

Colunas:
- `flag_recuperacao_judicial`
- `flag_falencia`
- `qtd_processos_trabalhistas_12m`
- `qtd_processos_fiscais_12m`
- `qtd_execucoes_12m`
- `flag_ceis`

### `gold_training_titulos`

Chave:
- `id_boleto`
- `data_snapshot_modelo`

Colunas:
- features internas do boleto;
- features externas do sacado;
- features externas do cedente;
- features macro;
- label `alvo_inadimplencia`.

## Hipotese de negocio a ser validada

A hipotese central do PRISMA passa a ser:

`A combinacao entre sinais internos de comportamento de pagamento e sinais externos de estresse cadastral, juridico e de credito aumenta a capacidade do modelo de antecipar fraude e inadimplencia, reduzindo aprovacoes indevidas sem derrubar excessivamente a taxa de conversao.`

## Metricas de sucesso

- aumento de `roc_auc` e `pr_auc`;
- melhora de `recall` em inadimplencia severa;
- reducao de falsos negativos em empresas com eventos externos criticos;
- lift de aprovacao segura apos camada de bloqueio;
- rastreabilidade de `motivos_risco`.

## Proximo passo tecnico sugerido

1. Criar datasets `silver_cnpj_externo`, `silver_evento_credito` e `silver_evento_juridico`.
2. Expandir `FEATURE_COLUMNS` do motor atual.
3. Separar regras de bloqueio da etapa probabilistica.
4. Reentreinar e comparar baseline vs modelo enriquecido.
5. Publicar novos motivos de risco na API e no dashboard.
