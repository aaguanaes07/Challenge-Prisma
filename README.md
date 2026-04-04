# PRISMA | Plataforma de Risco para FIDC

Projeto de simulacao de risco estrutural para FIDCs com:

- pipeline local de treino e scoring;
- dashboard Streamlit;
- API FastAPI;
- proposta de enriquecimento externo com Receita Federal e bureau de credito.

## Como rodar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Suba a API:

```bash
uvicorn src.api:app --reload
```

Suba o dashboard:

```bash
streamlit run src/app.py
```

## Endpoints principais

- `GET /health`
- `GET /model/metrics`
- `POST /predict`
- `POST /score-portfolio`

No endpoint `POST /predict`, agora e possivel informar opcionalmente:

- `cnpj_sacado`
- `cnpj_cedente`
- `id_sacado`
- `id_cedente`

Quando um `cnpj_*` e enviado, o motor tenta enriquecer o score com:

- cadastro publico via BrasilAPI;
- sinais de bureau via mock deterministico.

Quando o dataset nao possui CNPJ aberto, o motor usa `id_*` ou os identificadores internos anonimizados para gerar features externas sinteticas, mantendo o pipeline compativel com a base demo.

## Enriquecimento externo para a banca

O projeto agora esta preparado para demonstrar a combinacao de dados internos com sinais externos de CNPJ.

### 1. Receita Federal via BrasilAPI

Endpoint:

- `GET /external/receita/{cnpj}`

Exemplo:

```bash
curl http://127.0.0.1:8000/external/receita/19131243000197
```

Retorna, quando disponivel na BrasilAPI:

- razao social;
- nome fantasia;
- CNAE principal;
- capital social;
- quadro societario;
- situacao cadastral;
- porte e natureza juridica.

### 2. Bureau de credito com mock

Endpoint:

- `GET /external/bureau/mock/{cnpj}`

Exemplo:

```bash
curl http://127.0.0.1:8000/external/bureau/mock/19131243000197
```

O retorno e um mock deterministico por CNPJ para simular:

- score bureau;
- rating bureau;
- protestos;
- consultas recentes de credito;
- flag de recuperacao judicial;
- flag de falencia;
- indicadores de estresse financeiro.

### 3. Enriquecimento consolidado

Endpoint:

- `GET /external/enrichment/{cnpj}`

Exemplo:

```bash
curl http://127.0.0.1:8000/external/enrichment/19131243000197
```

Esse endpoint junta:

- dados cadastrais publicos via BrasilAPI;
- mock de bureau;
- features sugeridas para o modelo.

## Hipotese de negocio

A tese do PRISMA e que a combinacao de:

- comportamento interno do boleto;
- robustez cadastral do CNPJ;
- sinais externos de protesto, busca por credito e eventos juridicos;

melhora a deteccao antecipada de fraude e inadimplencia em comparacao ao uso isolado de historico interno.

## Documentacao adicional

- `docs/arquitetura_dados_modelo.md`
  Arquitetura em camadas e catalogo de novas features.

- `data_lake/README.md`
  Organizacao do lake e destinos recomendados para ingestao e curadoria.
