from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .prisma_core import (
        BASE_DIR,
        FEATURE_COLUMNS,
        load_or_train_model,
        portfolio_summary,
        predict_from_payload,
        score_dataframe,
    )
except ImportError:
    from prisma_core import (
        BASE_DIR,
        FEATURE_COLUMNS,
        load_or_train_model,
        portfolio_summary,
        predict_from_payload,
        score_dataframe,
    )

app = FastAPI(
    title="PRISMA Risk API",
    version="3.0.0",
    description="API de scoring para gestao de risco estrutural de FIDCs.",
)

MODEL_BUNDLE = load_or_train_model(BASE_DIR)


class TituloInput(BaseModel):
    vlr_nominal: float
    tipo_especie: str
    media_atraso_dias_sacado: float | None = None
    score_materialidade_v2_sacado: float | None = None
    score_quantidade_v2_sacado: float | None = None
    sacado_indice_liquidez_1m_sacado: float | None = None
    share_vl_inad_pag_bol_6_a_15d_sacado: float | None = None
    media_atraso_dias_cedente: float | None = None
    score_materialidade_v2_cedente: float | None = None
    score_quantidade_v2_cedente: float | None = None
    cedente_indice_liquidez_1m_cedente: float | None = None
    indicador_liquidez_quantitativo_3m_cedente: float | None = None
    uf_sacado: str | None = None
    uf_cedente: str | None = None


class PortfolioInput(BaseModel):
    registros: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": True,
        "trained_at": MODEL_BUNDLE["trained_at"],
        "default_rate": MODEL_BUNDLE["metrics"]["default_rate"],
    }


@app.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    return {
        "trained_at": MODEL_BUNDLE["trained_at"],
        "features": FEATURE_COLUMNS,
        "metrics": MODEL_BUNDLE["metrics"],
    }


@app.post("/predict")
def predict_risco(dados: TituloInput) -> dict[str, Any]:
    payload = dados.model_dump()
    return predict_from_payload(payload, MODEL_BUNDLE)


@app.post("/score-portfolio")
def score_portfolio_endpoint(payload: PortfolioInput) -> dict[str, Any]:
    records = pd.DataFrame(payload.registros)
    scored = score_dataframe(records, MODEL_BUNDLE)
    response = scored.copy()
    response["motivos_risco"] = response["motivos_risco"].apply(list)
    return {
        "summary": portfolio_summary(scored),
        "records": response.to_dict(orient="records"),
    }
