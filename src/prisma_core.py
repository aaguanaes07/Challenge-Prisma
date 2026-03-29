from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
ASSETS_DIR = BASE_DIR / "assets"
BRANDING_DIR = ASSETS_DIR / "branding"
DASHBOARD_DIR = ASSETS_DIR / "dashboard"
DATA_LAKE_DIR = BASE_DIR / "data_lake"
ARTIFACTS_DIR = DATA_LAKE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RAW_DIR = DATA_LAKE_DIR / "raw"
RAW_BOLETOS_DIR = RAW_DIR / "boletos"
RAW_AUXILIAR_DIR = RAW_DIR / "auxiliar"
BOLETOS_FILENAME = "base_boletos_fiap.csv"
AUXILIAR_FILENAME = "base_auxiliar_fiap.csv"
BOLETOS_PATH = RAW_BOLETOS_DIR / BOLETOS_FILENAME
AUXILIAR_PATH = RAW_AUXILIAR_DIR / AUXILIAR_FILENAME
MODEL_FILENAME = "modelo_prisma_risco.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

NUMERIC_FEATURES = [
    "vlr_nominal",
    "media_atraso_dias_sacado",
    "score_materialidade_v2_sacado",
    "score_quantidade_v2_sacado",
    "sacado_indice_liquidez_1m_sacado",
    "share_vl_inad_pag_bol_6_a_15d_sacado",
    "media_atraso_dias_cedente",
    "score_materialidade_v2_cedente",
    "score_quantidade_v2_cedente",
    "cedente_indice_liquidez_1m_cedente",
    "indicador_liquidez_quantitativo_3m_cedente",
]

CATEGORICAL_FEATURES = [
    "tipo_especie",
    "uf_sacado",
    "uf_cedente",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class RiskBand:
    label: str
    decision: str
    action: str


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _load_csvs(base_dir: Path = BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    boletos_candidates = [
        BOLETOS_PATH,
        base_dir / BOLETOS_FILENAME,
    ]
    auxiliar_candidates = [
        AUXILIAR_PATH,
        base_dir / AUXILIAR_FILENAME,
    ]

    boletos_file = next((path for path in boletos_candidates if path.exists()), None)
    auxiliar_file = next((path for path in auxiliar_candidates if path.exists()), None)

    if boletos_file is None:
        raise FileNotFoundError(f"Base de boletos não encontrada em: {boletos_candidates}")
    if auxiliar_file is None:
        raise FileNotFoundError(f"Base auxiliar não encontrada em: {auxiliar_candidates}")

    boletos = pd.read_csv(boletos_file)
    auxiliar = pd.read_csv(auxiliar_file)
    return boletos, auxiliar


def prepare_model_base(
    boletos: pd.DataFrame | None = None,
    auxiliar: pd.DataFrame | None = None,
    base_dir: Path = BASE_DIR,
) -> pd.DataFrame:
    if boletos is None or auxiliar is None:
        boletos, auxiliar = _load_csvs(base_dir)

    boletos = boletos.copy()
    auxiliar = auxiliar.copy()

    boletos["dt_emissao"] = pd.to_datetime(boletos["dt_emissao"], errors="coerce")
    boletos["dt_vencimento"] = pd.to_datetime(boletos["dt_vencimento"], errors="coerce")
    boletos["dt_pagamento"] = pd.to_datetime(boletos["dt_pagamento"], errors="coerce")

    boletos["dias_atraso_real"] = (
        boletos["dt_pagamento"] - boletos["dt_vencimento"]
    ).dt.days
    boletos["dias_para_vencer"] = (
        boletos["dt_vencimento"] - boletos["dt_emissao"]
    ).dt.days

    # Para gestao de risco FIDC, atraso severo e ausencia de pagamento sao eventos
    # materialmente mais relevantes do que atrasos operacionais curtos.
    boletos["alvo_inadimplencia"] = (
        boletos["dt_pagamento"].isna() | (boletos["dias_atraso_real"] > 30)
    ).astype(int)

    model_df = boletos.merge(
        auxiliar,
        left_on="id_pagador",
        right_on="id_cnpj",
        how="left",
    )
    model_df = model_df.merge(
        auxiliar,
        left_on="id_beneficiario",
        right_on="id_cnpj",
        how="left",
        suffixes=("_sacado", "_cedente"),
    )

    for column in CATEGORICAL_FEATURES:
        model_df[column] = model_df[column].fillna("Desconhecido").astype(str)

    return model_df


def build_training_pipeline() -> Pipeline:
    numeric_transformer = KNNImputer(n_neighbors=5)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=3)


def build_default_payloads(model_df: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        defaults[column] = float(model_df[column].median())
    for column in CATEGORICAL_FEATURES:
        mode = model_df[column].mode(dropna=True)
        defaults[column] = str(mode.iloc[0]) if not mode.empty else "Desconhecido"
    return defaults


def evaluate_model(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "default_rate": float(y_true.mean()),
    }


def train_and_save_model(base_dir: Path = BASE_DIR) -> dict[str, Any]:
    model_df = prepare_model_base(base_dir=base_dir)
    X = model_df[FEATURE_COLUMNS]
    y = model_df["alvo_inadimplencia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = build_training_pipeline()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, probabilities)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "defaults": build_default_payloads(model_df),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model(base_dir: Path = BASE_DIR) -> dict[str, Any]:
    candidate_paths = [
        MODEL_PATH,
        base_dir / MODEL_FILENAME,
    ]
    for model_file in candidate_paths:
        if model_file.exists():
            try:
                return joblib.load(model_file)
            except Exception:
                # Em ambientes como Streamlit Cloud, diferencas de versao de
                # Python ou dependencias podem invalidar artefatos serializados.
                # Nesses casos, reexecutamos o treinamento a partir da camada raw.
                break
    return train_and_save_model(base_dir=base_dir)


def _resolve_risk_band(probability: float) -> RiskBand:
    if probability < 0.08:
        return RiskBand("Baixo", "Aprovado", "Elegivel para carteira padrao")
    if probability < 0.18:
        return RiskBand("Moderado", "Analise Manual", "Revisar limites e garantias")
    return RiskBand("Alto", "Bloquear", "Suspender entrada ou pedir mitigadores")


def _reason_codes(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row.get("sacado_indice_liquidez_1m_sacado", 1) < 0.35:
        reasons.append("Liquidez recente do sacado abaixo do patamar recomendado")
    if row.get("media_atraso_dias_sacado", 0) > 20:
        reasons.append("Historico de atraso do sacado indica comportamento recorrente")
    if row.get("score_materialidade_v2_sacado", 1000) < 500:
        reasons.append("Score de materialidade do sacado em nivel fragil")
    if row.get("cedente_indice_liquidez_1m_cedente", 1) < 0.45:
        reasons.append("Liquidez do cedente sugere menor capacidade de originacao saudavel")
    if row.get("score_materialidade_v2_cedente", 1000) < 550:
        reasons.append("Cedente com materialidade abaixo do nivel de conforto")
    if row.get("share_vl_inad_pag_bol_6_a_15d_sacado", 0) > 0.25:
        reasons.append("Sacado com concentracao relevante de atrasos curtos recorrentes")
    if not reasons:
        if row.get("__probabilidade_inadimplencia__", 0) >= 0.18:
            reasons.append("Combinacao de sinais estruturais elevou o risco agregado acima do limite de conforto")
        else:
            reasons.append("Perfil equilibrado nos principais indicadores estruturais")
    return reasons[:3]


def _prepare_features(
    df: pd.DataFrame,
    defaults: dict[str, Any],
) -> pd.DataFrame:
    prepared = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = defaults[column]

    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(defaults[column])

    for column in CATEGORICAL_FEATURES:
        prepared[column] = (
            prepared[column]
            .fillna(defaults[column])
            .astype(str)
            .replace({"nan": defaults[column]})
        )

    return prepared


def score_dataframe(
    df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    prepared = _prepare_features(df, bundle["defaults"])
    probabilities = bundle["model"].predict_proba(prepared[FEATURE_COLUMNS])[:, 1]
    prepared["__probabilidade_inadimplencia__"] = probabilities

    scored = df.copy()
    scored["probabilidade_inadimplencia"] = probabilities
    scored["score_prisma"] = ((1 - probabilities) * 1000).round().astype(int)

    bands = [_resolve_risk_band(prob) for prob in probabilities]
    scored["faixa_risco"] = [band.label for band in bands]
    scored["decisao_credito"] = [band.decision for band in bands]
    scored["acao_recomendada"] = [band.action for band in bands]
    scored["motivos_risco"] = prepared.apply(_reason_codes, axis=1)
    return scored


def build_portfolio_from_source(
    boletos: pd.DataFrame | None = None,
    auxiliar: pd.DataFrame | None = None,
    base_dir: Path = BASE_DIR,
) -> pd.DataFrame:
    model_df = prepare_model_base(boletos=boletos, auxiliar=auxiliar, base_dir=base_dir)
    return model_df


def score_portfolio(
    boletos: pd.DataFrame | None = None,
    auxiliar: pd.DataFrame | None = None,
    base_dir: Path = BASE_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = load_or_train_model(base_dir=base_dir)
    portfolio = build_portfolio_from_source(boletos=boletos, auxiliar=auxiliar, base_dir=base_dir)
    scored = score_dataframe(portfolio, bundle)
    return scored, bundle


def build_manual_input(payload: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    record = dict(bundle["defaults"])
    record.update(payload)
    return pd.DataFrame([record])


def predict_from_payload(payload: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    active_bundle = bundle or load_or_train_model()
    scored = score_dataframe(build_manual_input(payload, active_bundle), active_bundle).iloc[0]
    return {
        "score_prisma": int(scored["score_prisma"]),
        "probabilidade_inadimplencia": float(round(scored["probabilidade_inadimplencia"] * 100, 2)),
        "faixa_risco": str(scored["faixa_risco"]),
        "status": str(scored["decisao_credito"]),
        "acao_recomendada": str(scored["acao_recomendada"]),
        "motivos_risco": list(scored["motivos_risco"]),
    }


def portfolio_summary(scored: pd.DataFrame) -> dict[str, Any]:
    total_titulos = int(len(scored))
    volume_total = float(scored["vlr_nominal"].fillna(0).sum())
    perda_esperada = float(
        (scored["vlr_nominal"].fillna(0) * scored["probabilidade_inadimplencia"]).sum()
    )
    aprovados = int((scored["decisao_credito"] == "Aprovado").sum())
    bloqueados = int((scored["decisao_credito"] == "Bloquear").sum())
    analise_manual = int((scored["decisao_credito"] == "Analise Manual").sum())

    return {
        "total_titulos": total_titulos,
        "volume_total": volume_total,
        "score_medio": float(scored["score_prisma"].mean()),
        "perda_esperada": perda_esperada,
        "ticket_medio": _safe_divide(volume_total, total_titulos),
        "aprovados": aprovados,
        "analise_manual": analise_manual,
        "bloqueados": bloqueados,
        "taxa_aprovacao": _safe_divide(aprovados, total_titulos),
        "inadimplencia_observada": float(scored["alvo_inadimplencia"].mean())
        if "alvo_inadimplencia" in scored.columns
        else None,
    }
