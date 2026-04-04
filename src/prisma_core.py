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

try:
    from .enrichment import (
        build_company_enrichment,
        build_model_features_from_profile,
        mock_external_profile,
        sanitize_cnpj,
    )
except ImportError:
    from enrichment import (
        build_company_enrichment,
        build_model_features_from_profile,
        mock_external_profile,
        sanitize_cnpj,
    )

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
    "idade_empresa_dias_sacado",
    "capital_social_sacado",
    "capital_social_por_valor_titulo_sacado",
    "qtd_socios_sacado",
    "empresa_ativa_sacado",
    "qtd_protestos_12m_sacado",
    "dias_desde_ultimo_protesto_sacado",
    "qtd_consultas_credito_30d_sacado",
    "score_bureau_sacado",
    "flag_recuperacao_judicial_sacado",
    "flag_falencia_sacado",
    "flag_spike_consulta_credito_sacado",
    "idade_empresa_dias_cedente",
    "capital_social_cedente",
    "capital_social_por_valor_titulo_cedente",
    "qtd_socios_cedente",
    "empresa_ativa_cedente",
    "qtd_protestos_12m_cedente",
    "dias_desde_ultimo_protesto_cedente",
    "qtd_consultas_credito_30d_cedente",
    "score_bureau_cedente",
    "flag_recuperacao_judicial_cedente",
    "flag_falencia_cedente",
    "flag_spike_consulta_credito_cedente",
    "cedente_volume_historico",
    "cedente_taxa_default",
    "cedente_ticket_medio",
    "score_cedente_proprio",
]

CATEGORICAL_FEATURES = [
    "tipo_especie",
    "uf_sacado",
    "uf_cedente",
    "status_cadastral_sacado",
    "natureza_juridica_sacado",
    "porte_empresa_sacado",
    "cnae_principal_codigo_sacado",
    "status_cadastral_cedente",
    "natureza_juridica_cedente",
    "porte_empresa_cedente",
    "cnae_principal_codigo_cedente",
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


def _build_external_feature_frame(
    entity_ids: pd.Series,
    nominal_values: pd.Series,
    role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entity_id, nominal in zip(entity_ids.fillna("Desconhecido"), nominal_values.fillna(0), strict=False):
        profile = mock_external_profile(str(entity_id), role=role)
        features = build_model_features_from_profile(profile, nominal_value=float(nominal or 0))
        rows.append(features)
    return pd.DataFrame(rows, index=entity_ids.index).add_suffix(f"_{role}")


def _apply_external_enrichment(model_df: pd.DataFrame) -> pd.DataFrame:
    enriched = model_df.copy()
    sacado_features = _build_external_feature_frame(
        enriched["id_pagador"],
        enriched["vlr_nominal"],
        role="sacado",
    )
    cedente_features = _build_external_feature_frame(
        enriched["id_beneficiario"],
        enriched["vlr_nominal"],
        role="cedente",
    )
    return pd.concat([enriched, sacado_features, cedente_features], axis=1)


def _build_nominal_reference(boletos: pd.DataFrame) -> dict[str, Any]:
    valid = boletos.copy()
    valid["vlr_nominal"] = pd.to_numeric(valid["vlr_nominal"], errors="coerce").fillna(0)
    overall_avg = float(valid["vlr_nominal"].mean())

    by_pagador = (
        valid.groupby("id_pagador", dropna=False)["vlr_nominal"].mean().astype(float).to_dict()
        if "id_pagador" in valid.columns
        else {}
    )
    by_beneficiario = (
        valid.groupby("id_beneficiario", dropna=False)["vlr_nominal"].mean().astype(float).to_dict()
        if "id_beneficiario" in valid.columns
        else {}
    )

    return {
        "overall_avg": overall_avg,
        "by_pagador": by_pagador,
        "by_beneficiario": by_beneficiario,
    }


def _build_delay_curve(boletos: pd.DataFrame) -> dict[str, Any]:
    atraso = pd.to_numeric(boletos.get("dias_atraso_real"), errors="coerce")
    atraso = atraso[atraso.notna() & (atraso >= 0)]
    if atraso.empty:
        return {
            "dias": [],
            "curva_percentual": [],
            "threshold_dias": 5,
            "share_ate_threshold": 0.0,
        }

    atraso_por_dia = atraso.round().astype(int).value_counts().sort_index()
    cumulative = (atraso_por_dia.cumsum() / atraso_por_dia.sum()) * 100
    share_ate_5 = float((atraso <= 5).mean())
    return {
        "dias": atraso_por_dia.index.astype(float).tolist(),
        "curva_percentual": cumulative.astype(float).tolist(),
        "threshold_dias": 5,
        "share_ate_threshold": share_ate_5,
    }


def _build_cedente_intelligence(boletos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cedente_info = (
        boletos.groupby("id_beneficiario", dropna=False)
        .agg(
            cedente_volume_historico=("id_boleto", "count"),
            cedente_taxa_default=("alvo_inadimplencia", "mean"),
            cedente_ticket_medio=("vlr_nominal", "mean"),
            cedente_volume_financeiro=("vlr_nominal", "sum"),
        )
        .reset_index()
    )

    def calcular_score_cedente(row: pd.Series) -> float:
        base_score = 1000 * (1 - float(row["cedente_taxa_default"]))
        bonus_volume = min(float(row["cedente_volume_historico"]) * 2, 100)
        return float((base_score * 0.9) + bonus_volume)

    cedente_info["score_cedente_proprio"] = cedente_info.apply(calcular_score_cedente, axis=1)

    alertas = cedente_info[
        (cedente_info["cedente_volume_historico"] > 5)
        & (cedente_info["cedente_taxa_default"] > 0.2)
    ].copy()

    intelligence_payload = {
        "table": cedente_info,
        "map_volume": cedente_info.set_index("id_beneficiario")["cedente_volume_historico"].astype(float).to_dict(),
        "map_taxa_default": cedente_info.set_index("id_beneficiario")["cedente_taxa_default"].astype(float).to_dict(),
        "map_ticket_medio": cedente_info.set_index("id_beneficiario")["cedente_ticket_medio"].astype(float).to_dict(),
        "map_score": cedente_info.set_index("id_beneficiario")["score_cedente_proprio"].astype(float).to_dict(),
        "alerts": alertas.sort_values(
            ["cedente_taxa_default", "cedente_volume_financeiro"],
            ascending=[False, False],
        ).to_dict(orient="records"),
    }
    return cedente_info, intelligence_payload


def _extract_feature_importance(model: CalibratedClassifierCV) -> list[dict[str, Any]]:
    aggregated: dict[str, list[float]] = {}

    for calibrated in getattr(model, "calibrated_classifiers_", []):
        estimator = getattr(calibrated, "estimator", None)
        if estimator is None:
            continue
        pipeline = estimator
        classifier = pipeline.named_steps.get("classifier")
        preprocessor = pipeline.named_steps.get("preprocessor")
        if classifier is None or preprocessor is None or not hasattr(classifier, "feature_importances_"):
            continue
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        for feature_name, importance in zip(feature_names, importances, strict=False):
            aggregated.setdefault(feature_name, []).append(float(importance))

    if not aggregated:
        return []

    feature_importance = (
        pd.DataFrame(
            {
                "feature": list(aggregated.keys()),
                "importance": [float(np.mean(values)) for values in aggregated.values()],
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )
    return feature_importance.to_dict(orient="records")


def _train_comparison_model(
    model_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    X = model_df[numeric_features + categorical_features]
    y = model_df["alvo_inadimplencia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    numeric_transformer = KNNImputer(n_neighbors=5)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, probabilities)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    importances = classifier.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    return {
        "metrics": metrics,
        "feature_importance": importance_df.to_dict(orient="records"),
    }


def _resolve_nominal_reference_value(
    nominal_reference: dict[str, Any],
    pagador_id: Any = None,
    beneficiario_id: Any = None,
) -> float:
    overall_avg = float(nominal_reference.get("overall_avg", 0) or 0)
    if pagador_id is not None:
        pagador_avg = nominal_reference.get("by_pagador", {}).get(pagador_id)
        if pagador_avg is not None:
            return float(pagador_avg)
    if beneficiario_id is not None:
        beneficiario_avg = nominal_reference.get("by_beneficiario", {}).get(beneficiario_id)
        if beneficiario_avg is not None:
            return float(beneficiario_avg)
    return overall_avg


def _apply_payload_external_enrichment(payload: dict[str, Any]) -> dict[str, Any]:
    enriched_payload = dict(payload)
    referencia_pagador = enriched_payload.pop("referencia_pagador", None)
    referencia_beneficiario = enriched_payload.pop("referencia_beneficiario", None)

    if referencia_pagador and not enriched_payload.get("cnpj_sacado") and not enriched_payload.get("id_sacado"):
        ref = str(referencia_pagador).strip()
        enriched_payload["cnpj_sacado" if ref.isdigit() and len(ref) == 14 else "id_sacado"] = ref

    if referencia_beneficiario and not enriched_payload.get("cnpj_cedente") and not enriched_payload.get("id_cedente"):
        ref = str(referencia_beneficiario).strip()
        enriched_payload["cnpj_cedente" if ref.isdigit() and len(ref) == 14 else "id_cedente"] = ref

    nominal = float(enriched_payload.get("vlr_nominal", 0) or 0)

    for role in ("sacado", "cedente"):
        cnpj_key = f"cnpj_{role}"
        entity_key = f"id_{role}"
        cnpj_value = enriched_payload.get(cnpj_key)
        entity_value = enriched_payload.get(entity_key)

        profile: dict[str, Any] | None = None
        if cnpj_value:
            try:
                profile = build_company_enrichment(sanitize_cnpj(str(cnpj_value)))
            except Exception:
                profile = None
        if profile is None and entity_value:
            profile = mock_external_profile(str(entity_value), role=role)
        if profile is None:
            continue

        feature_map = build_model_features_from_profile(profile, nominal_value=nominal)
        for key, value in feature_map.items():
            enriched_payload[f"{key}_{role}"] = value

        enriched_payload[f"real_cnpj_enrichment_{role}"] = bool(profile.get("real_cnpj_enrichment"))

        receita = profile.get("receita_federal", {})
        if role == "sacado" and not enriched_payload.get("uf_sacado"):
            enriched_payload["uf_sacado"] = receita.get("uf")
        if role == "cedente" and not enriched_payload.get("uf_cedente"):
            enriched_payload["uf_cedente"] = receita.get("uf")
        if role == "sacado" and not enriched_payload.get("cnae_principal_codigo_sacado"):
            enriched_payload["cnae_principal_codigo_sacado"] = receita.get("cnae_principal_codigo")
        if role == "cedente" and not enriched_payload.get("cnae_principal_codigo_cedente"):
            enriched_payload["cnae_principal_codigo_cedente"] = receita.get("cnae_principal_codigo")

    return enriched_payload


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

    nominal_reference = _build_nominal_reference(boletos)
    boletos["ticket_medio_pagador"] = boletos["id_pagador"].map(nominal_reference["by_pagador"])
    boletos["ticket_medio_beneficiario"] = boletos["id_beneficiario"].map(nominal_reference["by_beneficiario"])
    boletos["ticket_medio_geral"] = nominal_reference["overall_avg"]
    boletos["ticket_medio_referencia"] = boletos["ticket_medio_pagador"].fillna(
        boletos["ticket_medio_beneficiario"]
    ).fillna(boletos["ticket_medio_geral"])

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

    model_df = _apply_external_enrichment(model_df)
    cedente_info, _ = _build_cedente_intelligence(boletos)
    model_df = model_df.merge(
        cedente_info[[
            "id_beneficiario",
            "cedente_volume_historico",
            "cedente_taxa_default",
            "cedente_ticket_medio",
            "score_cedente_proprio",
        ]],
        on="id_beneficiario",
        how="left",
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
    boletos, auxiliar = _load_csvs(base_dir)
    model_df = prepare_model_base(boletos=boletos, auxiliar=auxiliar, base_dir=base_dir)
    cedente_info, cedente_payload = _build_cedente_intelligence(model_df)
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
    feature_importance = _extract_feature_importance(model)

    baseline_numeric = [
        "vlr_nominal",
        "media_atraso_dias_sacado",
        "score_materialidade_v2_sacado",
        "score_quantidade_v2_sacado",
        "sacado_indice_liquidez_1m_sacado",
        "media_atraso_dias_cedente",
        "score_materialidade_v2_cedente",
        "score_quantidade_v2_cedente",
        "cedente_indice_liquidez_1m_cedente",
        "indicador_liquidez_quantitativo_3m_cedente",
    ]
    baseline_categorical = ["tipo_especie", "uf_sacado", "uf_cedente"]
    baseline_comparison = _train_comparison_model(model_df, baseline_numeric, baseline_categorical)
    enriched_comparison = {
        "metrics": metrics,
        "feature_importance": feature_importance,
    }

    delay_curve = _build_delay_curve(model_df)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "defaults": build_default_payloads(model_df),
        "nominal_reference": _build_nominal_reference(boletos),
        "cedente_intelligence": {
            "records": cedente_payload["table"].sort_values(
                ["cedente_volume_financeiro", "cedente_taxa_default"],
                ascending=[False, False],
            ).to_dict(orient="records"),
            "alerts": cedente_payload["alerts"],
            "maps": {
                "volume": cedente_payload["map_volume"],
                "taxa_default": cedente_payload["map_taxa_default"],
                "ticket_medio": cedente_payload["map_ticket_medio"],
                "score": cedente_payload["map_score"],
            },
        },
        "delay_curve": delay_curve,
        "feature_importance": feature_importance,
        "model_evolution": {
            "baseline_interno": baseline_comparison,
            "modelo_enriquecido": enriched_comparison,
        },
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
                bundle = joblib.load(model_file)
                bundle_features = bundle.get("feature_columns", [])
                if list(bundle_features) != list(FEATURE_COLUMNS):
                    break
                if "nominal_reference" not in bundle:
                    break
                if "cedente_intelligence" not in bundle or "model_evolution" not in bundle:
                    break
                delay_days = bundle.get("delay_curve", {}).get("dias", [])
                if delay_days and len(delay_days) != len(set(delay_days)):
                    break
                return bundle
            except Exception:
                # Em ambientes como Streamlit Cloud, diferencas de versao de
                # Python ou dependencias podem invalidar artefatos serializados.
                # Nesses casos, reexecutamos o treinamento a partir da camada raw.
                break
    return train_and_save_model(base_dir=base_dir)


def _has_hard_block(row: pd.Series) -> tuple[bool, str | None]:
    for role in ("sacado", "cedente"):
        if not bool(row.get(f"real_cnpj_enrichment_{role}", False)):
            continue

        status = str(row.get(f"status_cadastral_{role}", "")).strip().upper()
        if status == "BAIXADA":
            return True, f"{role.capitalize()} com CNPJ baixado na Receita Federal"

        if float(row.get(f"flag_recuperacao_judicial_{role}", 0) or 0) >= 1:
            return True, f"{role.capitalize()} com indicio de recuperacao judicial"

    return False, None


def _resolve_probability_threshold(row: pd.Series) -> float:
    nominal = float(row.get("vlr_nominal", 0) or 0)
    reference = float(row.get("ticket_medio_referencia", 0) or 0)
    if reference > 0 and nominal >= reference:
        return 0.10
    return 0.15


def _resolve_risk_band(probability: float, row: pd.Series | None = None) -> RiskBand:
    if row is not None:
        hard_block, reason = _has_hard_block(row)
        if hard_block:
            return RiskBand("Alto", "Negado", reason or "Regra critica de bloqueio acionada")
        threshold = _resolve_probability_threshold(row)
    else:
        threshold = 0.15

    if probability < threshold:
        return RiskBand("Baixo", "Aprovado", "Elegivel para carteira padrao")
    if threshold <= 0.10:
        return RiskBand("Alto", "Negado", "Valor nominal acima da referencia e inadimplencia acima de 10%")
    return RiskBand("Alto", "Negado", "Inadimplencia acima do limite de aprovacao de 15%")


def _reason_codes(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    hard_block, hard_block_reason = _has_hard_block(row)
    if hard_block and hard_block_reason:
        reasons.append(hard_block_reason)
    threshold = _resolve_probability_threshold(row)
    reference = float(row.get("ticket_medio_referencia", 0) or 0)
    nominal = float(row.get("vlr_nominal", 0) or 0)
    if reference > 0 and nominal >= reference:
        reasons.append("Valor nominal acima do ticket medio de referencia, com regua mais conservadora de 10%")
    if row.get("flag_recuperacao_judicial_sacado", 0) >= 1:
        reasons.append("Sacado com indicio de recuperacao judicial no enriquecimento externo")
    if row.get("flag_falencia_sacado", 0) >= 1:
        reasons.append("Sacado com indicio de falencia no monitoramento externo")
    if row.get("qtd_protestos_12m_sacado", 0) >= 3:
        reasons.append("Sacado com historico relevante de protestos nos ultimos 12 meses")
    if row.get("flag_spike_consulta_credito_sacado", 0) >= 1:
        reasons.append("Sacado apresentou pico recente de busca por credito")
    if str(row.get("status_cadastral_sacado", "")).upper() in {"BAIXADA", "SUSPENSA", "INAPTA"}:
        reasons.append("Situacao cadastral do sacado fora do padrao esperado")
    if row.get("capital_social_por_valor_titulo_sacado", 999) < 0.25:
        reasons.append("Capital social do sacado parece baixo frente ao valor da operacao")
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
    if not reasons:
        if row.get("__probabilidade_inadimplencia__", 0) >= threshold:
            reasons.append(
                "Combinacao de sinais estruturais elevou a inadimplencia acima do limite de aprovacao"
            )
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

    bands = [_resolve_risk_band(prob, prepared.iloc[idx]) for idx, prob in enumerate(probabilities)]
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
    enriched_payload = _apply_payload_external_enrichment(payload)
    pagador_id = enriched_payload.get("id_sacado") or enriched_payload.get("id_pagador")
    beneficiario_id = enriched_payload.get("id_cedente") or enriched_payload.get("id_beneficiario")
    nominal_reference = bundle.get("nominal_reference", {})
    enriched_payload["ticket_medio_referencia"] = _resolve_nominal_reference_value(
        nominal_reference,
        pagador_id=pagador_id,
        beneficiario_id=beneficiario_id,
    )
    cedente_maps = bundle.get("cedente_intelligence", {}).get("maps", {})
    beneficiary_key = beneficiario_id
    if beneficiary_key is not None:
        enriched_payload["cedente_volume_historico"] = cedente_maps.get("volume", {}).get(beneficiary_key)
        enriched_payload["cedente_taxa_default"] = cedente_maps.get("taxa_default", {}).get(beneficiary_key)
        enriched_payload["cedente_ticket_medio"] = cedente_maps.get("ticket_medio", {}).get(beneficiary_key)
        enriched_payload["score_cedente_proprio"] = cedente_maps.get("score", {}).get(beneficiary_key)
    record.update(enriched_payload)
    return pd.DataFrame([record])


def predict_from_payload(payload: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    active_bundle = bundle or load_or_train_model()
    scored = score_dataframe(build_manual_input(payload, active_bundle), active_bundle).iloc[0]
    threshold = _resolve_probability_threshold(scored)
    ticket_reference = float(scored.get("ticket_medio_referencia", 0) or 0)
    nominal_value = float(scored.get("vlr_nominal", 0) or 0)
    return {
        "score_prisma": int(scored["score_prisma"]),
        "probabilidade_inadimplencia": float(round(scored["probabilidade_inadimplencia"] * 100, 2)),
        "faixa_risco": str(scored["faixa_risco"]),
        "status": str(scored["decisao_credito"]),
        "acao_recomendada": str(scored["acao_recomendada"]),
        "motivos_risco": list(scored["motivos_risco"]),
        "regua_inadimplencia_aplicada": float(round(threshold * 100, 2)),
        "ticket_medio_referencia": float(round(ticket_reference, 2)),
        "valor_nominal_analisado": float(round(nominal_value, 2)),
    }


def portfolio_summary(scored: pd.DataFrame) -> dict[str, Any]:
    total_titulos = int(len(scored))
    volume_total = float(scored["vlr_nominal"].fillna(0).sum())
    perda_esperada = float(
        (scored["vlr_nominal"].fillna(0) * scored["probabilidade_inadimplencia"]).sum()
    )
    aprovados = int((scored["decisao_credito"] == "Aprovado").sum())
    bloqueados = int((scored["decisao_credito"] == "Negado").sum())

    return {
        "total_titulos": total_titulos,
        "volume_total": volume_total,
        "score_medio": float(scored["score_prisma"].mean()),
        "perda_esperada": perda_esperada,
        "ticket_medio": _safe_divide(volume_total, total_titulos),
        "aprovados": aprovados,
        "analise_manual": 0,
        "bloqueados": bloqueados,
        "taxa_aprovacao": _safe_divide(aprovados, total_titulos),
        "inadimplencia_observada": float(scored["alvo_inadimplencia"].mean())
        if "alvo_inadimplencia" in scored.columns
        else None,
    }
