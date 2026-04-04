from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any

import requests

BRASIL_API_CNPJ_URL = "https://brasilapi.com.br/api/cnpj/v1/{cnpj}"


def sanitize_cnpj(cnpj: str) -> str:
    digits = "".join(char for char in str(cnpj) if char.isdigit())
    if len(digits) != 14:
        raise ValueError("CNPJ deve conter 14 digitos.")
    return digits


def _extract_primary_cnae(payload: dict[str, Any]) -> dict[str, Any]:
    codigo = payload.get("cnae_fiscal")
    descricao = payload.get("cnae_fiscal_descricao")

    if codigo or descricao:
        return {
            "codigo": str(codigo) if codigo is not None else None,
            "descricao": descricao,
        }

    secondary = payload.get("cnaes_secundarios") or []
    if secondary and isinstance(secondary, list):
        first = secondary[0]
        return {
            "codigo": str(first.get("codigo")) if first.get("codigo") is not None else None,
            "descricao": first.get("descricao"),
        }

    return {"codigo": None, "descricao": None}


def _extract_partner_board(payload: dict[str, Any]) -> list[dict[str, Any]]:
    qsa = payload.get("qsa") or payload.get("quadro_societario") or []
    partners: list[dict[str, Any]] = []

    if not isinstance(qsa, list):
        return partners

    for item in qsa:
        if not isinstance(item, dict):
            continue
        partners.append(
            {
                "nome_socio": item.get("nome_socio") or item.get("nome") or item.get("razao_social"),
                "qualificacao_socio": item.get("qualificacao_socio") or item.get("qual") or item.get("descricao"),
                "faixa_etaria": item.get("faixa_etaria"),
                "data_entrada_sociedade": item.get("data_entrada_sociedade"),
                "cpf_cnpj_socio": item.get("cnpj_cpf_do_socio") or item.get("cpf_cnpj_socio"),
            }
        )

    return partners


def normalize_brasilapi_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cnae = _extract_primary_cnae(payload)
    partners = _extract_partner_board(payload)
    situacao = (
        payload.get("descricao_situacao_cadastral")
        or payload.get("situacao_cadastral")
        or payload.get("descricao_situacao")
    )

    return {
        "fonte": "brasilapi",
        "cnpj": sanitize_cnpj(payload.get("cnpj", "")),
        "razao_social": payload.get("razao_social"),
        "nome_fantasia": payload.get("nome_fantasia"),
        "capital_social": float(payload.get("capital_social", 0) or 0),
        "cnae_principal_codigo": cnae["codigo"],
        "cnae_principal_descricao": cnae["descricao"],
        "situacao_cadastral": situacao,
        "empresa_ativa": str(situacao).strip().upper() == "ATIVA" if situacao else None,
        "data_inicio_atividade": payload.get("data_inicio_atividade"),
        "natureza_juridica": payload.get("natureza_juridica"),
        "porte": payload.get("porte"),
        "uf": payload.get("uf"),
        "municipio": payload.get("municipio"),
        "quadro_societario": partners,
        "qtd_socios": len(partners),
        "payload_original": payload,
    }


def fetch_brasilapi_company(cnpj: str, timeout: float = 8.0) -> dict[str, Any]:
    normalized_cnpj = sanitize_cnpj(cnpj)
    response = requests.get(
        BRASIL_API_CNPJ_URL.format(cnpj=normalized_cnpj),
        timeout=timeout,
    )
    response.raise_for_status()
    return normalize_brasilapi_payload(response.json())


def mock_bureau_report(cnpj: str) -> dict[str, Any]:
    normalized_cnpj = sanitize_cnpj(cnpj)
    digest = hashlib.sha256(normalized_cnpj.encode("utf-8")).hexdigest()
    return _mock_bureau_from_digest(normalized_cnpj, digest)


def _mock_bureau_from_digest(entity_id: str, digest: str) -> dict[str, Any]:
    seed = int(digest[:12], 16)

    protestos_12m = seed % 7
    consultas_30d = (seed // 7) % 18
    consultas_7d = min(consultas_30d, (seed // 13) % 8)
    score_bureau = 350 + (seed % 551)
    dias_desde_ultimo_protesto = None if protestos_12m == 0 else 5 + ((seed // 17) % 180)
    flag_rj = (seed % 29) == 0
    flag_falencia = (seed % 97) == 0
    flag_cheque_sem_fundo = (seed % 11) <= 1
    valor_total_protestos = round(float(protestos_12m * (15_000 + (seed % 85_000))), 2)
    intensidade_credito = round(float(consultas_30d / 30), 2)

    if score_bureau >= 750:
        rating = "AA"
    elif score_bureau >= 680:
        rating = "A"
    elif score_bureau >= 600:
        rating = "B"
    elif score_bureau >= 500:
        rating = "C"
    else:
        rating = "D"

    return {
        "fonte": "bureau_mock",
        "motor": "simulador_serasa",
        "cnpj": entity_id if entity_id.isdigit() and len(entity_id) == 14 else None,
        "entity_id": entity_id,
        "score_bureau": score_bureau,
        "rating_bureau": rating,
        "qtd_protestos_12m": protestos_12m,
        "vl_total_protestos_12m": valor_total_protestos,
        "dias_desde_ultimo_protesto": dias_desde_ultimo_protesto,
        "qtd_consultas_credito_7d": consultas_7d,
        "qtd_consultas_credito_30d": consultas_30d,
        "intensidade_busca_credito_recente": intensidade_credito,
        "flag_spike_consulta_credito": consultas_7d >= 5 or consultas_30d >= 12,
        "flag_cheque_sem_fundo": flag_cheque_sem_fundo,
        "flag_recuperacao_judicial": flag_rj,
        "flag_falencia": flag_falencia,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "observacao": "Mock deterministico para demonstrar integracao com bureau enquanto a licenca real nao foi contratada.",
    }


def mock_external_profile(entity_key: str, role: str = "sacado") -> dict[str, Any]:
    normalized_key = str(entity_key).strip()
    if not normalized_key:
        raise ValueError("Identificador da entidade nao pode ser vazio.")

    digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16)
    bureau = _mock_bureau_from_digest(normalized_key, digest)

    situacao = "BAIXADA" if (seed % 61) == 0 else "ATIVA"
    porte = ["ME", "EPP", "DEMAIS"][seed % 3]
    natureza = ["LTDA", "SA", "EMPRESARIO INDIVIDUAL", "SOCIEDADE SIMPLES"][seed % 4]
    cnae_codigo = str(1000 + (seed % 8999))
    ufs = ["SP", "RJ", "MG", "PR", "SC", "RS", "BA", "GO", "PE", "CE"]
    uf = ufs[seed % len(ufs)]
    capital_social = float(10_000 + (seed % 4_990_000))
    idade_empresa_dias = int(90 + (seed % 7_300))
    qtd_socios = int(1 + (seed % 6))

    return {
        "entity_id": normalized_key,
        "role": role,
        "receita_federal": {
            "fonte": "receita_mock",
            "cnpj": normalized_key if normalized_key.isdigit() and len(normalized_key) == 14 else None,
            "razao_social": f"Empresa Mock {normalized_key[:8]}",
            "capital_social": capital_social,
            "cnae_principal_codigo": cnae_codigo,
            "situacao_cadastral": situacao,
            "empresa_ativa": situacao == "ATIVA",
            "natureza_juridica": natureza,
            "porte": porte,
            "uf": uf,
            "qtd_socios": qtd_socios,
            "idade_empresa_dias": idade_empresa_dias,
            "observacao": "Perfil cadastral sintetico para treino e demo em base anonimizada.",
        },
        "bureau": bureau,
    }


def build_model_features_from_profile(profile: dict[str, Any], nominal_value: float | None = None) -> dict[str, Any]:
    receita = profile.get("receita_federal", {})
    bureau = profile.get("bureau", {})
    capital_social = float(receita.get("capital_social", 0) or 0)
    nominal = float(nominal_value or 0)
    capital_ratio = capital_social / nominal if nominal > 0 else capital_social

    return {
        "idade_empresa_dias": float(receita.get("idade_empresa_dias", 0) or 0),
        "capital_social": capital_social,
        "capital_social_por_valor_titulo": float(capital_ratio),
        "qtd_socios": float(receita.get("qtd_socios", 0) or 0),
        "empresa_ativa": 1.0 if receita.get("empresa_ativa") else 0.0,
        "qtd_protestos_12m": float(bureau.get("qtd_protestos_12m", 0) or 0),
        "dias_desde_ultimo_protesto": float(bureau.get("dias_desde_ultimo_protesto", 365) or 365),
        "qtd_consultas_credito_30d": float(bureau.get("qtd_consultas_credito_30d", 0) or 0),
        "score_bureau": float(bureau.get("score_bureau", 0) or 0),
        "flag_recuperacao_judicial": 1.0 if bureau.get("flag_recuperacao_judicial") else 0.0,
        "flag_falencia": 1.0 if bureau.get("flag_falencia") else 0.0,
        "flag_spike_consulta_credito": 1.0 if bureau.get("flag_spike_consulta_credito") else 0.0,
        "status_cadastral": str(receita.get("situacao_cadastral") or "Desconhecido"),
        "natureza_juridica": str(receita.get("natureza_juridica") or "Desconhecido"),
        "porte_empresa": str(receita.get("porte") or "Desconhecido"),
        "cnae_principal_codigo": str(receita.get("cnae_principal_codigo") or "Desconhecido"),
    }


def build_company_enrichment(cnpj: str, timeout: float = 8.0) -> dict[str, Any]:
    receita = fetch_brasilapi_company(cnpj, timeout=timeout)
    bureau = mock_bureau_report(cnpj)

    return {
        "cnpj": receita["cnpj"],
        "real_cnpj_enrichment": True,
        "consultado_em": datetime.now().isoformat(timespec="seconds"),
        "receita_federal": receita,
        "bureau": bureau,
        "features_sugeridas": {
            "idade_empresa_data_base": receita.get("data_inicio_atividade"),
            "capital_social": receita.get("capital_social"),
            "qtd_socios": receita.get("qtd_socios"),
            "empresa_ativa": receita.get("empresa_ativa"),
            "cnae_principal_codigo": receita.get("cnae_principal_codigo"),
            "qtd_protestos_12m": bureau.get("qtd_protestos_12m"),
            "dias_desde_ultimo_protesto": bureau.get("dias_desde_ultimo_protesto"),
            "qtd_consultas_credito_30d": bureau.get("qtd_consultas_credito_30d"),
            "score_bureau": bureau.get("score_bureau"),
            "flag_recuperacao_judicial": bureau.get("flag_recuperacao_judicial"),
        },
    }
