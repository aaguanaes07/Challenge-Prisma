from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from .prisma_core import (
        BASE_DIR,
        BRANDING_DIR,
        DASHBOARD_DIR,
        FEATURE_COLUMNS,
        build_manual_input,
        load_or_train_model,
        predict_from_payload,
        score_portfolio,
    )
except ImportError:
    from prisma_core import (
        BASE_DIR,
        BRANDING_DIR,
        DASHBOARD_DIR,
        FEATURE_COLUMNS,
        build_manual_input,
        load_or_train_model,
        predict_from_payload,
        score_portfolio,
    )

st.set_page_config(page_title="PRISMA | Gestao de Risco FIDC", layout="wide")

API_URL = "http://127.0.0.1:8000"
PRISMA_GREEN = "#00E096"
UFS_BRASIL = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]
CNAE_DIVISION_LABELS = {
    "10": "Industria de Transformacao (Fabricacao de produtos alimenticios)",
    "11": "Industria de Transformacao (Fabricacao de bebidas)",
    "13": "Industria de Transformacao (Fabricacao de produtos texteis)",
    "14": "Industria de Transformacao (Confeccao de artigos do vestuario e acessorios)",
    "15": "Industria de Transformacao (Preparacao de couros e fabricacao de calcados)",
    "16": "Industria de Transformacao (Fabricacao de produtos de madeira)",
    "17": "Industria de Transformacao (Fabricacao de celulose, papel e produtos de papel)",
    "18": "Industria de Transformacao (Impressao e reproducao de gravacoes)",
    "19": "Industria de Transformacao (Coque, derivados de petroleo e biocombustiveis)",
    "20": "Industria de Transformacao (Fabricacao de produtos quimicos)",
    "21": "Industria de Transformacao (Fabricacao de produtos farmoquimicos e farmaceuticos)",
    "22": "Industria de Transformacao (Fabricacao de produtos de borracha e de material plastico)",
    "23": "Industria de Transformacao (Fabricacao de produtos de minerais nao-metalicos)",
    "24": "Industria de Transformacao (Metalurgia)",
    "25": "Industria de Transformacao (Fabricacao de produtos de metal, exceto maquinas e equipamentos)",
    "26": "Industria de Transformacao (Fabricacao de equipamentos de informatica, produtos eletronicos e opticos)",
    "27": "Industria de Transformacao (Fabricacao de maquinas, aparelhos e materiais eletricos)",
    "28": "Industria de Transformacao (Fabricacao de maquinas e equipamentos)",
    "29": "Industria de Transformacao (Fabricacao de veiculos automotores, reboques e carrocerias)",
    "30": "Industria de Transformacao (Fabricacao de outros equipamentos de transporte)",
    "31": "Industria de Transformacao (Fabricacao de moveis)",
    "32": "Industria de Transformacao (Fabricacao de produtos diversos)",
    "33": "Industria de Transformacao (Manutencao, reparacao e instalacao de maquinas e equipamentos)",
    "35": "Eletricidade e Gas (Eletricidade, gas e outras utilidades)",
    "36": "Agua, Esgoto e Gestao de Residuos (Captacao, tratamento e distribuicao de agua)",
    "37": "Agua, Esgoto e Gestao de Residuos (Esgoto e atividades relacionadas)",
    "38": "Agua, Esgoto e Gestao de Residuos (Coleta, tratamento e disposicao de residuos)",
    "39": "Agua, Esgoto e Gestao de Residuos (Descontaminacao e outros servicos de gestao de residuos)",
    "41": "Construcao (Construcao de edificios)",
    "42": "Construcao (Obras de infraestrutura)",
    "43": "Construcao (Servicos especializados para construcao)",
    "45": "Comercio; Reparacao de Veiculos (Comercio e reparacao de veiculos automotores e motocicletas)",
    "46": "Comercio (Comercio por atacado, exceto veiculos automotores e motocicletas)",
    "47": "Comercio (Comercio varejista)",
    "49": "Transporte e Armazenagem (Transporte terrestre)",
    "50": "Transporte e Armazenagem (Transporte aquaviario)",
    "51": "Transporte e Armazenagem (Transporte aereo)",
    "52": "Transporte e Armazenagem (Armazenamento e atividades auxiliares dos transportes)",
    "53": "Transporte e Armazenagem (Correio e outras atividades de entrega)",
    "55": "Alojamento e Alimentacao (Alojamento)",
    "56": "Alojamento e Alimentacao (Alimentacao)",
    "58": "Informacao e Comunicacao (Edicao e edicao integrada a impressao)",
    "59": "Informacao e Comunicacao (Atividades cinematograficas, producao de videos e de programas de televisao)",
    "60": "Informacao e Comunicacao (Atividades de radio e de televisao)",
    "61": "Informacao e Comunicacao (Telecomunicacoes)",
    "62": "Informacao e Comunicacao (Atividades dos servicos de tecnologia da informacao)",
    "63": "Informacao e Comunicacao (Atividades de prestacao de servicos de informacao)",
    "64": "Atividades Financeiras, de Seguros e Servicos Relacionados (Atividades de servicos financeiros)",
    "65": "Atividades Financeiras, de Seguros e Servicos Relacionados (Seguros, resseguros, previdencia complementar e planos de saude)",
    "66": "Atividades Financeiras, de Seguros e Servicos Relacionados (Atividades auxiliares dos servicos financeiros, seguros e previdencia complementar)",
    "68": "Atividades Imobiliarias",
    "69": "Atividades Profissionais, Cientificas e Tecnicas (Atividades juridicas, de contabilidade e de auditoria)",
    "70": "Atividades Profissionais, Cientificas e Tecnicas (Atividades de sedes de empresas e consultoria em gestao empresarial)",
    "71": "Atividades Profissionais, Cientificas e Tecnicas (Servicos de arquitetura e engenharia; testes e analises tecnicas)",
    "72": "Atividades Profissionais, Cientificas e Tecnicas (Pesquisa e desenvolvimento cientifico)",
    "73": "Atividades Profissionais, Cientificas e Tecnicas (Publicidade e pesquisa de mercado)",
    "74": "Atividades Profissionais, Cientificas e Tecnicas (Outras atividades profissionais, cientificas e tecnicas)",
    "75": "Atividades Profissionais, Cientificas e Tecnicas (Atividades veterinarias)",
    "77": "Atividades Administrativas e Servicos Complementares (Alugueis nao-imobiliarios e gestao de ativos intangiveis)",
    "78": "Atividades Administrativas e Servicos Complementares (Selecao, agenciamento e locacao de mao de obra)",
    "80": "Atividades Administrativas e Servicos Complementares (Atividades de vigilancia, seguranca e investigacao)",
    "81": "Atividades Administrativas e Servicos Complementares (Servicos para edificios e atividades paisagisticas)",
    "82": "Atividades Administrativas e Servicos Complementares (Servicos de escritorio, apoio administrativo e outros servicos)",
    "85": "Educacao",
    "86": "Saude Humana e Servicos Sociais (Atividades de atencao a saude humana)",
    "87": "Saude Humana e Servicos Sociais (Atividades de atencao a saude humana integradas com assistencia social, prestadas em residencias coletivas)",
    "88": "Saude Humana e Servicos Sociais (Servicos de assistencia social sem alojamento)",
    "89": "Saude Humana e Servicos Sociais (Servicos de assistencia social sem alojamento)",
    "90": "Artes, Cultura, Esporte e Recreacao (Atividades artisticas, criativas e de espetaculos)",
    "93": "Artes, Cultura, Esporte e Recreacao (Atividades esportivas e de recreacao e lazer)",
    "94": "Outras Atividades de Servicos (Atividades de organizacoes associativas)",
    "95": "Outras Atividades de Servicos (Reparacao e manutencao de equipamentos de informatica e objetos pessoais)",
    "96": "Outras Atividades de Servicos (Outras atividades de servicos pessoais)",
}


@st.cache_resource
def get_model_bundle():
    return load_or_train_model(BASE_DIR)


@st.cache_data
def load_default_portfolio():
    scored, bundle = score_portfolio(base_dir=BASE_DIR)
    return scored, bundle


def api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def find_logo_path(with_name: bool) -> Path | None:
    exact_candidates = (
        [
            "prisma_logo_nome.png",
            "logo_prisma_nome.png",
            "prisma_com_nome.png",
            "prisma_wordmark.png",
            "prisma_logo_nome.jpeg",
            "logo_prisma_nome.jpeg",
            "prisma_com_nome.jpeg",
        ]
        if with_name
        else [
            "prisma_logo.png",
            "logo_prisma.png",
            "prisma_sem_nome.png",
            "prisma_mark.png",
            "prisma_logo.jpeg",
            "logo_prisma.jpeg",
            "prisma_sem_nome.jpeg",
        ]
    )
    for candidate in exact_candidates:
        for base_path in (BRANDING_DIR, BASE_DIR):
            path = base_path / candidate
            if path.exists():
                return path

    keyword = "nome" if with_name else "logo"
    for scan_dir in (BRANDING_DIR, BASE_DIR):
        if not scan_dir.exists():
            continue
        for path in scan_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                stem = path.stem.lower()
                if "prisma" in stem and keyword in stem:
                    return path

    for scan_dir in (BRANDING_DIR, BASE_DIR):
        if not scan_dir.exists():
            continue
        for path in scan_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                stem = path.stem.lower()
                if "prisma" in stem:
                    if with_name and "nome" in stem:
                        return path
                    if not with_name and "nome" not in stem:
                        return path
    return None


def image_to_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower().replace(".", "") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def format_currency_compact(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"R$ {value / 1_000_000:.1f} Mi"
    if abs(value) >= 1_000:
        return f"R$ {value / 1_000:.0f}k"
    return f"R$ {value:,.0f}".replace(",", ".")


def build_form_options(scored: pd.DataFrame) -> dict[str, list[str]]:
    def clean_sorted(values: pd.Series) -> list[str]:
        items = sorted({str(value).strip() for value in values.dropna().astype(str) if str(value).strip() and str(value).strip() != "Desconhecido"})
        return items

    cnae_sacado = clean_sorted(scored.get("cd_cnae_prin_sacado", pd.Series(dtype=object)))
    cnae_cedente = clean_sorted(scored.get("cd_cnae_prin_cedente", pd.Series(dtype=object)))

    return {
        "uf_options": UFS_BRASIL,
        "cnae_sacado_options": cnae_sacado,
        "cnae_cedente_options": cnae_cedente,
    }


def format_cnae_option(option: str) -> str:
    option = str(option).strip()
    if not option:
        return "Selecionar"
    numeric = option.split(".")[0]
    prefix = numeric[:2]
    description = CNAE_DIVISION_LABELS.get(prefix)
    if description:
        return f"{description} | CNAE {numeric}"
    return f"CNAE {numeric}"


def render_rule_banner(result: dict) -> None:
    approved = result["status"] == "Aprovado"
    border = "#1f9d71" if approved else "#FF5A5F"
    background = "rgba(0, 224, 150, 0.12)" if approved else "rgba(255, 90, 95, 0.12)"
    label = "Regua favoravel" if approved else "Regua de bloqueio"
    html = f"""
    <div style="
        margin: 0.6rem 0 1rem 0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid {border};
        background: {background};
        color: #f3fff9;
        font-size: 0.95rem;
    ">
        <strong style="color:{border};">{label}</strong><br>
        Regua aplicada: <strong>{result['regua_inadimplencia_aplicada']:.0f}%</strong> |
        Valor nominal analisado: <strong>{format_currency_compact(result['valor_nominal_analisado'])}</strong> |
        Ticket medio de referencia: <strong>{format_currency_compact(result['ticket_medio_referencia'])}</strong>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def build_dashboard_payload(scored: pd.DataFrame) -> dict:
    total_titulos = int(len(scored))
    volume_total = float(scored["vlr_nominal"].fillna(0).sum())
    aprovados = scored[scored["decisao_credito"] == "Aprovado"].copy()
    bloqueados = scored[scored["decisao_credito"] == "Negado"].copy()
    watchlist = scored[scored["decisao_credito"] != "Aprovado"].copy()

    blocked_volume = float(bloqueados["vlr_nominal"].fillna(0).sum())
    approved_volume = float(aprovados["vlr_nominal"].fillna(0).sum())
    observed_default = float(scored["alvo_inadimplencia"].mean() * 100) if "alvo_inadimplencia" in scored.columns else 0.0
    top10 = (
        aprovados.groupby("id_pagador", dropna=False)["vlr_nominal"].sum().sort_values(ascending=False).head(10).sum()
        if not aprovados.empty
        else 0.0
    )
    top10_pct = (top10 / approved_volume * 100) if approved_volume else 0.0
    daily_var = float((scored["vlr_nominal"].fillna(0) * scored["probabilidade_inadimplencia"]).quantile(0.95))
    rating_medio = float(aprovados["score_prisma"].mean()) if not aprovados.empty else float(scored["score_prisma"].mean())

    liq_df = scored.copy()
    liq_df["liq_bucket"] = pd.cut(
        liq_df["sacado_indice_liquidez_1m_sacado"].fillna(0).clip(0, 1),
        bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    liq_curve = liq_df.groupby("liq_bucket", observed=False)["probabilidade_inadimplencia"].mean().reset_index()
    liquidity_labels = [f"{bucket.right:.1f}" for bucket in liq_curve["liq_bucket"]]
    liquidity_values = [round(float(value * 100), 1) for value in liq_curve["probabilidade_inadimplencia"].fillna(0)]

    top_cedentes = (
        scored.groupby("id_beneficiario", dropna=False)
        .agg(risco=("probabilidade_inadimplencia", "mean"), volume=("vlr_nominal", "sum"))
        .sort_values(["risco", "volume"], ascending=[False, False])
        .head(5)
        .reset_index()
    )
    radar_labels = [f"Cedente {idx + 1}" for idx in range(len(top_cedentes))]
    radar_risk = [round(float(value * 100), 1) for value in top_cedentes["risco"]]
    radar_volume = [round(float(value / 1000), 1) for value in top_cedentes["volume"]]

    top_setores = (
        scored.groupby("cd_cnae_prin_sacado", dropna=False)
        .agg(risco=("probabilidade_inadimplencia", "mean"), qtd=("id_boleto", "count"))
        .sort_values(["risco", "qtd"], ascending=[False, False])
        .head(5)
        .reset_index()
    )
    sector_labels = [f"CNAE {int(value)}" if pd.notna(value) else "Nao informado" for value in top_setores["cd_cnae_prin_sacado"]]
    sector_values = [round(float(value * 100), 1) for value in top_setores["risco"]]
    sector_colors = ["#FF3B30" if value >= 15 else "#00E096" for value in sector_values]

    alerts_rows = scored.sort_values(["probabilidade_inadimplencia", "vlr_nominal"], ascending=[False, False]).head(7)
    alerts = []
    for _, row in alerts_rows.iterrows():
        status = "red" if row["decisao_credito"] == "Negado" else "green"
        sacado = str(row.get("id_pagador", "Sem ID"))
        alerts.append(
            {
                "s": status,
                "cnpj": f"{sacado[:8]}***",
                "m": row["motivos_risco"][0],
                "l": f"{float(row.get('sacado_indice_liquidez_1m_sacado', 0)):.2f}",
            }
        )

    return {
        "protected_capital": format_currency_compact(blocked_volume),
        "optimized_return": f"+{(approved_volume / volume_total * 1.7):.2f}%" if volume_total else "+0.00%",
        "default_rate": f"{observed_default:.1f}%",
        "var_95": format_currency_compact(daily_var),
        "entry_count": f"{total_titulos:,}".replace(",", "."),
        "entry_volume": format_currency_compact(volume_total),
        "blocked_count": f"{len(bloqueados):,}".replace(",", "."),
        "blocked_pct": round((len(bloqueados) / total_titulos * 100), 1) if total_titulos else 0.0,
        "approved_count": f"{len(aprovados):,}".replace(",", "."),
        "approved_volume": format_currency_compact(approved_volume),
        "rejection_rate": f"{(len(bloqueados) / total_titulos * 100):.0f}%" if total_titulos else "0%",
        "watchlist_count": int(len(watchlist)),
        "rating_medio": f"{rating_medio:.0f} / 1000",
        "concentration_top10": f"{top10_pct:.0f}%",
        "liquidity_labels": liquidity_labels,
        "liquidity_values": liquidity_values,
        "radar_labels": radar_labels,
        "radar_risk": radar_risk,
        "radar_volume": radar_volume,
        "alerts": alerts,
        "sector_labels": sector_labels,
        "sector_values": sector_values,
        "sector_colors": sector_colors,
    }


def _format_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div style='color:#8f9ba3;font-size:0.9rem;'>Sem dados disponiveis.</div>"
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table class='alert-table'><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def build_analytic_payload(scored: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    volume_aprovado = float(scored.loc[scored["decisao_credito"] == "Aprovado", "vlr_nominal"].sum())
    volume_negado = float(scored.loc[scored["decisao_credito"] == "Negado", "vlr_nominal"].sum())

    volume_decisao = (
        scored.groupby("decisao_credito", dropna=False)["vlr_nominal"]
        .sum()
        .reindex(["Aprovado", "Negado"], fill_value=0)
    )
    decision_labels = list(volume_decisao.index)
    decision_values = [round(float(v / 1_000_000), 2) for v in volume_decisao.values]

    delay_curve = bundle.get("delay_curve", {})
    delay_days = delay_curve.get("dias", [])
    delay_values = delay_curve.get("curva_percentual", [])
    if len(delay_days) > 120:
        idxs = np.linspace(0, len(delay_days) - 1, 120).astype(int)
        delay_days = [round(float(delay_days[i]), 1) for i in idxs]
        delay_values = [round(float(delay_values[i]), 2) for i in idxs]
    else:
        delay_days = [round(float(v), 1) for v in delay_days]
        delay_values = [round(float(v), 2) for v in delay_values]

    evolution_rows = []
    for key, label in (
        ("baseline_interno", "Modelo Interno"),
        ("modelo_enriquecido", "Modelo Enriquecido"),
    ):
        metrics = bundle.get("model_evolution", {}).get(key, {}).get("metrics", {})
        evolution_rows.append(
            {
                "Etapa": label,
                "Acuracia": f"{metrics.get('accuracy', 0) * 100:.2f}%",
                "Recall": f"{metrics.get('recall', 0) * 100:.2f}%",
                "Precisao": f"{metrics.get('precision', 0) * 100:.2f}%",
                "PR AUC": f"{metrics.get('pr_auc', 0):.3f}",
            }
        )
    evolution_df = pd.DataFrame(evolution_rows)

    feature_importance = pd.DataFrame(bundle.get("feature_importance", []))
    if not feature_importance.empty:
        feature_importance["Variavel"] = feature_importance["feature"].str.replace(r"^(num|cat)__", "", regex=True)
        feature_importance["Importancia"] = feature_importance["importance"].map(lambda v: f"{float(v):.4f}")
        feature_importance = feature_importance[["Variavel", "Importancia"]]

    cedente_view = (
        scored[[
            "id_beneficiario",
            "cedente_volume_historico",
            "cedente_taxa_default",
            "cedente_ticket_medio",
            "score_cedente_proprio",
        ]]
        .drop_duplicates("id_beneficiario")
        .copy()
    )
    cedente_view["cedente_taxa_default"] = cedente_view["cedente_taxa_default"].fillna(0)
    cedentes_alerta = cedente_view[
        (cedente_view["cedente_volume_historico"].fillna(0) > 5)
        & (cedente_view["cedente_taxa_default"] > 0.2)
    ].copy()

    top_fin = (
        scored.groupby("id_beneficiario", dropna=False)
        .agg(
            volume_total=("vlr_nominal", "sum"),
            ticket_medio=("vlr_nominal", "mean"),
            taxa_risco=("probabilidade_inadimplencia", "mean"),
            qtd_boletos=("id_boleto", "count"),
        )
        .reset_index()
    )
    top_volume = top_fin.sort_values("volume_total", ascending=False).head(10).copy()
    top_risk = top_fin.sort_values(["taxa_risco", "volume_total"], ascending=[False, False]).head(10).copy()
    top_ticket = top_fin.sort_values("ticket_medio", ascending=False).head(10).copy()

    for df in (top_volume, top_risk, top_ticket):
        df["volume_total"] = df["volume_total"].map(format_currency_compact)
        df["ticket_medio"] = df["ticket_medio"].map(format_currency_compact)
        df["taxa_risco"] = df["taxa_risco"].map(lambda v: f"{float(v) * 100:.2f}%")

    if not cedentes_alerta.empty:
        cedentes_alerta["cedente_taxa_default"] = cedentes_alerta["cedente_taxa_default"].map(lambda v: f"{float(v) * 100:.2f}%")
        cedentes_alerta["cedente_ticket_medio"] = cedentes_alerta["cedente_ticket_medio"].map(format_currency_compact)
        cedentes_alerta["score_cedente_proprio"] = cedentes_alerta["score_cedente_proprio"].map(lambda v: f"{float(v):.0f}")

    cenarios = [
        ("Ativo saudavel", {
            "vlr_nominal": 15000.0, "tipo_especie": "DM DUPLICATA MERCANTIL",
            "referencia_pagador": "cenario-saudavel-pagador", "referencia_beneficiario": "cenario-saudavel-cedente",
            "media_atraso_dias_sacado": 2.0, "score_materialidade_v2_sacado": 850.0,
            "score_quantidade_v2_sacado": 880.0, "sacado_indice_liquidez_1m_sacado": 0.85,
            "media_atraso_dias_cedente": 2.0, "score_materialidade_v2_cedente": 850.0,
            "score_quantidade_v2_cedente": 860.0, "cedente_indice_liquidez_1m_cedente": 0.82,
            "indicador_liquidez_quantitativo_3m_cedente": 0.80,
        }),
        ("Ativo estressado", {
            "vlr_nominal": 80000.0, "tipo_especie": "DM DUPLICATA MERCANTIL",
            "referencia_pagador": "cenario-ruim-pagador", "referencia_beneficiario": "cenario-ruim-cedente",
            "media_atraso_dias_sacado": 90.0, "score_materialidade_v2_sacado": 120.0,
            "score_quantidade_v2_sacado": 80.0, "sacado_indice_liquidez_1m_sacado": 0.10,
            "media_atraso_dias_cedente": 40.0, "score_materialidade_v2_cedente": 200.0,
            "score_quantidade_v2_cedente": 180.0, "cedente_indice_liquidez_1m_cedente": 0.18,
            "indicador_liquidez_quantitativo_3m_cedente": 0.15,
        }),
        ("Ativo limitrofe", {
            "vlr_nominal": 18000.0, "tipo_especie": "DM DUPLICATA MERCANTIL",
            "referencia_pagador": "cenario-bbb-pagador", "referencia_beneficiario": "cenario-bbb-cedente",
            "media_atraso_dias_sacado": 12.0, "score_materialidade_v2_sacado": 580.0,
            "score_quantidade_v2_sacado": 650.0, "sacado_indice_liquidez_1m_sacado": 0.68,
            "media_atraso_dias_cedente": 7.0, "score_materialidade_v2_cedente": 720.0,
            "score_quantidade_v2_cedente": 760.0, "cedente_indice_liquidez_1m_cedente": 0.75,
            "indicador_liquidez_quantitativo_3m_cedente": 0.72,
        }),
    ]
    scenario_rows = []
    for nome, payload in cenarios:
        result = predict_from_payload(payload, bundle)
        scenario_rows.append({
            "Cenario": nome,
            "Probabilidade": f"{result['probabilidade_inadimplencia']:.2f}%",
            "Decisao": result["status"],
            "Regua": f"{result['regua_inadimplencia_aplicada']:.0f}%",
            "Acao": result["acao_recomendada"],
        })

    return {
        "approved_volume": format_currency_compact(volume_aprovado),
        "denied_volume": format_currency_compact(volume_negado),
        "cedentes_alerta": str(len(cedentes_alerta)),
        "share_5d": f"{delay_curve.get('share_ate_threshold', 0) * 100:.1f}%",
        "delay_caption": f"{delay_curve.get('share_ate_threshold', 0) * 100:.2f}% dos boletos pagos ficaram em ate {delay_curve.get('threshold_dias', 5)} dias de atraso.",
        "decision_labels": decision_labels,
        "decision_values": decision_values,
        "delay_labels": delay_days,
        "delay_values": delay_values,
        "evolution_table": _format_html_table(evolution_df),
        "importance_table": _format_html_table(feature_importance),
        "top_volume_table": _format_html_table(top_volume.rename(columns={"id_beneficiario": "Cedente", "volume_total": "Volume Total", "ticket_medio": "Ticket Medio", "taxa_risco": "Taxa de Risco", "qtd_boletos": "Qtd Boletos"})),
        "top_risk_table": _format_html_table(top_risk.rename(columns={"id_beneficiario": "Cedente", "volume_total": "Volume Total", "ticket_medio": "Ticket Medio", "taxa_risco": "Taxa de Risco", "qtd_boletos": "Qtd Boletos"})),
        "top_ticket_table": _format_html_table(top_ticket.rename(columns={"id_beneficiario": "Cedente", "volume_total": "Volume Total", "ticket_medio": "Ticket Medio", "taxa_risco": "Taxa de Risco", "qtd_boletos": "Qtd Boletos"})),
        "alert_table": _format_html_table(cedentes_alerta.rename(columns={"id_beneficiario": "Cedente", "cedente_volume_historico": "Volume Historico", "cedente_taxa_default": "Taxa Default", "cedente_ticket_medio": "Ticket Medio", "score_cedente_proprio": "Score Cedente"})),
        "scenarios_table": _format_html_table(pd.DataFrame(scenario_rows)),
    }


def render_header() -> None:
    logo_with_name = image_to_data_uri(find_logo_path(with_name=True))
    logo_without_name = image_to_data_uri(find_logo_path(with_name=False))
    top_logo = logo_with_name or logo_without_name
    header_visual = (
        f'<div class="hero-logo-wrap hero-logo-wrap-wide"><img src="{top_logo}" class="hero-logo hero-logo-wide" alt="Logo PRISMA"></div>'
        if top_logo
        else '<div class="hero-logo hero-fallback">PRISMA</div>'
    )
    header_html = """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(11, 110, 79, 0.28), transparent 30%),
                radial-gradient(circle at left center, rgba(5, 59, 80, 0.22), transparent 28%),
                linear-gradient(180deg, #08151d 0%, #0d1117 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            border: 1px solid rgba(64, 180, 145, 0.25);
            background: linear-gradient(135deg, rgba(15, 40, 52, 0.95), rgba(10, 20, 28, 0.95));
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.28);
        }
        .hero-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }
        .hero-logo-wrap {
            width: 72px;
            height: 72px;
            flex: 0 0 72px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .hero-logo-wrap-wide {
            width: 220px;
            height: 88px;
            flex: 0 0 220px;
            justify-content: flex-start;
        }
        .hero-logo {
            max-width: 72px;
            max-height: 72px;
            width: auto;
            height: auto;
            object-fit: contain;
            filter: drop-shadow(0 0 14px rgba(0, 224, 150, 0.25));
        }
        .hero-logo-wide {
            max-width: 210px;
            max-height: 84px;
        }
        .hero-fallback {
            min-width: 150px;
            min-height: 58px;
            padding: 0.8rem 1.2rem;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #10e3ae, #05b67a);
            color: white;
            font-weight: 700;
            letter-spacing: 0.06em;
        }
        .hero h1 {
            margin: 0;
            color: #7af0c1;
            letter-spacing: 0.03em;
            font-size: 2rem;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            color: #d8f1e7;
            max-width: 960px;
        }
        .chip {
            display: inline-block;
            margin-top: 0.8rem;
            margin-right: 0.6rem;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.82rem;
            border: 1px solid rgba(122, 240, 193, 0.32);
            color: #7af0c1;
            background: rgba(122, 240, 193, 0.08);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background: #00E096 !important;
        }
        .stTabs [data-baseweb="tab-list"] > div {
            background: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00E096 !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        button[data-baseweb="tab"] {
            color: #d6e7e0 !important;
            border-bottom: none !important;
        }
        div[data-baseweb="tab-list"] {
            border-bottom: 1px solid rgba(124, 138, 150, 0.18) !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] > div {
            background-color: rgba(35, 37, 48, 0.95) !important;
        }
        .stSlider {
            background: transparent !important;
        }
        .stSlider > div {
            background: transparent !important;
        }
        .stSlider [data-baseweb="slider"] {
            background: transparent !important;
        }
        .stSlider [data-baseweb="slider"] > div {
            background: transparent !important;
        }
        .stSlider [data-baseweb="slider"] > div > div > div {
            background: #ffffff !important;
        }
        .stSlider [data-baseweb="slider"]::before,
        .stSlider [data-baseweb="slider"]::after {
            background: transparent !important;
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider [role="slider"],
        .stSlider [role="slider"]:focus,
        .stSlider [role="slider"]:hover {
            background: #ffffff !important;
            border: 2px solid #d8dde3 !important;
            box-shadow: none !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            color: #00E096 !important;
        }
        .stSlider [data-testid="stThumbValue"] * {
            color: #00E096 !important;
        }
        .stSlider p,
        .stSlider span {
            color: #ffffff !important;
        }
        .stSlider [data-testid="stSliderTickBarMin"],
        .stSlider [data-testid="stSliderTickBarMax"] {
            background: transparent !important;
            color: #00E096 !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider [data-testid="stSliderTickBarMin"] *,
        .stSlider [data-testid="stSliderTickBarMax"] * {
            background: transparent !important;
            color: #00E096 !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider div[data-testid="stSliderTickBarMin"],
        .stSlider div[data-testid="stSliderTickBarMax"],
        .stSlider div[data-testid="stSliderTickBarMin"] > *,
        .stSlider div[data-testid="stSliderTickBarMax"] > * {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider div[data-testid="stSliderTickBarMin"] p,
        .stSlider div[data-testid="stSliderTickBarMax"] p,
        .stSlider div[data-testid="stSliderTickBarMin"] span,
        .stSlider div[data-testid="stSliderTickBarMax"] span,
        .stSlider [data-testid="stThumbValue"] p,
        .stSlider [data-testid="stThumbValue"] span {
            color: #00E096 !important;
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider [data-testid="stSliderTickBarMin"] label,
        .stSlider [data-testid="stSliderTickBarMax"] label,
        .stSlider [data-testid="stSliderTickBarMin"] div,
        .stSlider [data-testid="stSliderTickBarMax"] div {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #00E096 !important;
        }
        .stSlider [data-testid="stSliderTickBarMin"],
        .stSlider [data-testid="stSliderTickBarMax"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider [data-testid="stSliderTickBarMin"]::before,
        .stSlider [data-testid="stSliderTickBarMin"]::after,
        .stSlider [data-testid="stSliderTickBarMax"]::before,
        .stSlider [data-testid="stSliderTickBarMax"]::after {
            content: none !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider div[data-testid="stTickBar"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stSlider [data-testid="stThumbValue"],
        .stSlider [data-testid="stThumbValue"] *,
        .stSlider [data-testid="stThumbValue"] p,
        .stSlider [data-testid="stThumbValue"] span,
        .stSlider [data-testid="stThumbValue"] div,
        .stSlider [data-testid="stThumbValue"] label,
        .stSlider [data-testid="stSliderThumbValue"] p,
        .stSlider [data-testid="stSliderThumbValue"] .stMarkdownContainer p {
            color: #00E096 !important;
        }
        .stSlider [data-testid="stSliderTickBar"] p,
        .stSlider [data-testid="stSliderTickBar"] .stMarkdownContainer p,
        .stSlider [data-testid="stSliderTickBarMin"] p,
        .stSlider [data-testid="stSliderTickBarMax"] p {
            color: #ffffff !important;
        }
        .stMarkdownContainer,
        .stMarkdownContainer *,
        .stMarkdownContainer p,
        .stMarkdownContainer span,
        .stMarkdownContainer label,
        .stMarkdownContainer div {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
        </style>
        <div class="hero">
            <div class="hero-brand">
                __HEADER_VISUAL__
                <h1>PRISMA | Plataforma de Risco para FIDCs</h1>
            </div>
            <p>
                Monitoramento da carteira e simulador manual.
            </p>
            <span class="chip">Analise Estrutural</span>
            <span class="chip">Monitoramento de Carteira</span>
            <span class="chip">Score PRISMA</span>
        </div>
    """
    st.markdown(header_html.replace("__HEADER_VISUAL__", header_visual), unsafe_allow_html=True)


def render_dashboard_html(scored: pd.DataFrame) -> None:
    try:
        with open(DASHBOARD_DIR / "index.html", "r", encoding="utf-8") as file:
            html_dashboard = file.read()
        payload = build_dashboard_payload(scored)
        bundle = get_model_bundle()
        analytic = build_analytic_payload(scored, bundle)
        logo_with_name = image_to_data_uri(find_logo_path(with_name=True))
        logo_without_name = image_to_data_uri(find_logo_path(with_name=False))
        dashboard_logo = logo_without_name or logo_with_name
        if dashboard_logo:
            logo_markup = f'<div class="logo-wrap"><img src="{dashboard_logo}" alt="Logo Prisma" class="logo-img"></div>'
        else:
            logo_markup = '<div class="logo-fallback">PRISMA</div>'

        html_dashboard = html_dashboard.replace(
            """        .logo-img {
            height: 50px;
            width: auto;
            border-radius: 4px;
            filter: drop-shadow(0 0 8px var(--brand-green-shadow));
        }""",
            """        .logo-img {
            max-width: 68px;
            max-height: 68px;
            width: auto;
            height: auto;
            object-fit: contain;
            filter: drop-shadow(0 0 8px var(--brand-green-shadow));
        }
        .logo-wrap {
            width: 72px;
            height: 72px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .logo-fallback {
            width: 72px;
            height: 72px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.72rem;
            text-align: center;
            background: linear-gradient(135deg, #10e3ae, #05b67a);
            box-shadow: 0 0 18px var(--brand-green-shadow);
        }""",
            1,
        )
        html_dashboard = html_dashboard.replace(
            """            <!-- LOGO PRISMA (PLACEHOLDER) -->
            <img src="https://placehold.co/180x60/00E096/ffffff?text=LOGO+PRISMA" alt="Logo Prisma" class="logo-img">
            
            <div style="text-align: left;">
                <h1>PRISMA <span style="font-size: 1rem; color: var(--brand-green); opacity: 0.7;">| INTELIGÃŠNCIA</span></h1>
            </div>""",
            f"""            {logo_markup}""",
            1,
        )
        html_dashboard = html_dashboard.replace(
            'src="prisma_logo.png"',
            f'src="{dashboard_logo}"' if dashboard_logo else 'src="prisma_logo.png"',
            1,
        )
        html_dashboard = html_dashboard.replace("R$ 5.2 Mi", payload["protected_capital"], 1)
        html_dashboard = html_dashboard.replace("+1.50%", payload["optimized_return"], 1)
        html_dashboard = html_dashboard.replace("7.1%", payload["default_rate"], 1)
        html_dashboard = html_dashboard.replace("R$ 450k", payload["var_95"], 1)
        html_dashboard = html_dashboard.replace("7,118 Ativos", f"{payload['entry_count']} Ativos", 1)
        html_dashboard = html_dashboard.replace("R$ 166 Milhões", payload["entry_volume"], 1)
        html_dashboard = html_dashboard.replace("2,118 Rejeitados (30%)", f"{payload['blocked_count']} Rejeitados ({payload['blocked_pct']}%)", 1)
        html_dashboard = html_dashboard.replace("5,000 Ativos", f"{payload['approved_count']} Ativos", 1)
        html_dashboard = html_dashboard.replace("R$ 115 Milhões (Líquido)", payload["approved_volume"], 1)
        html_dashboard = html_dashboard.replace("31%", payload["rejection_rate"], 1)
        html_dashboard = html_dashboard.replace("> 25%", "threshold dinâmico do modelo", 1)
        html_dashboard = html_dashboard.replace("> 50%", "probabilidade acima da régua PRISMA", 1)
        html_dashboard = html_dashboard.replace("> 25%", "probabilidade acima da régua PRISMA", 1)
        html_dashboard = html_dashboard.replace("5", str(payload["watchlist_count"]), 1)
        html_dashboard = html_dashboard.replace("945 / 1000", payload["rating_medio"], 1)
        html_dashboard = html_dashboard.replace("12%", payload["concentration_top10"], 1)
        html_dashboard = html_dashboard.replace(
            "const alertsData = [\n            { s: 'red', cnpj: '12.993.***/0001-45', m: 'Queda Brusca Liquidez (-40%)', l: '0.12' },\n            { s: 'red', cnpj: '08.112.***/0001-12', m: 'InadimplÃªncia Recente', l: '0.18' },\n            { s: 'yellow', cnpj: '33.554.***/0001-90', m: 'Score Materialidade em Queda', l: '0.45' },\n            { s: 'red', cnpj: '04.221.***/0001-33', m: 'Setor em Crise (Conservas)', l: '0.29' },\n            { s: 'yellow', cnpj: '10.888.***/0001-88', m: 'Atraso TÃ©cnico (3 dias)', l: '0.55' },\n            { s: 'green', cnpj: '55.123.***/0001-22', m: 'RecuperaÃ§Ã£o de Score', l: '0.71' },\n            { s: 'red', cnpj: '99.123.***/0001-00', m: 'Pedido de RecuperaÃ§Ã£o Judicial', l: '0.05' },\n        ];",
            f"const alertsData = {json.dumps(payload['alerts'], ensure_ascii=False)};",
        )
        html_dashboard = html_dashboard.replace(
            "labels: ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],",
            f"labels: {json.dumps(payload['liquidity_labels'])},",
        )
        html_dashboard = html_dashboard.replace(
            "data: [82.0, 65.5, 35.4, 29.4, 19.0, 22.3, 13.1, 9.4, 5.3, 1.9],",
            f"data: {json.dumps(payload['liquidity_values'])},",
            1,
        )
        html_dashboard = html_dashboard.replace(
            "labels: ['Cedente A', 'Cedente B', 'Cedente C', 'Cedente D', 'Cedente E'],",
            f"labels: {json.dumps(payload['radar_labels'])},",
        )
        html_dashboard = html_dashboard.replace(
            "data: [89.5, 83.3, 76.8, 66.6, 57.1],",
            f"data: {json.dumps(payload['radar_risk'])},",
            1,
        )
        html_dashboard = html_dashboard.replace(
            "data: [10, 29, 64, 4, 5],",
            f"data: {json.dumps(payload['radar_volume'])},",
            1,
        )
        html_dashboard = html_dashboard.replace(
            "labels: ['Ind. Conservas', 'Atac. Medicamentos', 'Ind. TÃªxtil', 'Proc. Alimentos', 'Transporte Cargas'],",
            f"labels: {json.dumps(payload['sector_labels'], ensure_ascii=False)},",
        )
        html_dashboard = html_dashboard.replace(
            "data: [77.4, 74.3, 55.6, 52.6, 43.9],",
            f"data: {json.dumps(payload['sector_values'])},",
            1,
        )
        html_dashboard = html_dashboard.replace(
            "backgroundColor: ['#FF3B30', '#FF3B30', BRAND_GREEN, BRAND_GREEN, '#238666'],",
            f"backgroundColor: {json.dumps(payload['sector_colors'])},",
        )
        html_dashboard = html_dashboard.replace("__AN_APPROVED_VOLUME__", analytic["approved_volume"])
        html_dashboard = html_dashboard.replace("__AN_DENIED_VOLUME__", analytic["denied_volume"])
        html_dashboard = html_dashboard.replace("__AN_CEDENTES_ALERTA__", analytic["cedentes_alerta"])
        html_dashboard = html_dashboard.replace("__AN_SHARE_5D__", analytic["share_5d"])
        html_dashboard = html_dashboard.replace("__AN_DELAY_CAPTION__", analytic["delay_caption"])
        html_dashboard = html_dashboard.replace("__AN_EVOLUTION_TABLE__", analytic["evolution_table"])
        html_dashboard = html_dashboard.replace("__AN_IMPORTANCE_TABLE__", analytic["importance_table"])
        html_dashboard = html_dashboard.replace("__AN_TOP_VOLUME_TABLE__", analytic["top_volume_table"])
        html_dashboard = html_dashboard.replace("__AN_TOP_RISK_TABLE__", analytic["top_risk_table"])
        html_dashboard = html_dashboard.replace("__AN_TOP_TICKET_TABLE__", analytic["top_ticket_table"])
        html_dashboard = html_dashboard.replace("__AN_ALERT_TABLE__", analytic["alert_table"])
        html_dashboard = html_dashboard.replace("__AN_SCENARIOS_TABLE__", analytic["scenarios_table"])
        html_dashboard = html_dashboard.replace("__AN_DECISION_LABELS__", json.dumps(analytic["decision_labels"], ensure_ascii=False))
        html_dashboard = html_dashboard.replace("__AN_DECISION_VALUES__", json.dumps(analytic["decision_values"]))
        html_dashboard = html_dashboard.replace("__AN_DELAY_LABELS__", json.dumps(analytic["delay_labels"]))
        html_dashboard = html_dashboard.replace("__AN_DELAY_VALUES__", json.dumps(analytic["delay_values"]))
        components.html(html_dashboard, height=2200, scrolling=True)
    except FileNotFoundError:
        st.error("Arquivo 'index.html' não encontrado.")


def render_dashboard_insights(scored: pd.DataFrame, bundle: dict[str, Any]) -> None:
    st.markdown("### Inteligencia Analitica do PRISMA")

    volume_decisao = (
        scored.groupby("decisao_credito", dropna=False)["vlr_nominal"]
        .sum()
        .sort_values(ascending=False)
        .rename("volume")
        .reset_index()
    )
    volume_decisao["volume_milhoes"] = volume_decisao["volume"] / 1_000_000

    cedente_columns = [
        "id_beneficiario",
        "cedente_volume_historico",
        "cedente_taxa_default",
        "cedente_ticket_medio",
        "score_cedente_proprio",
    ]
    cedente_view = scored[cedente_columns].drop_duplicates("id_beneficiario").copy()
    cedente_view["cedente_taxa_default"] = cedente_view["cedente_taxa_default"].fillna(0)

    top_volume_financeiro = (
        scored.groupby("id_beneficiario", dropna=False)
        .agg(
            volume_total=("vlr_nominal", "sum"),
            ticket_medio=("vlr_nominal", "mean"),
            taxa_risco=("probabilidade_inadimplencia", "mean"),
            qtd_boletos=("id_boleto", "count"),
        )
        .reset_index()
        .sort_values("volume_total", ascending=False)
    )
    top_ticket = top_volume_financeiro.sort_values("ticket_medio", ascending=False).head(10)
    top_risco = top_volume_financeiro.sort_values(["taxa_risco", "volume_total"], ascending=[False, False]).head(10)
    cedentes_alerta = cedente_view[
        (cedente_view["cedente_volume_historico"].fillna(0) > 5)
        & (cedente_view["cedente_taxa_default"] > 0.2)
    ].sort_values(["cedente_taxa_default", "cedente_volume_historico"], ascending=[False, False])

    comparative_rows = []
    evolution = bundle.get("model_evolution", {})
    for key, label in (
        ("baseline_interno", "Modelo Interno"),
        ("modelo_enriquecido", "Modelo Enriquecido"),
    ):
        metrics = evolution.get(key, {}).get("metrics", {})
        comparative_rows.append(
            {
                "Etapa do Modelo": label,
                "Acuracia Geral": f"{metrics.get('accuracy', 0) * 100:.2f}%",
                "Deteccao de Risco (Recall)": f"{metrics.get('recall', 0) * 100:.2f}%",
                "Precisao (Acerto no Risco)": f"{metrics.get('precision', 0) * 100:.2f}%",
                "ROC AUC": f"{metrics.get('roc_auc', 0):.3f}",
                "PR AUC": f"{metrics.get('pr_auc', 0):.3f}",
            }
        )
    evolution_df = pd.DataFrame(comparative_rows)

    delay_curve = bundle.get("delay_curve", {})
    delay_df = pd.DataFrame(
        {
            "dias_atraso": delay_curve.get("dias", []),
            "curva_percentual": delay_curve.get("curva_percentual", []),
        }
    )
    if len(delay_df) > 200:
        delay_df = delay_df.iloc[np.linspace(0, len(delay_df) - 1, 200).astype(int)]

    feature_importance = pd.DataFrame(bundle.get("feature_importance", []))
    if not feature_importance.empty:
        feature_importance["feature_label"] = feature_importance["feature"].str.replace(
            r"^(num|cat)__", "", regex=True
        )

    cenarios = [
        (
            "Ativo saudavel",
            {
                "vlr_nominal": 15000.0,
                "tipo_especie": "DM DUPLICATA MERCANTIL",
                "referencia_pagador": "cenario-saudavel-pagador",
                "referencia_beneficiario": "cenario-saudavel-cedente",
                "media_atraso_dias_sacado": 2.0,
                "score_materialidade_v2_sacado": 850.0,
                "score_quantidade_v2_sacado": 880.0,
                "sacado_indice_liquidez_1m_sacado": 0.85,
                "media_atraso_dias_cedente": 2.0,
                "score_materialidade_v2_cedente": 850.0,
                "score_quantidade_v2_cedente": 860.0,
                "cedente_indice_liquidez_1m_cedente": 0.82,
                "indicador_liquidez_quantitativo_3m_cedente": 0.80,
            },
        ),
        (
            "Ativo estressado",
            {
                "vlr_nominal": 80000.0,
                "tipo_especie": "DM DUPLICATA MERCANTIL",
                "referencia_pagador": "cenario-ruim-pagador",
                "referencia_beneficiario": "cenario-ruim-cedente",
                "media_atraso_dias_sacado": 90.0,
                "score_materialidade_v2_sacado": 120.0,
                "score_quantidade_v2_sacado": 80.0,
                "sacado_indice_liquidez_1m_sacado": 0.10,
                "media_atraso_dias_cedente": 40.0,
                "score_materialidade_v2_cedente": 200.0,
                "score_quantidade_v2_cedente": 180.0,
                "cedente_indice_liquidez_1m_cedente": 0.18,
                "indicador_liquidez_quantitativo_3m_cedente": 0.15,
            },
        ),
        (
            "Ativo limitrofe",
            {
                "vlr_nominal": 18000.0,
                "tipo_especie": "DM DUPLICATA MERCANTIL",
                "referencia_pagador": "cenario-bbb-pagador",
                "referencia_beneficiario": "cenario-bbb-cedente",
                "media_atraso_dias_sacado": 12.0,
                "score_materialidade_v2_sacado": 580.0,
                "score_quantidade_v2_sacado": 650.0,
                "sacado_indice_liquidez_1m_sacado": 0.68,
                "media_atraso_dias_cedente": 7.0,
                "score_materialidade_v2_cedente": 720.0,
                "score_quantidade_v2_cedente": 760.0,
                "cedente_indice_liquidez_1m_cedente": 0.75,
                "indicador_liquidez_quantitativo_3m_cedente": 0.72,
            },
        ),
    ]
    scenarios_df = pd.DataFrame(
        [
            {
                "Cenario": nome,
                "Probabilidade": f"{result['probabilidade_inadimplencia']:.2f}%",
                "Decisao": result["status"],
                "Regua": f"{result['regua_inadimplencia_aplicada']:.0f}%",
                "Acao": result["acao_recomendada"],
            }
            for nome, payload in cenarios
            for result in [predict_from_payload(payload, bundle)]
        ]
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Volume por Classe de Decisao")
        st.bar_chart(volume_decisao.set_index("decisao_credito")["volume_milhoes"])
    with right:
        st.markdown("#### Curva Cumulativa de Atrasos")
        if not delay_df.empty:
            st.line_chart(delay_df.set_index("dias_atraso")["curva_percentual"])
            st.caption(
                f"{delay_curve.get('share_ate_threshold', 0) * 100:.2f}% dos boletos pagos ficaram em ate "
                f"{delay_curve.get('threshold_dias', 5)} dias de atraso."
            )
        else:
            st.caption("Sem atraso observavel suficiente para construir a curva.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Comparativo Baseline Interno vs Modelo Enriquecido")
        st.dataframe(evolution_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Variaveis Mais Relevantes do Modelo")
        if not feature_importance.empty:
            st.dataframe(
                feature_importance[["feature_label", "importance"]].rename(
                    columns={"feature_label": "Variavel", "importance": "Importancia"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Importancias de variaveis indisponiveis.")

    st.markdown("#### Inteligencia do Cedente")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Cedentes monitorados", f"{cedente_view['id_beneficiario'].nunique():,}".replace(",", "."))
    col_b.metric("Cedentes alerta", f"{len(cedentes_alerta):,}".replace(",", "."))
    col_c.metric("Maior taxa de default", f"{cedente_view['cedente_taxa_default'].max() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top Cedentes por Volume")
        st.dataframe(
            top_volume_financeiro.head(10).rename(
                columns={
                    "id_beneficiario": "Cedente",
                    "volume_total": "Volume Total",
                    "ticket_medio": "Ticket Medio",
                    "taxa_risco": "Taxa de Risco",
                    "qtd_boletos": "Qtd Boletos",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("#### Top Cedentes por Taxa de Risco")
        st.dataframe(
            top_risco.rename(
                columns={
                    "id_beneficiario": "Cedente",
                    "volume_total": "Volume Total",
                    "ticket_medio": "Ticket Medio",
                    "taxa_risco": "Taxa de Risco",
                    "qtd_boletos": "Qtd Boletos",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Ranking de Ticket Medio por Cedente")
        st.dataframe(
            top_ticket.rename(
                columns={
                    "id_beneficiario": "Cedente",
                    "volume_total": "Volume Total",
                    "ticket_medio": "Ticket Medio",
                    "taxa_risco": "Taxa de Risco",
                    "qtd_boletos": "Qtd Boletos",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("#### Cedentes Alerta: Volume > 5 e Default > 20%")
        if cedentes_alerta.empty:
            st.caption("Nenhum cedente em alerta com a regua atual.")
        else:
            st.dataframe(
                cedentes_alerta.rename(
                    columns={
                        "id_beneficiario": "Cedente",
                        "cedente_volume_historico": "Volume Historico",
                        "cedente_taxa_default": "Taxa Default",
                        "cedente_ticket_medio": "Ticket Medio",
                        "score_cedente_proprio": "Score Cedente",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("#### Cenarios de Teste do Simulador")
    st.dataframe(scenarios_df, use_container_width=True, hide_index=True)


def render_simulator(bundle: dict, form_options: dict[str, list[str]]) -> None:
    st.markdown("### Simulador de Risco")
    defaults = bundle["defaults"]
    api_online = api_health()
    st.caption(
        "API online" if api_online else "API offline: usando o mesmo motor de risco localmente no Streamlit."
    )

    with st.form("simulator_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            vlr_nominal = st.number_input("Valor nominal (R$)", min_value=0.0, value=2500.0, step=100.0)
            tipo_especie = st.selectbox(
                "Especie do titulo",
                [
                    "DM DUPLICATA MERCANTIL",
                    "DMI DUPLICATA MERCANTIL INDICACAO",
                    "DS DUPLICATA DE SERVICO",
                    "NP NOTA PROMISSORIA",
                    "OUTROS",
                ],
            )
            referencia_pagador = st.text_input("CNPJ ou ID do pagador (opcional)", value="")
            referencia_beneficiario = st.text_input("CNPJ ou ID do beneficiario (opcional)", value="")
            uf_sacado = st.selectbox("UF do sacado (opcional)", [""] + form_options["uf_options"], index=0)
            uf_cedente = st.selectbox("UF do cedente (opcional)", [""] + form_options["uf_options"], index=0)
            cnae_principal_codigo_sacado = st.selectbox(
                "Setor/CNAE do sacado (opcional)",
                [""] + form_options["cnae_sacado_options"],
                index=0,
                format_func=format_cnae_option,
            )
            cnae_principal_codigo_cedente = st.selectbox(
                "Setor/CNAE do cedente (opcional)",
                [""] + form_options["cnae_cedente_options"],
                index=0,
                format_func=format_cnae_option,
            )

        with col2:
            media_atraso_dias_sacado = st.number_input(
                "Media de atraso do sacado (dias)",
                value=float(defaults["media_atraso_dias_sacado"]),
            )
            score_materialidade_v2_sacado = st.slider(
                "Score materialidade sacado",
                0.0,
                1000.0,
                float(defaults["score_materialidade_v2_sacado"]),
            )
            score_quantidade_v2_sacado = st.slider(
                "Score quantidade sacado",
                0.0,
                1000.0,
                float(defaults["score_quantidade_v2_sacado"]),
            )
            sacado_indice_liquidez_1m_sacado = st.slider(
                "Liquidez 1M do sacado",
                0.0,
                1.0,
                float(defaults["sacado_indice_liquidez_1m_sacado"]),
            )

        with col3:
            media_atraso_dias_cedente = st.number_input(
                "Media de atraso do cedente (dias)",
                value=float(defaults["media_atraso_dias_cedente"]),
            )
            score_materialidade_v2_cedente = st.slider(
                "Score materialidade cedente",
                0.0,
                1000.0,
                float(defaults["score_materialidade_v2_cedente"]),
            )
            score_quantidade_v2_cedente = st.slider(
                "Score quantidade cedente",
                0.0,
                1000.0,
                float(defaults["score_quantidade_v2_cedente"]),
            )
            cedente_indice_liquidez_1m_cedente = st.slider(
                "Liquidez 1M do cedente",
                0.0,
                1.0,
                float(defaults["cedente_indice_liquidez_1m_cedente"]),
            )
            indicador_liquidez_quantitativo_3m_cedente = st.slider(
                "Liquidez quantitativa 3M do cedente",
                0.0,
                1.0,
                float(defaults["indicador_liquidez_quantitativo_3m_cedente"]),
            )

        submitted = st.form_submit_button("Analisar titulo")

    if not submitted:
        return

    payload = {
        "vlr_nominal": vlr_nominal,
        "tipo_especie": tipo_especie,
        "referencia_pagador": referencia_pagador.strip() or None,
        "referencia_beneficiario": referencia_beneficiario.strip() or None,
        "media_atraso_dias_sacado": media_atraso_dias_sacado,
        "score_materialidade_v2_sacado": score_materialidade_v2_sacado,
        "score_quantidade_v2_sacado": score_quantidade_v2_sacado,
        "sacado_indice_liquidez_1m_sacado": sacado_indice_liquidez_1m_sacado,
        "media_atraso_dias_cedente": media_atraso_dias_cedente,
        "score_materialidade_v2_cedente": score_materialidade_v2_cedente,
        "score_quantidade_v2_cedente": score_quantidade_v2_cedente,
        "cedente_indice_liquidez_1m_cedente": cedente_indice_liquidez_1m_cedente,
        "indicador_liquidez_quantitativo_3m_cedente": indicador_liquidez_quantitativo_3m_cedente,
        "uf_sacado": uf_sacado or None,
        "uf_cedente": uf_cedente or None,
        "cnae_principal_codigo_sacado": cnae_principal_codigo_sacado.strip() or None,
        "cnae_principal_codigo_cedente": cnae_principal_codigo_cedente.strip() or None,
    }

    if payload["referencia_pagador"] or payload["referencia_beneficiario"]:
        st.caption("Se a referencia tiver 14 digitos, o sistema trata como CNPJ e tenta enriquecer com BrasilAPI; caso contrario, usa como ID interno.")

    if api_online:
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException:
            result = predict_from_payload(payload, bundle)
    else:
        result = predict_from_payload(payload, bundle)

    col1, col2, col3 = st.columns(3)
    col1.metric("Score PRISMA", f"{result['score_prisma']}/1000")
    col2.metric("Probabilidade de inadimplencia", f"{result['probabilidade_inadimplencia']:.2f}%")
    col3.metric("Decisao", result["status"])

    st.info(f"Faixa de risco: {result['faixa_risco']} | Acao: {result['acao_recomendada']}")
    render_rule_banner(result)
    st.markdown("**Principais fatores observados**")
    for reason in result["motivos_risco"]:
        st.write(f"- {reason}")

    with st.expander("Registro preparado para scoring"):
        st.dataframe(build_manual_input(payload, bundle)[FEATURE_COLUMNS], use_container_width=True)

def main() -> None:
    render_header()
    scored, bundle = load_default_portfolio()
    form_options = build_form_options(scored)
    tab1, tab2 = st.tabs(["Simulador de Risco", "Dashboard Gerencial"])
    with tab1:
        render_simulator(bundle, form_options)
    with tab2:
        render_dashboard_html(scored)


if __name__ == "__main__":
    main()
