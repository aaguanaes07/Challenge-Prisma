from __future__ import annotations

import base64
import json
from pathlib import Path
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


def build_dashboard_payload(scored: pd.DataFrame) -> dict:
    total_titulos = int(len(scored))
    volume_total = float(scored["vlr_nominal"].fillna(0).sum())
    aprovados = scored[scored["decisao_credito"] == "Aprovado"].copy()
    bloqueados = scored[scored["decisao_credito"] == "Bloquear"].copy()
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
    sector_colors = ["#FF3B30" if value >= 18 else "#00E096" for value in sector_values]

    alerts_rows = scored.sort_values(["probabilidade_inadimplencia", "vlr_nominal"], ascending=[False, False]).head(7)
    alerts = []
    for _, row in alerts_rows.iterrows():
        status = "red" if row["decisao_credito"] == "Bloquear" else ("yellow" if row["decisao_credito"] == "Analise Manual" else "green")
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
        </style>
        <div class="hero">
            <div class="hero-brand">
                __HEADER_VISUAL__
                <h1>PRISMA | Plataforma de Risco para FIDCs</h1>
            </div>
            <p>
                MVP funcional para a Sprint 3 com score calibrado, monitoramento da carteira,
                simulador manual e backend desacoplado via FastAPI.
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
        components.html(html_dashboard, height=1450, scrolling=True)
    except FileNotFoundError:
        st.error("Arquivo 'index.html' não encontrado.")


def render_simulator(bundle: dict) -> None:
    st.markdown("### Simulador de Risco Estrutural")
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
            cnpj_sacado = st.text_input("CNPJ do sacado (opcional)", value="")
            cnpj_cedente = st.text_input("CNPJ do cedente (opcional)", value="")
            uf_sacado = st.text_input("UF do sacado", value=str(defaults["uf_sacado"]))[:2].upper()
            uf_cedente = st.text_input("UF do cedente", value=str(defaults["uf_cedente"]))[:2].upper()

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
            share_vl_inad_pag_bol_6_a_15d_sacado = st.slider(
                "Share de atrasos curtos do sacado",
                0.0,
                1.0,
                float(defaults["share_vl_inad_pag_bol_6_a_15d_sacado"]),
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
        "cnpj_sacado": cnpj_sacado.strip() or None,
        "cnpj_cedente": cnpj_cedente.strip() or None,
        "media_atraso_dias_sacado": media_atraso_dias_sacado,
        "score_materialidade_v2_sacado": score_materialidade_v2_sacado,
        "score_quantidade_v2_sacado": score_quantidade_v2_sacado,
        "sacado_indice_liquidez_1m_sacado": sacado_indice_liquidez_1m_sacado,
        "share_vl_inad_pag_bol_6_a_15d_sacado": share_vl_inad_pag_bol_6_a_15d_sacado,
        "media_atraso_dias_cedente": media_atraso_dias_cedente,
        "score_materialidade_v2_cedente": score_materialidade_v2_cedente,
        "score_quantidade_v2_cedente": score_quantidade_v2_cedente,
        "cedente_indice_liquidez_1m_cedente": cedente_indice_liquidez_1m_cedente,
        "indicador_liquidez_quantitativo_3m_cedente": indicador_liquidez_quantitativo_3m_cedente,
        "uf_sacado": uf_sacado or defaults["uf_sacado"],
        "uf_cedente": uf_cedente or defaults["uf_cedente"],
    }

    if payload["cnpj_sacado"] or payload["cnpj_cedente"]:
        st.caption("CNPJs informados ativam enriquecimento externo: BrasilAPI para cadastro e bureau mock para credito.")

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
    st.markdown("**Principais fatores observados**")
    for reason in result["motivos_risco"]:
        st.write(f"- {reason}")

    with st.expander("Registro preparado para scoring"):
        st.dataframe(build_manual_input(payload, bundle)[FEATURE_COLUMNS], use_container_width=True)

def main() -> None:
    render_header()
    scored, bundle = load_default_portfolio()
    tab1, tab2 = st.tabs(["Simulador de Risco", "Dashboard Gerencial"])
    with tab1:
        render_simulator(bundle)
    with tab2:
        render_dashboard_html(scored)


if __name__ == "__main__":
    main()
