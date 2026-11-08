"""Streamlit dashboard: Russian salaries, prices and the labour market, 2000-2023."""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    COLORS,
    SECTORS,
    SHORT_SECTORS,
    calculate_indicators,
    load_raw,
    period_stats,
    unemployment_correlation,
)

st.set_page_config(
    page_title="Зарплаты России · аналитика",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Учебный проект о зарплатах в России в 2000–2023 гг."},
)

@st.cache_data(show_spinner=False)
def prepare_data():
    return calculate_indicators(load_raw())

DATA = prepare_data()
YEAR_MIN, YEAR_MAX = 2000, 2023


def rubles(value):
    return f"{value:,.0f}".replace(",", " ") + " ₽"


def percent(value, signed=False):
    if not np.isfinite(value):
        return "—"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def base_chart(fig, height=410):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=22, r=22, t=28, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#CED6E8", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#162238", font_color="#F6FAFF"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#35445D", dtick=3, tickfont=dict(color="#9AAAC2"))
    fig.update_yaxes(gridcolor="#29344B", zerolinecolor="#38445B", tickfont=dict(color="#9AAAC2"))
    return fig


def boundary_line(fig, years):
    if years[0] < 2017 <= years[1]:
        fig.add_vline(x=2017, line_dash="dot", line_color="#71839A", opacity=0.9)
        fig.add_annotation(x=2017, y=1.0, yref="paper", text="ОКВЭД2", showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=5, font=dict(color="#9AAAC2", size=10))
    return fig


def reset_filters():
    st.session_state.industry_selector = list(SECTORS)
    st.session_state.period_selector = (YEAR_MIN, YEAR_MAX)
    st.session_state.price_mode = "В рублях 2023 года*"
    st.session_state.show_inflation = False


st.markdown("""
<style>
:root { --accent: #6FA8FF; }
[data-testid="stAppViewContainer"] { background: #0D1422; }
[data-testid="stSidebar"] { background: #121D2F; border-right: 1px solid #26354B; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1330px; }
.hero {
  padding: 2.0rem 2.2rem;
  border-radius: 20px;
  background: radial-gradient(circle at 95% 20%, #254676 0%, transparent 41%),
              linear-gradient(115deg, #14263F 0%, #182A4D 47%, #172035 100%);
  border: 1px solid #315078; margin-bottom: 1.1rem;
}
.hero .kicker { font-size: .77rem; letter-spacing: .14em; color: #8CC4FF; font-weight: 750; }
.hero h1 { color: #F5F9FF; font-size: clamp(1.8rem, 3.1vw, 2.9rem); margin: .48rem 0 .55rem 0; line-height: 1.16; font-weight: 790; }
.hero p { color: #CED9ED; max-width: 740px; line-height: 1.55; font-size: 1rem; margin: 0; }
.insight { margin: 1rem 0; padding: .82rem 1.1rem; border-left: 3px solid #74B3FF;
           border-radius: 4px 12px 12px 4px; background: #16253B; color: #D9E6F6; }
.muted-small { font-size: .83rem; color: #9DAEC4; }
[data-testid="stMetric"] { border: 1px solid #2B3D56; background: #142237;
                             padding: 17px 18px; border-radius: 14px; min-height: 124px; }
[data-testid="stMetricLabel"] { color: #AFC1DA; }
[data-testid="stMetricValue"] { color: #FAFCFF; font-weight: 730; font-size: 1.7rem; }
[data-testid="stTabs"] button[role="tab"] { font-weight: 650; }
[data-testid="stPlotlyChart"] { border: 1px solid #27354B; background: #101D2E;
                                  border-radius: 15px; padding: .7rem .3rem .2rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Настройки исследования")
    st.caption("Фильтры применяются ко всем графикам и таблицам.")
    st.button("↺ Сбросить фильтры", on_click=reset_filters, use_container_width=True)
    selected = st.multiselect(
        "Отрасли",
        options=list(SECTORS),
        default=list(SECTORS),
        format_func=lambda key: SECTORS[key],
        key="industry_selector",
    )
    if not selected:
        st.warning("Выберите хотя бы одну отрасль.")
        st.stop()
    year_range = st.slider(
        "Период, годы", min_value=YEAR_MIN, max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX), step=1, key="period_selector"
    )
    st.divider()
    price_mode = st.radio(
        "Вид реальной динамики",
        options=["В рублях 2023 года*", "Индекс (начало периода = 100)"],
        key="price_mode",
    )
    show_inflation = st.toggle("Показывать инфляцию на графике темпов", value=False, key="show_inflation")
    st.divider()
    st.caption("**Данные:** Росстат · CPI декабрь к декабрю · безработица")
    st.caption("**Диапазон:** 2000–2023 · 24 года")
    st.caption("*Реальная зарплата — приближённый пересчёт с декабрьским ИПЦ.")

start, end = year_range
filtered = DATA[DATA["year"].between(start, end) & DATA["sector_key"].isin(selected)].copy()
period_title = f"{start}—{end}"

a = """
<div class="hero">
  <div class="kicker">ИНТЕРАКТИВНОЕ ИССЛЕДОВАНИЕ · РОССИЯ · 2000–2023</div>
  <h1>Зарплаты в цифрах.<br/>Покупательная способность в контексте.</h1>
  <p>Как менялись доходы в разных отраслях, сколько роста «съедала» инфляция<br/>
     и есть ли связь с безработицей? Выберите период и исследуйте данные.</p>
</div>
"""
st.markdown(a, unsafe_allow_html=True)

st.caption(f"Сейчас: **{period_title}** · выбранных отраслей: **{len(selected)}**")

kpi_cols = st.columns(len(selected), gap="small")
for col, key in zip(kpi_cols, selected):
    stats = period_stats(DATA, key, start, end)
    with col:
        st.metric(
            label=SECTORS[key],
            value=rubles(stats["salary_end"]),
            delta=f"Реально за период: {percent(stats['real_growth_pct'], signed=True)}",
            delta_color="normal",
            help="Номинальная среднемесячная начисленная зарплата в конечном году периода. "
                 "Δ — изменение покупательной способности относительно начала периода.",
        )

st.markdown("<div class='muted-small'>Зарплаты — среднемесячные начисленные, до НДФЛ. Пересчёт на инфляцию приблизительный.</div>", unsafe_allow_html=True)
st.write("")

tab_nominal, tab_real, tab_growth, tab_market, tab_data = st.tabs([
    "💸 Номинальные зарплаты",
    "📈 Реальная динамика",
    "📊 Инфляция и рост",
    "🔎 Рынок труда",
    "🗂 Данные и методика",
])

with tab_nominal:
    st.subheader("Как менялись начисленные зарплаты")
    st.caption("Рубли соответствующего года, без корректировки на изменение уровня цен.")
    fig = go.Figure()
    for key in selected:
        sub = filtered[filtered["sector_key"] == key]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["salary_nominal"], mode="lines+markers",
            name=SHORT_SECTORS[key], line=dict(color=COLORS[key], width=3),
            marker=dict(size=6),
            customdata=sub["sector"],
            hovertemplate="<b>%{customdata}</b><br>%{x} год: %{y:,.0f} ₽<extra></extra>",
        ))
    fig = boundary_line(base_chart(fig), year_range)
    fig.update_yaxes(title="Рублей в месяц", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    if start < end:
        biggest = max(selected, key=lambda k: period_stats(DATA, k, start, end)["nominal_growth_pct"])
        s = period_stats(DATA, biggest, start, end)
        st.markdown(
            f"<div class='insight'>За {period_title} наиболее сильный <b>номинальный рост</b> "
            f"среди выбранных отраслей — <b>{SECTORS[biggest].lower()}</b>: "
            f"<b>{percent(s['nominal_growth_pct'], signed=True)}</b>. "
            "Но рост в рублях сам по себе не отражает рост покупательной способности.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Выбран один год. Для анализа роста расширьте период в боковом меню.")

with tab_real:
    st.subheader("Покупательная способность зарплат")
    if price_mode.startswith("В рублях"):
        st.caption("Условные рубли конца 2023 года*: номинальная зарплата с поправкой на накопленную инфляцию.")
    else:
        st.caption("Индекс реальной зарплаты: значение в первом выбранном году принято равным 100.")
    fig = go.Figure()
    for key in selected:
        sub = filtered[filtered["sector_key"] == key].sort_values("year")
        values = sub["salary_real_2023"]
        if price_mode.startswith("Индекс"):
            values = values / values.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=sub["year"], y=values, mode="lines+markers", name=SHORT_SECTORS[key],
            line=dict(color=COLORS[key], width=3), marker=dict(size=6),
            hovertemplate=("%{x} год: %{y:,.1f}<extra>" + SHORT_SECTORS[key] + "</extra>"),
        ))
    fig = boundary_line(base_chart(fig), year_range)
    fig.update_yaxes(title="Индекс (начало периода = 100)" if price_mode.startswith("Индекс") else "Условные ₽ 2023 года*")
    st.plotly_chart(fig, use_container_width=True)
    if start < end:
        best = max(selected, key=lambda k: period_stats(DATA, k, start, end)["real_factor"])
        s = period_stats(DATA, best, start, end)
        st.markdown(
            f"<div class='insight'>С учётом инфляции сильнее всего среди выбранных отраслей "
            f"выросла зарплата в сфере <b>{SECTORS[best].lower()}</b>: "
            f"в <b>{s['real_factor']:.2f} раза</b> за {period_title}.</div>",
            unsafe_allow_html=True,
        )
    st.info("* Это аналитическая оценка по годовому индексу цен декабрь/декабрь, а не официальный индекс реальной зарплаты. "
            "С 2017 года изменился отраслевой классификатор (ОКВЭД2).")

with tab_growth:
    st.subheader("Номинальный рост, инфляция и реальный результат")
    y = st.selectbox("Отрасль для сравнения", selected, format_func=lambda k: SECTORS[k], key="growth_sector")
    sub = filtered[filtered["sector_key"] == y].sort_values("year")
    # Keep comparisons within the displayed period; 2000 has no YoY baseline.
    sub = sub[sub["year"] >= max(2001, start)]
    if sub.empty:
        st.info("Для сравнения год-к-году выберите период, содержащий год после 2000.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sub["year"], y=sub["nominal_growth_pct"], name="Номинальный рост",
            marker_color="#59B9F6", opacity=0.9,
            hovertemplate="%{x}: %{y:+.1f}%<extra>Номинальный рост</extra>",
        ))
        fig.add_trace(go.Bar(
            x=sub["year"], y=sub["real_growth_pct"], name="Реальный рост*",
            marker_color="#A8A0F2", opacity=0.9,
            hovertemplate="%{x}: %{y:+.1f}%<extra>Реальный рост*</extra>",
        ))
        if show_inflation:
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["inflation_pct"], name="Инфляция",
                mode="lines+markers", line=dict(color="#F7B56C", width=3),
                hovertemplate="%{x}: %{y:.2f}%<extra>Инфляция</extra>",
            ))
        fig = boundary_line(base_chart(fig, 390), year_range)
        fig.update_layout(barmode="group", bargap=0.14)
        fig.update_yaxes(title="Изменение к предыдущему году, %", ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Карта реального роста по отраслям")
    heat = filtered[filtered["year"] >= max(2001, start)].pivot(
        index="sector_key", columns="year", values="real_growth_pct"
    ).reindex(selected)
    if not heat.empty and heat.shape[1] > 0:
        values = heat.to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(
            x=heat.columns.to_list(),
            y=[SHORT_SECTORS[k] for k in selected],
            z=values,
            zmin=-20, zmax=20, zmid=0,
            colorscale=[[0, "#C96168"], [0.5, "#233449"], [1, "#4DB1A8"]],
            colorbar=dict(title="%", thickness=12),
            hovertemplate="%{y}, %{x}: %{z:+.1f}%<extra></extra>",
        ))
        fig = base_chart(fig, 235)
        fig.update_xaxes(dtick=1, tickangle=-45)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Тёплые тона — сокращение покупательной способности, бирюзовые — рост. "
               "2017 год может быть искажён переходом на ОКВЭД2.")

with tab_market:
    st.subheader("Реальные зарплаты и безработица")
    st.caption("Дополнительный анализ: сопоставляем **изменение безработицы** (п.п.) "
               "с **годовым реальным ростом зарплат** (%), чтобы не сравнивать два тренда уровней.")
    sector_for_corr = st.selectbox(
        "Отрасль", selected, format_func=lambda k: SECTORS[k], key="corr_sector"
    )
    exclude_shocks = st.checkbox("Исключить 2009, 2020 и 2022 (проверка устойчивости)")
    points, r = unemployment_correlation(DATA, sector_for_corr, start, end, exclude_shocks)
    if len(points) < 3 or not np.isfinite(r):
        st.warning("Слишком мало наблюдений для корректного расчёта корреляции. Расширьте период.")
    else:
        metric_col, n_col = st.columns(2)
        metric_col.metric("Корреляция Пирсона, r", f"{r:+.2f}")
        n_col.metric("Число годовых наблюдений", str(len(points)))
        xs = points["unemployment_change_pp"].to_numpy(dtype=float)
        ys = points["real_growth_pct"].to_numpy(dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name="Годы",
            marker=dict(color=COLORS[sector_for_corr], size=11, line=dict(color="#13243A", width=1.3)),
            text=points["year"],
            hovertemplate="<b>%{text} год</b><br>Δ безработицы: %{x:+.1f} п.п."
                          "<br>Реальный рост: %{y:+.1f}%<extra></extra>",
        ))
        if len(set(np.round(xs, 10))) > 1:
            slope, intercept = np.polyfit(xs, ys, 1)
            line_x = np.linspace(float(xs.min()), float(xs.max()), 100)
            fig.add_trace(go.Scatter(
                x=line_x, y=slope * line_x + intercept, mode="lines",
                name="Линейный тренд", line=dict(color="#F7B56C", width=2, dash="dash"),
                hoverinfo="skip",
            ))
        fig = base_chart(fig, 380)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Изменение безработицы, п.п.", dtick=None)
        fig.update_yaxes(title="Реальный рост зарплаты, %*")
        st.plotly_chart(fig, use_container_width=True)
        orientation = "отрицательная" if r < 0 else "положительная"
        st.markdown(
            f"<div class='insight'>За выбранные годы наблюдается <b>{orientation} связь</b> "
            f"между изменением безработицы и реальной динамикой зарплат "
            f"(<b>r = {r:+.2f}</b>, n = {len(points)}). "
            "Корреляция не означает причинной зависимости.</div>",
            unsafe_allow_html=True,
        )
    st.caption("Сравнение 2016→2017 исключено из корреляции из-за смены ОКВЭД; "
               "годы 2009, 2020 и 2022 можно исключить отдельным переключателем.")

with tab_data:
    st.subheader("Данные и воспроизводимость")
    st.markdown("""
**Что измеряем.** Среднемесячную *номинальную начисленную* зарплату работников организаций
по трём видам экономической деятельности, инфляцию декабрь к декабрю и среднегодовую безработицу.

**Источники:**
- [Росстат — рынок труда и заработная плата](https://rosstat.gov.ru/labor_market_employment_salaries)
- [Инфляция в России — историческая таблица](https://xn----ctbjnaatncev9av3a8f8b.xn--p1ai/таблицы-инфляции)
- [Данные по безработице — справочная таблица](https://www.russiaeconomy.org/ru/tools/indicators/)
- [Зарплаты 2000–2016 (переложение Росстата)](https://github.com/Olga57/Start-DS/blob/main/salary_1.csv) и
  [2017–2023](https://github.com/Olga57/Start-DS/blob/main/salary_2.csv)

**Как считаем:**
- Номинальный темп: `(зарплата[t] / зарплата[t−1] − 1) × 100%`.
- Реальный темп*: `(зарплата[t] / зарплата[t−1] / (1 + инфляция[t]/100) − 1) × 100%`.
- В рублях 2023 года*: `зарплата[t] × индекс_цен[2023] / индекс_цен[t]`, при `индекс_цен[2000] = 1`.

**Ограничения:** переход на ОКВЭД2 в 2017 г.; годовая инфляция декабрь/декабрь
не является точным дефлятором среднегодовой зарплаты. Это учебная оценка реальной
динамики, а не официальный показатель Росстата. Корреляции не доказывают причинность.
""")
    export = filtered[[
        "year", "sector", "salary_nominal", "salary_real_2023", "nominal_growth_pct",
        "real_growth_pct", "inflation_pct", "unemployment_pct",
    ]].sort_values(["year", "sector"]).rename(columns={
        "year": "Год", "sector": "Отрасль", "salary_nominal": "Номинальная зарплата, ₽",
        "salary_real_2023": "Условно в рублях 2023 г., ₽*",
        "nominal_growth_pct": "Номинальный рост, %",
        "real_growth_pct": "Реальный рост, %*",
        "inflation_pct": "Инфляция, %", "unemployment_pct": "Безработица, %",
    })
    st.dataframe(export, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Скачать выбранные данные (CSV)",
        data=export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"zarplaty_{start}_{end}.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption("Приложение работает на локальном CSV, без API-ключей и внешних запросов.")

st.divider()
st.caption("Учебный проект · данные 2000–2023 · расчёты воспроизводимы · *приближённая поправка на инфляцию")
