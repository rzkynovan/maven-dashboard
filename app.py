"""
Maven Fuzzy Factory — E-Commerce Analytics Dashboard
Dash app, siap deploy via Docker.
"""

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 0. LOAD & PREPROCESS DATA (sekali saat startup)
# ─────────────────────────────────────────────
DATA = "data/"

sessions   = pd.read_csv(DATA + "sessions_clean.csv",    parse_dates=["created_at"])
orders     = pd.read_csv(DATA + "orders_clean.csv",      parse_dates=["created_at"])
refunds    = pd.read_csv(DATA + "refunds_clean.csv",     parse_dates=["created_at"])
pageviews  = pd.read_csv(DATA + "pageviews_clean.csv",   parse_dates=["created_at"])
order_items= pd.read_csv(DATA + "order_items_clean.csv", parse_dates=["created_at"])
products   = pd.read_csv(DATA + "products_clean.csv",    parse_dates=["created_at"])

# Period bulanan
sessions["ym"] = sessions["created_at"].dt.to_period("M")
orders["ym"]   = orders["created_at"].dt.to_period("M")
refunds["ym"]  = refunds["created_at"].dt.to_period("M")

# Channel classification
def classify_channel(row):
    src  = str(row["utm_source"]).lower()
    camp = str(row["utm_campaign"]).lower()
    ref  = str(row["http_referer"]).lower()
    if src == "gsearch" and camp == "nonbrand":   return "Gsearch NonBrand"
    elif src == "gsearch" and camp == "brand":    return "Gsearch Brand"
    elif src == "bsearch" and camp == "nonbrand": return "Bsearch NonBrand"
    elif src == "bsearch" and camp == "brand":    return "Bsearch Brand"
    elif src == "socialbook":                     return "Social Media"
    elif src == "nan" and "gsearch" in ref:       return "Organic Google"
    elif src == "nan" and "bsearch" in ref:       return "Organic Bing"
    elif src == "nan" and ref == "nan":           return "Direct/Referral"
    else:                                         return "Other"

sessions["channel"] = sessions.apply(classify_channel, axis=1)

# ── Monthly trend ──────────────────────────────
monthly_sess = sessions.groupby("ym").size().rename("sessions")
monthly_ord  = orders.groupby("ym").size().rename("orders")
monthly      = pd.concat([monthly_sess, monthly_ord], axis=1).fillna(0).reset_index()
monthly["date"]      = monthly["ym"].dt.to_timestamp()
monthly["conv_rate"] = monthly["orders"] / monthly["sessions"] * 100
monthly["mom_sess"]  = monthly["sessions"].pct_change() * 100

# ── Revenue monthly ────────────────────────────
rev_m = orders.groupby("ym").agg(
    revenue  = ("price_usd", "sum"),
    cogs     = ("cogs_usd",  "sum"),
    n_orders = ("order_id",  "count")
).reset_index()
rev_m["date"]           = rev_m["ym"].dt.to_timestamp()
rev_m                   = rev_m.merge(monthly[["ym","sessions"]], on="ym", how="left")
rev_m["rev_per_order"]  = rev_m["revenue"] / rev_m["n_orders"]
rev_m["rev_per_session"]= rev_m["revenue"] / rev_m["sessions"]
rev_m["gross_margin"]   = rev_m["revenue"] - rev_m["cogs"]
rev_m["gm_pct"]         = rev_m["gross_margin"] / rev_m["revenue"] * 100

# ── Channel performance ────────────────────────
sess_ord = sessions.merge(
    orders[["website_session_id","order_id","price_usd","cogs_usd"]],
    on="website_session_id", how="left"
)
ch_perf = sess_ord.groupby("channel").agg(
    total_sessions = ("website_session_id", "count"),
    total_orders   = ("order_id", lambda x: x.notna().sum()),
    total_revenue  = ("price_usd", "sum")
).reset_index()
ch_perf["cvr_pct"]    = ch_perf["total_orders"] / ch_perf["total_sessions"] * 100
ch_perf["aov"]        = ch_perf["total_revenue"] / ch_perf["total_orders"].replace(0, np.nan)
ch_perf["rev_per_sess"]= ch_perf["total_revenue"] / ch_perf["total_sessions"]
ch_perf = ch_perf.sort_values("total_sessions", ascending=False)

# ── Refunds monthly ────────────────────────────
ref_m = refunds.groupby("ym").agg(
    refund_count  = ("order_item_refund_id", "count"),
    refund_amount = ("refund_amount_usd", "sum")
).reset_index()
ref_m["date"] = ref_m["ym"].dt.to_timestamp()

# ── Device & top pages ─────────────────────────
device_dist = sessions["device_type"].value_counts().reset_index()
device_dist.columns = ["device_type", "count"]
top_pages = pageviews["pageview_url"].value_counts().head(10).reset_index()
top_pages.columns = ["page", "views"]

# ── KPI summary ────────────────────────────────
total_sessions_all = len(sessions)
total_orders_all   = len(orders)
total_revenue_all  = orders["price_usd"].sum()
total_refunds_all  = refunds["refund_amount_usd"].sum()
net_revenue_all    = total_revenue_all - total_refunds_all
overall_cvr        = total_orders_all / total_sessions_all * 100
avg_rpo            = total_revenue_all / total_orders_all
avg_rps            = total_revenue_all / total_sessions_all

# ── Date range untuk filter ────────────────────
date_min = monthly["date"].min()
date_max = monthly["date"].max()
months_all = monthly["date"].dt.to_period("M").astype(str).tolist()

# ─────────────────────────────────────────────
# WARNA
# ─────────────────────────────────────────────
CHANNEL_COLORS = {
    "Gsearch NonBrand": "#4C72B0", "Direct/Referral": "#55A868",
    "Gsearch Brand":    "#8172B2", "Bsearch NonBrand":"#DD8452",
    "Organic Google":   "#C44E52", "Bsearch Brand":   "#56d364",
    "Social Media":     "#e3b341", "Organic Bing":    "#79c0ff",
    "Other":            "#8b949e",
}
BG    = "#0f172a"
CARD  = "#1e293b"
TEXT  = "#e2e8f0"
DIM   = "#94a3b8"
BLUE  = "#3b82f6"
GREEN = "#22c55e"
RED   = "#ef4444"
AMBER = "#f59e0b"
PURPLE= "#a855f7"

def kpi_card(label, value, color=BLUE):
    return html.Div([
        html.Div(label, style={"color": DIM, "fontSize": "11px", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        html.Div(value, style={"color": color, "fontSize": "18px", "fontWeight": "700", "whiteSpace": "nowrap"}),
    ], style={
        "backgroundColor": CARD,
        "borderRadius": "8px",
        "padding": "10px 14px",
        "border": f"1px solid #1e293b",
        "borderTop": f"2px solid {color}",
        "minWidth": "0",
    })

# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
GAP  = "8px"
GCFG = {"displayModeBar": False}

def gchart(chart_id, height="300px"):
    return dcc.Graph(id=chart_id, config=GCFG,
                     style={"height": height, "width": "100%"})

def panel(title, chart_id, height="300px"):
    return html.Div([
        html.Div(title, style={
            "color": TEXT, "fontWeight": "600", "fontSize": "13px",
            "padding": "12px 14px 0 14px",
        }),
        gchart(chart_id, height),
    ], style={
        "backgroundColor": CARD,
        "borderRadius": "10px",
        "border": "1px solid #1e293b",
        "overflow": "hidden",
        "display": "flex", "flexDirection": "column",
    })

app = Dash(
    __name__,
    title="Maven Fuzzy Factory Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# inject CSS global: hilangkan margin body, scrollbar gelap
app.index_string = """<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{background:#0f172a;overflow-x:hidden;scrollbar-width:thin;scrollbar-color:#334155 #0f172a}
body::-webkit-scrollbar{width:6px}
body::-webkit-scrollbar-track{background:#0f172a}
body::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
.rc-slider-track{background:#3b82f6!important}
.rc-slider-handle{border-color:#3b82f6!important;background:#3b82f6!important}
</style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

app.layout = html.Div(style={
    "backgroundColor": BG,
    "fontFamily": "'Inter','Segoe UI',sans-serif",
    "color": TEXT,
    "minHeight": "100vh",
    "display": "flex", "flexDirection": "column",
}, children=[

    # ── HEADER ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("Maven Fuzzy Factory", style={
                "fontSize": "18px", "fontWeight": "800", "color": TEXT,
            }),
            html.Span(" · E-Commerce Analytics", style={
                "fontSize": "12px", "color": DIM, "marginLeft": "8px",
            }),
        ]),
        html.Div([
            html.Span("Rentang Bulan:", style={"color": DIM, "fontSize": "11px", "marginRight": "12px", "whiteSpace": "nowrap"}),
            dcc.RangeSlider(
                id="date-slider",
                min=0, max=len(months_all) - 1,
                value=[0, len(months_all) - 1],
                marks={i: {"label": months_all[i], "style": {"color": DIM, "fontSize": "9px"}}
                       for i in range(0, len(months_all), 6)},
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
            ),
        ], style={"flex": "1", "display": "flex", "alignItems": "center",
                  "marginLeft": "32px", "minWidth": "0"}),
    ], style={
        "display": "flex", "alignItems": "center",
        "backgroundColor": CARD,
        "padding": "10px 16px",
        "borderBottom": "1px solid #1e293b",
        "position": "sticky", "top": "0", "zIndex": "100",
        "gap": "8px",
    }),

    # ── KPI ROW ─────────────────────────────────────────────────────────────
    html.Div(id="kpi-cards", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(7, 1fr)",
        "gap": GAP,
        "padding": f"{GAP} {GAP} 0 {GAP}",
    }),

    # ── GRID UTAMA ──────────────────────────────────────────────────────────
    html.Div([

        # Baris 1 — 2:1
        html.Div([
            panel("① Tren Sessions & Orders",   "chart-trend",  "290px"),
            panel("② Conversion Rate Trend",    "chart-cvr",    "290px"),
        ], style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": GAP}),

        # Baris 2 — 1:1:1
        html.Div([
            panel("③ Sessions per Channel",    "chart-ch-sessions", "300px"),
            panel("③ CVR per Channel",         "chart-ch-cvr",      "300px"),
            panel("③ Revenue Share (Donut)",   "chart-ch-donut",    "300px"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": GAP}),

        # Baris 3 — 2:1
        html.Div([
            panel("④ Revenue per Order & Session", "chart-revenue", "300px"),
            panel("⑤ COGS vs Gross Margin",        "chart-margin",  "300px"),
        ], style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": GAP}),

        # Baris 4 — full width
        panel("⑥ Monthly Refund Count & Amount", "chart-refund", "260px"),

        # Baris 5 — 1:2
        html.Div([
            panel("⑦ Sessions by Device Type",        "chart-device", "280px"),
            panel("⑦ Top 10 Halaman (Pageviews)",     "chart-pages",  "280px"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": GAP}),

    ], style={
        "display": "grid",
        "gridTemplateColumns": "1fr",
        "gap": GAP,
        "padding": GAP,
        "flex": "1",
    }),
])

# ─────────────────────────────────────────────
# HELPER: layout plotly gelap
# ─────────────────────────────────────────────
PLOT_BG = "#162032"   # sedikit lebih terang dari CARD agar kontras

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, size=11),
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=DIM),
        ),
        xaxis=dict(
            gridcolor="#1e3a5f", linecolor="#334155",
            tickfont=dict(color=DIM),
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            gridcolor="#1e3a5f", linecolor="#334155",
            tickfont=dict(color=DIM),
            zerolinecolor="#334155",
        ),
    )
    base.update(kwargs)
    return base   # kembalikan dict, bukan go.Layout

# ─────────────────────────────────────────────
# HELPER: filter data berdasarkan slider
# ─────────────────────────────────────────────
def filter_monthly(slider_val):
    lo, hi = slider_val
    return monthly.iloc[lo: hi + 1]

def filter_rev(slider_val):
    lo, hi = slider_val
    # filter rev_m berdasarkan bulan yang sama
    m_filtered = filter_monthly(slider_val)
    return rev_m[rev_m["ym"].isin(m_filtered["ym"])]

def filter_ref(slider_val):
    lo, hi = slider_val
    m_filtered = filter_monthly(slider_val)
    return ref_m[ref_m["ym"].isin(m_filtered["ym"])]

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

@app.callback(
    Output("kpi-cards", "children"),
    Input("date-slider", "value"),
)
def update_kpi(slider_val):
    m  = filter_monthly(slider_val)
    rv = filter_rev(slider_val)
    rf = filter_ref(slider_val)

    sess  = m["sessions"].sum()
    ord_  = m["orders"].sum()
    rev   = rv["revenue"].sum()
    ref_  = rf["refund_amount"].sum()
    cvr   = ord_ / sess * 100 if sess > 0 else 0
    rpo   = rev / ord_ if ord_ > 0 else 0
    rps   = rev / sess if sess > 0 else 0
    net   = rev - ref_

    return [
        kpi_card("Total Sessions",      f"{int(sess):,}",   BLUE),
        kpi_card("Total Orders",        f"{int(ord_):,}",   GREEN),
        kpi_card("Total Revenue",       f"${rev:,.0f}",     AMBER),
        kpi_card("Net Revenue",         f"${net:,.0f}",     PURPLE),
        kpi_card("Conversion Rate",     f"{cvr:.2f}%",      "#06b6d4"),
        kpi_card("Avg Rev / Order",     f"${rpo:.2f}",      "#f97316"),
        kpi_card("Avg Rev / Session",   f"${rps:.4f}",      "#ec4899"),
    ]


@app.callback(
    Output("chart-trend", "figure"),
    Input("date-slider", "value"),
)
def update_trend(slider_val):
    m = filter_monthly(slider_val).copy()
    m["date"] = pd.to_datetime(m["ym"].astype(str))
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=m["date"], y=m["sessions"],
        name="Sessions", marker_color=BLUE, opacity=0.6,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=m["date"], y=m["orders"],
        name="Orders", line=dict(color=GREEN, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=True)

    layout = dark_layout(legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)))
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Sessions", secondary_y=False,
                     tickfont=dict(color=BLUE), gridcolor="#1e3a5f", linecolor="#334155")
    fig.update_yaxes(title_text="Orders",   secondary_y=True,
                     tickfont=dict(color=GREEN), gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#1e3a5f", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-cvr", "figure"),
    Input("date-slider", "value"),
)
def update_cvr(slider_val):
    m   = filter_monthly(slider_val).copy()
    m["date"] = pd.to_datetime(m["ym"].astype(str))
    avg = m["conv_rate"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["date"], y=m["conv_rate"],
        fill="tozeroy", fillcolor="rgba(168,85,247,0.25)",
        line=dict(color=PURPLE, width=2.5),
        mode="lines+markers", marker=dict(size=4),
        name="CVR (%)",
    ))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1,
                  yref="y", y0=avg, y1=avg,
                  line=dict(color=DIM, dash="dash", width=1.5))
    fig.add_annotation(xref="paper", x=1.0, yref="y", y=avg,
                       text=f"Avg {avg:.2f}%", showarrow=False,
                       font=dict(color=DIM, size=10), xanchor="right")
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(**dark_layout())
    return fig


@app.callback(
    Output("chart-ch-sessions", "figure"),
    Input("date-slider", "value"),
)
def update_ch_sessions(_):
    df = ch_perf.sort_values("total_sessions")
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["channel"], x=df["total_sessions"],
        orientation="h", marker_color=colors, opacity=0.85,
        text=[f"{int(v):,}" for v in df["total_sessions"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_layout(**dark_layout(margin=dict(l=130, r=20, t=10, b=40)))
    fig.update_xaxes(tickformat=",")
    return fig


@app.callback(
    Output("chart-ch-cvr", "figure"),
    Input("date-slider", "value"),
)
def update_ch_cvr(_):
    df = ch_perf.sort_values("cvr_pct")
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["channel"], x=df["cvr_pct"],
        orientation="h", marker_color=colors, opacity=0.85,
        text=[f"{v:.2f}%" for v in df["cvr_pct"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_layout(**dark_layout(margin=dict(l=130, r=20, t=10, b=40)))
    fig.update_xaxes(ticksuffix="%")
    return fig


@app.callback(
    Output("chart-ch-donut", "figure"),
    Input("date-slider", "value"),
)
def update_ch_donut(_):
    df = ch_perf.sort_values("total_revenue", ascending=False).head(6)
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=df["channel"], values=df["total_revenue"],
        hole=0.55, marker_colors=colors,
        textinfo="percent", textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**dark_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color=DIM, size=10), orientation="v", bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


@app.callback(
    Output("chart-revenue", "figure"),
    Input("date-slider", "value"),
)
def update_revenue(slider_val):
    rv = filter_rev(slider_val).copy()
    rv["date"] = pd.to_datetime(rv["ym"].astype(str))
    rpo_avg = rv["rev_per_order"].mean()
    rps_avg = rv["rev_per_session"].mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=rv["date"], y=rv["rev_per_order"],
        name="Rev / Order ($)",
        line=dict(color=AMBER, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=False)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1,
                  yref="y", y0=rpo_avg, y1=rpo_avg,
                  line=dict(color="rgba(245,158,11,0.5)", dash="dash", width=1.5))
    fig.add_annotation(xref="paper", x=1.0, yref="y", y=rpo_avg,
                       text=f"Avg ${rpo_avg:.2f}", showarrow=False,
                       font=dict(color=DIM, size=10), xanchor="right")

    fig.add_trace(go.Scatter(
        x=rv["date"], y=rv["rev_per_session"],
        name="Rev / Session ($)",
        line=dict(color=GREEN, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=True)
    # add_hline tidak support secondary_y → pakai add_shape dengan yref="y2"
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1,
        yref="y2", y0=rps_avg, y1=rps_avg,
        line=dict(color="rgba(34,197,94,0.5)", dash="dot", width=1.5),
    )
    fig.add_annotation(
        xref="paper", x=1.0, yref="y2", y=rps_avg,
        text=f"Avg ${rps_avg:.4f}", showarrow=False,
        font=dict(color=DIM, size=10), xanchor="right",
    )

    layout = dark_layout(legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)))
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Rev/Order ($)",   secondary_y=False,
                     tickprefix="$", tickfont=dict(color=AMBER),
                     gridcolor="#1e3a5f", linecolor="#334155")
    fig.update_yaxes(title_text="Rev/Session ($)", secondary_y=True,
                     tickprefix="$", tickfont=dict(color=GREEN),
                     gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#1e3a5f", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-margin", "figure"),
    Input("date-slider", "value"),
)
def update_margin(slider_val):
    rv = filter_rev(slider_val).copy()
    rv["date"] = pd.to_datetime(rv["ym"].astype(str))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rv["date"], y=rv["cogs"],
        name="COGS", marker_color=RED, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=rv["date"], y=rv["gross_margin"],
        name="Gross Margin", marker_color=GREEN, opacity=0.8,
    ))
    fig.update_layout(**dark_layout(
        barmode="stack",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    ))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


@app.callback(
    Output("chart-refund", "figure"),
    Input("date-slider", "value"),
)
def update_refund(slider_val):
    rf = filter_ref(slider_val).copy()
    rf["date"] = pd.to_datetime(rf["ym"].astype(str))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=rf["date"], y=rf["refund_count"],
        name="Refund Count", marker_color=RED, opacity=0.75,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=rf["date"], y=rf["refund_amount"],
        name="Refund Amount ($)",
        line=dict(color=AMBER, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=True)
    layout = dark_layout(legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)))
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Refund Count",    secondary_y=False,
                     tickfont=dict(color=RED),
                     gridcolor="#1e3a5f", linecolor="#334155")
    fig.update_yaxes(title_text="Refund Amount ($)", secondary_y=True,
                     tickprefix="$", tickfont=dict(color=AMBER),
                     gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#1e3a5f", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-device", "figure"),
    Input("date-slider", "value"),
)
def update_device(_):
    colors = [BLUE, AMBER, GREEN, PURPLE][:len(device_dist)]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=device_dist["device_type"],
        values=device_dist["count"],
        hole=0.45,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Sessions: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**dark_layout(margin=dict(l=10, r=10, t=10, b=10)))
    return fig


@app.callback(
    Output("chart-pages", "figure"),
    Input("date-slider", "value"),
)
def update_pages(_):
    df = top_pages.sort_values("views")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["page"], x=df["views"],
        orientation="h",
        marker_color=BLUE, opacity=0.85,
        text=[f"{int(v):,}" for v in df["views"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_layout(**dark_layout(margin=dict(l=160, r=20, t=10, b=40)))
    fig.update_xaxes(tickformat=",")
    return fig


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
