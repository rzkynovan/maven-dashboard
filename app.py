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

def card(children, style=None):
    base = {
        "backgroundColor": CARD,
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "16px",
        "border": f"1px solid #334155",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi_card(label, value, color=BLUE):
    return html.Div([
        html.P(label, style={"color": DIM, "fontSize": "13px", "margin": "0 0 4px 0"}),
        html.H3(value, style={"color": color, "margin": "0", "fontSize": "22px", "fontWeight": "700"}),
    ], style={
        "backgroundColor": CARD,
        "borderRadius": "10px",
        "padding": "16px 20px",
        "border": f"1px solid {color}33",
        "flex": "1",
        "minWidth": "150px",
    })

# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
app = Dash(
    __name__,
    title="Maven Fuzzy Factory Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server untuk Gunicorn

app.layout = html.Div(style={
    "backgroundColor": BG,
    "minHeight": "100vh",
    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
    "color": TEXT,
    "padding": "0",
}, children=[

    # ── HEADER ──────────────────────────────────
    html.Div([
        html.Div([
            html.H1("Maven Fuzzy Factory", style={
                "margin": "0", "fontSize": "24px", "fontWeight": "800", "color": TEXT
            }),
            html.P("E-Commerce Analytics Dashboard", style={
                "margin": "2px 0 0 0", "color": DIM, "fontSize": "13px"
            }),
        ]),
        html.Div([
            html.Label("Filter Rentang Bulan:", style={"color": DIM, "fontSize": "12px", "marginBottom": "4px"}),
            dcc.RangeSlider(
                id="date-slider",
                min=0, max=len(months_all) - 1,
                value=[0, len(months_all) - 1],
                marks={
                    i: {"label": months_all[i], "style": {"color": DIM, "fontSize": "10px"}}
                    for i in range(0, len(months_all), 6)
                },
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
            ),
        ], style={"flex": "1", "marginLeft": "40px", "maxWidth": "700px"}),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "backgroundColor": CARD,
        "padding": "20px 32px",
        "borderBottom": "1px solid #334155",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # ── MAIN CONTENT ────────────────────────────
    html.Div(style={"padding": "24px 32px"}, children=[

        # ── KPI CARDS ───────────────────────────
        html.Div(id="kpi-cards", style={
            "display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"
        }),

        # ── BARIS 1: Trend & CVR ────────────────
        html.Div([
            html.Div([
                card([
                    html.H4("① Tren Sessions & Orders", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-trend", config={"displayModeBar": False}, style={"height": "300px"}),
                ])
            ], style={"flex": "2", "minWidth": "400px"}),
            html.Div([
                card([
                    html.H4("② Conversion Rate Trend", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-cvr", config={"displayModeBar": False}, style={"height": "300px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── BARIS 2: Channel ─────────────────────
        html.Div([
            html.Div([
                card([
                    html.H4("③ Sessions per Channel", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-ch-sessions", config={"displayModeBar": False}, style={"height": "300px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
            html.Div([
                card([
                    html.H4("③ CVR per Channel", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-ch-cvr", config={"displayModeBar": False}, style={"height": "300px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
            html.Div([
                card([
                    html.H4("③ Revenue Share (Donut)", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-ch-donut", config={"displayModeBar": False}, style={"height": "300px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── BARIS 3: Revenue ─────────────────────
        html.Div([
            html.Div([
                card([
                    html.H4("④ Revenue per Order & Session", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-revenue", config={"displayModeBar": False}, style={"height": "320px"}),
                ])
            ], style={"flex": "2", "minWidth": "400px"}),
            html.Div([
                card([
                    html.H4("⑤ COGS vs Gross Margin", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-margin", config={"displayModeBar": False}, style={"height": "320px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── BARIS 4: Refund ──────────────────────
        card([
            html.H4("⑥ Monthly Refund Count & Amount", style={"color": TEXT, "margin": "0 0 12px 0"}),
            dcc.Graph(id="chart-refund", config={"displayModeBar": False}, style={"height": "280px"}),
        ]),

        # ── BARIS 5: Device & Pages ──────────────
        html.Div([
            html.Div([
                card([
                    html.H4("⑦ Sessions by Device Type", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-device", config={"displayModeBar": False}, style={"height": "280px"}),
                ])
            ], style={"flex": "1", "minWidth": "280px"}),
            html.Div([
                card([
                    html.H4("⑦ Top 10 Halaman (Pageviews)", style={"color": TEXT, "margin": "0 0 12px 0"}),
                    dcc.Graph(id="chart-pages", config={"displayModeBar": False}, style={"height": "280px"}),
                ])
            ], style={"flex": "2", "minWidth": "400px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    ]),

    # ── FOOTER ──────────────────────────────────
    html.Div("Maven Fuzzy Factory Analytics • Built with Dash", style={
        "textAlign": "center", "color": DIM, "fontSize": "12px",
        "padding": "20px", "borderTop": "1px solid #334155",
    }),
])

# ─────────────────────────────────────────────
# HELPER: layout plotly gelap
# ─────────────────────────────────────────────
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=11),
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=DIM),
        ),
        xaxis=dict(
            gridcolor="#334155", linecolor="#334155",
            tickfont=dict(color=DIM),
        ),
        yaxis=dict(
            gridcolor="#334155", linecolor="#334155",
            tickfont=dict(color=DIM),
        ),
    )
    base.update(kwargs)
    return go.Layout(**base)

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
    m = filter_monthly(slider_val)
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

    fig.update_layout(dark_layout(
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    ))
    fig.update_yaxes(title_text="Sessions", secondary_y=False,
                     tickfont=dict(color=BLUE), gridcolor="#334155", linecolor="#334155")
    fig.update_yaxes(title_text="Orders",   secondary_y=True,
                     tickfont=dict(color=GREEN), gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#334155", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-cvr", "figure"),
    Input("date-slider", "value"),
)
def update_cvr(slider_val):
    m   = filter_monthly(slider_val)
    avg = m["conv_rate"].mean()
    fig = go.Figure(layout=dark_layout())
    fig.add_trace(go.Scatter(
        x=m["date"], y=m["conv_rate"],
        fill="tozeroy", fillcolor=f"{PURPLE}30",
        line=dict(color=PURPLE, width=2.5),
        mode="lines+markers", marker=dict(size=4),
        name="CVR (%)",
    ))
    fig.add_hline(y=avg, line_dash="dash", line_color=DIM,
                  annotation_text=f"Avg {avg:.2f}%",
                  annotation_font_color=DIM)
    fig.update_yaxes(ticksuffix="%")
    return fig


@app.callback(
    Output("chart-ch-sessions", "figure"),
    Input("date-slider", "value"),
)
def update_ch_sessions(_):
    df = ch_perf.sort_values("total_sessions")
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure(layout=dark_layout(margin=dict(l=130, r=20, t=10, b=40)))
    fig.add_trace(go.Bar(
        y=df["channel"], x=df["total_sessions"],
        orientation="h", marker_color=colors, opacity=0.85,
        text=[f"{int(v):,}" for v in df["total_sessions"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_xaxes(tickformat=",")
    return fig


@app.callback(
    Output("chart-ch-cvr", "figure"),
    Input("date-slider", "value"),
)
def update_ch_cvr(_):
    df = ch_perf.sort_values("cvr_pct")
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure(layout=dark_layout(margin=dict(l=130, r=20, t=10, b=40)))
    fig.add_trace(go.Bar(
        y=df["channel"], x=df["cvr_pct"],
        orientation="h", marker_color=colors, opacity=0.85,
        text=[f"{v:.2f}%" for v in df["cvr_pct"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_xaxes(ticksuffix="%")
    return fig


@app.callback(
    Output("chart-ch-donut", "figure"),
    Input("date-slider", "value"),
)
def update_ch_donut(_):
    df = ch_perf.sort_values("total_revenue", ascending=False).head(6)
    colors = [CHANNEL_COLORS.get(c, "#8b949e") for c in df["channel"]]
    fig = go.Figure(layout=dark_layout(margin=dict(l=10, r=10, t=10, b=10)))
    fig.add_trace(go.Pie(
        labels=df["channel"], values=df["total_revenue"],
        hole=0.55, marker_colors=colors,
        textinfo="percent", textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        legend=dict(font=dict(color=DIM, size=10), orientation="v",
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


@app.callback(
    Output("chart-revenue", "figure"),
    Input("date-slider", "value"),
)
def update_revenue(slider_val):
    rv  = filter_rev(slider_val)
    rpo_avg = rv["rev_per_order"].mean()
    rps_avg = rv["rev_per_session"].mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=rv["date"], y=rv["rev_per_order"],
        name="Rev / Order ($)",
        line=dict(color=AMBER, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=False)
    fig.add_hline(y=rpo_avg, line_dash="dash", line_color=f"{AMBER}80",
                  annotation_text=f"Avg ${rpo_avg:.2f}",
                  annotation_font_color=DIM)

    fig.add_trace(go.Scatter(
        x=rv["date"], y=rv["rev_per_session"],
        name="Rev / Session ($)",
        line=dict(color=GREEN, width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ), secondary_y=True)
    fig.add_hline(y=rps_avg, line_dash="dot", line_color=f"{GREEN}80",
                  annotation_text=f"Avg ${rps_avg:.4f}",
                  annotation_font_color=DIM, secondary_y=True)

    fig.update_layout(dark_layout(
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    ))
    fig.update_yaxes(title_text="Rev/Order ($)",   secondary_y=False,
                     tickprefix="$", tickfont=dict(color=AMBER),
                     gridcolor="#334155", linecolor="#334155")
    fig.update_yaxes(title_text="Rev/Session ($)", secondary_y=True,
                     tickprefix="$", tickfont=dict(color=GREEN),
                     gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#334155", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-margin", "figure"),
    Input("date-slider", "value"),
)
def update_margin(slider_val):
    rv = filter_rev(slider_val)
    fig = go.Figure(layout=dark_layout())
    fig.add_trace(go.Bar(
        x=rv["date"], y=rv["cogs"],
        name="COGS", marker_color=RED, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=rv["date"], y=rv["gross_margin"],
        name="Gross Margin", marker_color=GREEN, opacity=0.8,
    ))
    fig.update_layout(
        barmode="stack",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


@app.callback(
    Output("chart-refund", "figure"),
    Input("date-slider", "value"),
)
def update_refund(slider_val):
    rf = filter_ref(slider_val)
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
    fig.update_layout(dark_layout(
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    ))
    fig.update_yaxes(title_text="Refund Count",    secondary_y=False,
                     tickfont=dict(color=RED),
                     gridcolor="#334155", linecolor="#334155")
    fig.update_yaxes(title_text="Refund Amount ($)", secondary_y=True,
                     tickprefix="$", tickfont=dict(color=AMBER),
                     gridcolor="rgba(0,0,0,0)", linecolor="#334155")
    fig.update_xaxes(gridcolor="#334155", linecolor="#334155", tickfont=dict(color=DIM))
    return fig


@app.callback(
    Output("chart-device", "figure"),
    Input("date-slider", "value"),
)
def update_device(_):
    colors = [BLUE, AMBER, GREEN, PURPLE][:len(device_dist)]
    fig = go.Figure(layout=dark_layout(margin=dict(l=10, r=10, t=10, b=10)))
    fig.add_trace(go.Pie(
        labels=device_dist["device_type"],
        values=device_dist["count"],
        hole=0.45,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Sessions: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    return fig


@app.callback(
    Output("chart-pages", "figure"),
    Input("date-slider", "value"),
)
def update_pages(_):
    df = top_pages.sort_values("views")
    fig = go.Figure(layout=dark_layout(margin=dict(l=160, r=20, t=10, b=40)))
    fig.add_trace(go.Bar(
        y=df["page"], x=df["views"],
        orientation="h",
        marker_color=BLUE, opacity=0.85,
        text=[f"{int(v):,}" for v in df["views"]],
        textposition="outside", textfont=dict(color=DIM, size=10),
    ))
    fig.update_xaxes(tickformat=",")
    return fig


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
