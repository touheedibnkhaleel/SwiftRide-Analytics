import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwiftRide Analytics",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, [data-testid="stMetricLabel"] {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid rgba(255,200,50,0.15);
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.02em;
    padding: 6px 0;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #252540 100%);
    border: 1px solid rgba(255,200,50,0.2);
    border-radius: 14px;
    padding: 20px 24px 16px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(255,200,50,0.12);
}
.kpi-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #ffcc2f;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #f0f0ff;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #8888aa;
    margin-top: 6px;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #ffcc2f;
    text-transform: uppercase;
    border-left: 3px solid #ffcc2f;
    padding-left: 10px;
    margin: 24px 0 16px;
}

/* Sidebar footer */
.sidebar-footer {
    position: fixed;
    bottom: 0;
    width: 260px;
    padding: 14px 20px;
    background: rgba(10,10,20,0.95);
    border-top: 1px solid rgba(255,200,50,0.15);
    font-size: 0.73rem;
    color: #6666aa !important;
    line-height: 1.7;
}
.sidebar-footer span {
    color: #ffcc2f !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

/* Page title */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0f0ff;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #8888aa;
    margin-bottom: 28px;
}

/* Plotly chart container */
.chart-wrap {
    background: #13131f;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="#13131f",
    plot_bgcolor="#13131f",
    font_color="#c8c8e0",
    font_family="DM Sans",
    title_font_family="Syne",
    title_font_color="#f0f0ff",
    colorway=["#ffcc2f", "#ff6b6b", "#4ecdc4", "#a78bfa", "#fb923c",
              "#34d399", "#60a5fa", "#f472b6"],
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
)

def apply_theme(fig, title=None, height=380):
    fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"], family=CHART_THEME["font_family"]),
        title_font=dict(family=CHART_THEME["title_font_family"],
                        color=CHART_THEME["title_font_color"], size=15),
        height=height,
        margin=dict(l=40, r=20, t=52, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#c8c8e0"),
    )
    if title:
        fig.update_layout(title_text=title)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.1)",
    )
    return fig

VEHICLE_COLORS = {
    "Bike":     "#ffcc2f",
    "Rickshaw": "#ff6b6b",
    "Car":      "#4ecdc4",
    "SUV":      "#a78bfa",
}

# ─────────────────────────────────────────────────────────────────
# DATABASE HELPER
# ─────────────────────────────────────────────────────────────────
DB_PATH = "swiftride.db"

@st.cache_data(ttl=60)
def query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 24px;'>
        <div style='font-family:Syne; font-size:1.6rem; font-weight:800;
                    color:#ffcc2f; letter-spacing:-0.02em;'>🚗 SwiftRide</div>
        <div style='font-size:0.72rem; color:#6666aa; letter-spacing:0.1em;
                    text-transform:uppercase; margin-top:4px;'>Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Executive Overview",
         "🗺️ Trip Analytics",
         "🏆 Driver Performance",
         "🤖 ML Fare Predictor"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class='sidebar-footer'>
        <span>SwiftRide Analytics v1.0</span><br>
        Powered by Claude + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "📊 Executive Overview":

    st.markdown("<div class='page-title'>Executive Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>High-level performance summary across all cities and vehicle types</div>",
                unsafe_allow_html=True)

    # ── Section A: KPI Cards ──────────────────────────────────────
    with st.spinner("Loading KPIs..."):
        kpi_df = query("""
            SELECT
                SUM(CASE WHEN status='completed' THEN fare_pkr ELSE 0 END)  AS total_revenue,
                COUNT(*)                                                      AS total_trips,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)          AS completed_trips,
                AVG(CASE WHEN status='completed' THEN fare_pkr END)           AS avg_fare
            FROM trips
        """)

    row = kpi_df.iloc[0]
    total_rev     = row["total_revenue"]
    total_trips   = int(row["total_trips"])
    completed     = int(row["completed_trips"])
    completion_rt = completed / total_trips * 100
    avg_fare      = row["avg_fare"]

    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, label, value, sub=""):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "Total Revenue",    f"PKR {total_rev:,.0f}", "completed trips")
    kpi(k2, "Total Trips",      f"{total_trips:,}",       "all statuses")
    kpi(k3, "Completed Trips",  f"{completed:,}",          "status = completed")
    kpi(k4, "Completion Rate",  f"{completion_rt:.1f}%",   "completed / total")
    kpi(k5, "Average Fare",     f"PKR {avg_fare:,.0f}",    "per completed trip")

    st.markdown("<div class='section-title'>Monthly Revenue Trend</div>", unsafe_allow_html=True)

    # ── Section B: Monthly Revenue ────────────────────────────────
    with st.spinner("Building revenue trend..."):
        monthly_df = query("""
            SELECT SUBSTR(trip_date,1,7) AS month,
                   SUM(fare_pkr)         AS revenue
            FROM trips
            WHERE status = 'completed'
            GROUP BY month
            ORDER BY month
        """)

    latest_month = monthly_df["month"].max()

    fig_line = px.line(
        monthly_df, x="month", y="revenue",
        markers=True,
        color_discrete_sequence=["#ffcc2f"],
        labels={"month": "Month", "revenue": "Revenue (PKR)"},
    )
    fig_line.update_traces(line_width=2.5, marker_size=6)
    fig_line.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=latest_month, x1=latest_month,
        y0=0, y1=1,
        line=dict(color="rgba(255,107,107,0.7)", dash="dash", width=2),
    )
    fig_line.add_annotation(
        x=latest_month, yref="paper", y=1.05,
        text="Latest",
        showarrow=False,
        font=dict(color="#ff6b6b", size=11),
    )

    apply_theme(fig_line, "Monthly Revenue Trend", height=360)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<div class='section-title'>City & Fleet Breakdown</div>", unsafe_allow_html=True)

    # ── Section C: Trips by City + Fleet Mix ─────────────────────
    with st.spinner("Loading city and fleet data..."):
        city_trips_df = query("""
            SELECT ci.city_name,
                   COUNT(*) AS trips
            FROM trips t
            JOIN cities ci ON t.city_id = ci.city_id
            WHERE t.status = 'completed'
            GROUP BY ci.city_name
            ORDER BY trips DESC
        """)

        fleet_df = query("""
            SELECT vehicle_type, COUNT(*) AS trips
            FROM trips
            WHERE status = 'completed'
            GROUP BY vehicle_type
        """)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_city = px.bar(
            city_trips_df, x="trips", y="city_name",
            orientation="h",
            color="trips",
            color_continuous_scale=[[0, "#1e2a4a"], [1, "#ffcc2f"]],
            labels={"trips": "Completed Trips", "city_name": ""},
        )
        fig_city.update_coloraxes(showscale=False)
        apply_theme(fig_city, "Trips by City")
        st.plotly_chart(fig_city, use_container_width=True)

    with col_right:
        fig_pie = px.pie(
            fleet_df, names="vehicle_type", values="trips",
            color="vehicle_type",
            color_discrete_map=VEHICLE_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(
            textfont_color="#f0f0ff",
            marker_line_width=2,
            marker_line_color="#13131f",
        )
        apply_theme(fig_pie, "Fleet Mix")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Section D: City Summary Table ────────────────────────────
    st.markdown("<div class='section-title'>City-Level Summary</div>", unsafe_allow_html=True)

    with st.spinner("Loading city summary..."):
        city_summary = query("""
            SELECT
                ci.city_name,
                COUNT(*)                                                         AS total_trips,
                ROUND(SUM(CASE WHEN t.status='completed' THEN t.fare_pkr END),0) AS total_revenue,
                ROUND(AVG(CASE WHEN t.status='completed' THEN t.fare_pkr END),0) AS avg_fare,
                ROUND(
                    SUM(CASE WHEN t.status='completed' THEN 1.0 ELSE 0 END)
                    / COUNT(*) * 100, 1
                )                                                                AS completion_rate_pct
            FROM trips t
            JOIN cities ci ON t.city_id = ci.city_id
            GROUP BY ci.city_name
            ORDER BY total_revenue DESC
        """)

    city_summary["total_revenue"] = city_summary["total_revenue"].apply(
        lambda x: f"PKR {x:,.0f}" if pd.notna(x) else "—"
    )
    city_summary["avg_fare"] = city_summary["avg_fare"].apply(
        lambda x: f"PKR {x:,.0f}" if pd.notna(x) else "—"
    )
    city_summary["completion_rate_pct"] = city_summary["completion_rate_pct"].apply(
        lambda x: f"{x}%" if pd.notna(x) else "—"
    )
    city_summary.columns = ["City", "Total Trips", "Total Revenue", "Avg Fare", "Completion Rate"]

    st.dataframe(
        city_summary,
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────
# PAGE 2 — TRIP ANALYTICS
# ─────────────────────────────────────────────────────────────────
elif page == "🗺️ Trip Analytics":

    st.markdown("<div class='page-title'>Trip Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Demand patterns, pricing dynamics, and weather impact</div>",
                unsafe_allow_html=True)

    # ── Section A: Heatmap ───────────────────────────────────────
    st.markdown("<div class='section-title'>Demand Heatmap</div>", unsafe_allow_html=True)

    with st.spinner("Building heatmap..."):
        heat_df = query("""
            SELECT trip_hour, day_of_week, COUNT(*) AS trips
            FROM trips
            GROUP BY trip_hour, day_of_week
        """)

    pivot = heat_df.pivot(index="day_of_week", columns="trip_hour", values="trips").fillna(0)
    pivot = pivot.reindex(range(7), fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)

    DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(range(24)),
        y=DAY_LABELS,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#c8c8e0"),
            title=dict(text="Trips", font=dict(color="#c8c8e0")),
        ),
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Trips: %{z}<extra></extra>",
    ))
    apply_theme(fig_heat, "Trip Demand Heatmap (Hour vs Day)", height=340)
    fig_heat.update_xaxes(title_text="Hour of Day", dtick=1)
    fig_heat.update_yaxes(title_text="")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Section B: Fare by vehicle + Scatter ─────────────────────
    st.markdown("<div class='section-title'>Fare Analysis</div>", unsafe_allow_html=True)

    with st.spinner("Loading fare data..."):
        avg_fare_df = query("""
            SELECT vehicle_type, ROUND(AVG(fare_pkr), 0) AS avg_fare
            FROM trips
            WHERE status = 'completed'
            GROUP BY vehicle_type
        """)

        scatter_df = query("""
            SELECT distance_km, fare_pkr, vehicle_type
            FROM trips
            WHERE status = 'completed'
            ORDER BY RANDOM()
            LIMIT 500
        """)

    col_l, col_r = st.columns(2)

    with col_l:
        fig_vfare = px.bar(
            avg_fare_df, x="vehicle_type", y="avg_fare",
            color="vehicle_type",
            color_discrete_map=VEHICLE_COLORS,
            labels={"vehicle_type": "Vehicle Type", "avg_fare": "Avg Fare (PKR)"},
            text_auto=True,
        )
        fig_vfare.update_traces(textfont_color="#13131f", textposition="outside")
        apply_theme(fig_vfare, "Average Fare by Vehicle Type")
        st.plotly_chart(fig_vfare, use_container_width=True)

    with col_r:
        fig_scat = px.scatter(
            scatter_df, x="distance_km", y="fare_pkr",
            color="vehicle_type",
            color_discrete_map=VEHICLE_COLORS,
            opacity=0.65,
            labels={"distance_km": "Distance (km)", "fare_pkr": "Fare (PKR)",
                    "vehicle_type": "Vehicle"},
        )
        apply_theme(fig_scat, "Fare vs Distance (sample of 500)")
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Section C: Peak vs Off-Peak ──────────────────────────────
    st.markdown("<div class='section-title'>Peak vs Off-Peak Pricing</div>", unsafe_allow_html=True)

    with st.spinner("Analysing peak hours..."):
        peak_df = query("""
            SELECT vehicle_type,
                   ROUND(AVG(CASE WHEN is_peak_hour=1 THEN fare_pkr END), 0) AS peak_fare,
                   ROUND(AVG(CASE WHEN is_peak_hour=0 THEN fare_pkr END), 0) AS offpeak_fare
            FROM trips
            WHERE status = 'completed'
            GROUP BY vehicle_type
        """)

    peak_melt = peak_df.melt(
        id_vars="vehicle_type",
        value_vars=["peak_fare", "offpeak_fare"],
        var_name="period",
        value_name="avg_fare",
    )
    peak_melt["period"] = peak_melt["period"].map(
        {"peak_fare": "🔴 Peak Hours", "offpeak_fare": "🟢 Off-Peak"}
    )

    fig_peak = px.bar(
        peak_melt, x="vehicle_type", y="avg_fare",
        color="period",
        barmode="group",
        color_discrete_sequence=["#ff6b6b", "#4ecdc4"],
        labels={"vehicle_type": "Vehicle Type", "avg_fare": "Avg Fare (PKR)", "period": ""},
        text_auto=True,
    )
    fig_peak.update_traces(textfont_color="#13131f")
    apply_theme(fig_peak, "Peak vs Off-Peak Fares by Vehicle Type", height=360)
    st.plotly_chart(fig_peak, use_container_width=True)

    # ── Section D: Rain Impact ───────────────────────────────────
    st.markdown("<div class='section-title'>Rain Impact</div>", unsafe_allow_html=True)

    with st.spinner("Calculating rain stats..."):
        rain_df = query("""
            SELECT
                ROUND(AVG(CASE WHEN is_raining=1 THEN fare_pkr END), 0)  AS rain_fare,
                ROUND(AVG(CASE WHEN is_raining=0 THEN fare_pkr END), 0)  AS dry_fare
            FROM trips WHERE status='completed'
        """)

        rain_daily = query("""
            SELECT is_raining,
                   COUNT(*) * 1.0 / COUNT(DISTINCT trip_date) AS trips_per_day
            FROM trips
            GROUP BY is_raining
        """)

    rain_fare = int(rain_df["rain_fare"].iloc[0])
    dry_fare  = int(rain_df["dry_fare"].iloc[0])

    rain_row = rain_daily[rain_daily["is_raining"] == 1]
    dry_row  = rain_daily[rain_daily["is_raining"] == 0]
    rain_tpd = round(rain_row["trips_per_day"].iloc[0], 1) if not rain_row.empty else 0
    dry_tpd  = round(dry_row["trips_per_day"].iloc[0], 1) if not dry_row.empty else 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("☔ Avg Fare (Rain)",    f"PKR {rain_fare:,}")
    mc2.metric("☀️ Avg Fare (Dry)",    f"PKR {dry_fare:,}",
               delta=f"–{rain_fare - dry_fare:,} PKR vs rain", delta_color="normal")
    mc3.metric("☔ Trips/Day (Rain)",  f"{rain_tpd}")
    mc4.metric("☀️ Trips/Day (Dry)",  f"{dry_tpd}")

    st.info("💡 **Surge Pricing:** During rainy weather, SwiftRide applies a **1.5×** surge multiplier "
            "for peak-hour trips and **1.3×** surge for off-peak rainy trips. "
            "This balances driver incentives with increased demand on wet days.")


# ─────────────────────────────────────────────────────────────────
# PAGE 3 — DRIVER PERFORMANCE
# ─────────────────────────────────────────────────────────────────
elif page == "🏆 Driver Performance":

    st.markdown("<div class='page-title'>Driver Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Leaderboards, ratings, and earnings breakdown by driver</div>",
                unsafe_allow_html=True)

    # ── Section A: Top 10 Leaderboard ────────────────────────────
    st.markdown("<div class='section-title'>Top 10 Drivers — Earnings Leaderboard</div>",
                unsafe_allow_html=True)

    with st.spinner("Loading leaderboard..."):
        leaderboard = query("""
            SELECT
                d.name                                              AS driver,
                ci.city_name                                        AS city,
                d.vehicle_type,
                COUNT(CASE WHEN t.status='completed' THEN 1 END)    AS completed_trips,
                ROUND(SUM(CASE WHEN t.status='completed'
                               THEN t.fare_pkr END), 0)             AS total_earnings,
                ROUND(AVG(r.rider_rating_given), 2)                 AS avg_rating,
                ROUND(
                    COUNT(CASE WHEN t.status='completed' THEN 1.0 END)
                    / COUNT(t.trip_id) * 100, 1
                )                                                   AS completion_rate_pct
            FROM drivers d
            JOIN trips   t  ON d.driver_id = t.driver_id
            JOIN cities  ci ON d.city_id   = ci.city_id
            LEFT JOIN reviews r ON r.driver_id = d.driver_id
                               AND r.trip_id   = t.trip_id
            GROUP BY d.driver_id
            ORDER BY total_earnings DESC
            LIMIT 10
        """)

    leaderboard["total_earnings"] = leaderboard["total_earnings"].apply(
        lambda x: f"PKR {x:,.0f}" if pd.notna(x) else "—"
    )
    leaderboard["completion_rate_pct"] = leaderboard["completion_rate_pct"].apply(
        lambda x: f"{x}%" if pd.notna(x) else "—"
    )
    leaderboard.columns = ["Driver", "City", "Vehicle", "Trips",
                           "Total Earnings", "Avg Rating", "Completion Rate"]

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    # ── Section B: Rating by city + Earnings per trip ────────────
    st.markdown("<div class='section-title'>Ratings & Earnings</div>", unsafe_allow_html=True)

    with st.spinner("Loading rating and earnings data..."):
        city_rating = query("""
            SELECT ci.city_name,
                   ROUND(AVG(r.rider_rating_given), 2) AS avg_rating
            FROM reviews r
            JOIN trips  t  ON r.trip_id = t.trip_id
            JOIN cities ci ON t.city_id = ci.city_id
            GROUP BY ci.city_name
            ORDER BY avg_rating DESC
        """)

        earn_df = query("""
            SELECT vehicle_type,
                   ROUND(AVG(fare_pkr), 0) AS avg_earnings_per_trip
            FROM trips
            WHERE status='completed'
            GROUP BY vehicle_type
        """)

    col_l, col_r = st.columns(2)

    with col_l:
        fig_crat = px.bar(
            city_rating, x="avg_rating", y="city_name",
            orientation="h",
            color="avg_rating",
            color_continuous_scale=[[0, "#1e2a4a"], [1, "#ffcc2f"]],
            labels={"avg_rating": "Avg Rating", "city_name": ""},
            text_auto=True,
        )
        fig_crat.update_coloraxes(showscale=False)
        fig_crat.update_xaxes(range=[3.5, 5.0])
        apply_theme(fig_crat, "Average Driver Rating by City")
        st.plotly_chart(fig_crat, use_container_width=True)

    with col_r:
        fig_earn = px.bar(
            earn_df, x="vehicle_type", y="avg_earnings_per_trip",
            color="vehicle_type",
            color_discrete_map=VEHICLE_COLORS,
            labels={"vehicle_type": "Vehicle Type",
                    "avg_earnings_per_trip": "Avg Earnings / Trip (PKR)"},
            text_auto=True,
        )
        fig_earn.update_traces(textfont_color="#13131f")
        apply_theme(fig_earn, "Average Earnings Per Trip by Vehicle Type")
        st.plotly_chart(fig_earn, use_container_width=True)

    # ── Section C: Active Drivers per Month ──────────────────────
    st.markdown("<div class='section-title'>Driver Activity Over Time</div>",
                unsafe_allow_html=True)

    with st.spinner("Calculating monthly active drivers..."):
        active_monthly = query("""
            SELECT SUBSTR(trip_date,1,7) AS month,
                   COUNT(DISTINCT driver_id) AS active_drivers
            FROM trips
            WHERE status='completed'
            GROUP BY month
            ORDER BY month
        """)

    fig_act = px.line(
        active_monthly, x="month", y="active_drivers",
        markers=True,
        color_discrete_sequence=["#4ecdc4"],
        labels={"month": "Month", "active_drivers": "Active Drivers"},
    )
    fig_act.update_traces(line_width=2.5, marker_size=7,
                          fill="tozeroy",
                          fillcolor="rgba(78,205,196,0.08)")
    apply_theme(fig_act, "Active Drivers Per Month", height=340)
    st.plotly_chart(fig_act, use_container_width=True)

    # ── Section D: Rating Distribution ───────────────────────────
    st.markdown("<div class='section-title'>Driver Rating Distribution</div>",
                unsafe_allow_html=True)

    with st.spinner("Loading rating distribution..."):
        ratings_df = query("""
            SELECT rider_rating_given FROM reviews
        """)

    fig_hist = px.histogram(
        ratings_df, x="rider_rating_given",
        nbins=9,
        color_discrete_sequence=["#a78bfa"],
        labels={"rider_rating_given": "Rating Given by Rider", "count": "Count"},
    )
    fig_hist.update_traces(marker_line_width=1.5, marker_line_color="#13131f")
    apply_theme(fig_hist, "Distribution of Driver Ratings", height=340)
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 4 — ML FARE PREDICTOR
# ─────────────────────────────────────────────────────────────────
elif page == "🤖 ML Fare Predictor":

    st.markdown("<div class='page-title'>ML Fare Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Random Forest model trained on historical trip data</div>",
                unsafe_allow_html=True)

    # ── Model Training (cached) ───────────────────────────────────
    @st.cache_resource
    def train_model():
        df = query("""
            SELECT distance_km, duration_mins, is_peak_hour, is_raining,
                   surge_multiplier, day_of_week, trip_hour,
                   vehicle_type, fare_pkr
            FROM trips
            WHERE status = 'completed'
        """)

        ohe = pd.get_dummies(df["vehicle_type"], prefix="vtype")
        X   = pd.concat([
            df[["distance_km", "duration_mins", "is_peak_hour", "is_raining",
                "surge_multiplier", "day_of_week", "trip_hour"]],
            ohe,
        ], axis=1)
        y = df["fare_pkr"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return model, r2, mae, rmse, X.columns.tolist(), y_test, y_pred

    with st.spinner("Training Random Forest model..."):
        model, r2, mae, rmse, feature_cols, y_test, y_pred = train_model()

    # ── Section A: Model Metrics ──────────────────────────────────
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score",   f"{r2:.4f}")
    mc2.metric("MAE",        f"PKR {mae:,.1f}")
    mc3.metric("RMSE",       f"PKR {rmse:,.1f}")

    if r2 > 0.85:
        st.success(f"✅ Excellent model fit! R² = {r2:.4f} — the model explains "
                   f"{r2*100:.1f}% of fare variance.")
    else:
        st.warning(f"⚠️ Model R² = {r2:.4f}. Consider adding more features or "
                   "tuning hyperparameters to improve accuracy.")

    # ── Section B: Feature Importance ────────────────────────────
    st.markdown("<div class='section-title'>What Drives Fare Prices?</div>",
                unsafe_allow_html=True)

    feat_imp = pd.DataFrame({
        "feature":   feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    NICE_NAMES = {
        "distance_km":     "Distance (km)",
        "duration_mins":   "Duration (mins)",
        "surge_multiplier":"Surge Multiplier",
        "is_peak_hour":    "Peak Hour",
        "trip_hour":       "Trip Hour",
        "day_of_week":     "Day of Week",
        "is_raining":      "Raining",
        "vtype_Bike":      "Vehicle: Bike",
        "vtype_Car":       "Vehicle: Car",
        "vtype_Rickshaw":  "Vehicle: Rickshaw",
        "vtype_SUV":       "Vehicle: SUV",
    }
    feat_imp["feature"] = feat_imp["feature"].map(lambda x: NICE_NAMES.get(x, x))

    fig_fi = px.bar(
        feat_imp, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=[[0, "#1e2a4a"], [1, "#ffcc2f"]],
        labels={"importance": "Importance", "feature": ""},
        text=feat_imp["importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig_fi.update_coloraxes(showscale=False)
    fig_fi.update_traces(textposition="outside", textfont_color="#c8c8e0")
    apply_theme(fig_fi, "What Drives Fare Prices?", height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Section C: Live Predictor ─────────────────────────────────
    st.markdown("<div class='section-title'>🔮 Predict a Fare</div>", unsafe_allow_html=True)

    PEAK_HOURS_SET = set(range(7, 10)) | set(range(17, 21))
    DAY_MAP = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    with st.spinner("Loading average fares..."):
        avg_by_vehicle = query("""
            SELECT vehicle_type, ROUND(AVG(fare_pkr),0) AS avg_fare
            FROM trips WHERE status='completed'
            GROUP BY vehicle_type
        """).set_index("vehicle_type")["avg_fare"].to_dict()

    inp1, inp2 = st.columns(2)

    with inp1:
        sel_vehicle   = st.selectbox("Vehicle Type", ["Bike", "Rickshaw", "Car", "SUV"])
        sel_distance  = st.slider("Distance (km)", 1.0, 30.0, 8.0, 0.5)
        sel_hour      = st.slider("Trip Hour (0–23)", 0, 23, 8)

    with inp2:
        sel_day       = st.selectbox("Day of Week", list(DAY_MAP.keys()))
        sel_raining   = st.checkbox("☔ Currently raining?")

    # Derive auto fields
    is_peak    = 1 if sel_hour in PEAK_HOURS_SET else 0
    if sel_raining and is_peak:
        surge = 1.5
    elif is_peak:
        surge = 1.3
    else:
        surge = 1.0
    duration_est = sel_distance * 4.0
    day_int      = DAY_MAP[sel_day]

    # Build feature vector aligned with training columns
    input_dict = {
        "distance_km":    sel_distance,
        "duration_mins":  duration_est,
        "is_peak_hour":   is_peak,
        "is_raining":     int(sel_raining),
        "surge_multiplier": surge,
        "day_of_week":    day_int,
        "trip_hour":      sel_hour,
        "vtype_Bike":     1 if sel_vehicle == "Bike"     else 0,
        "vtype_Car":      1 if sel_vehicle == "Car"      else 0,
        "vtype_Rickshaw": 1 if sel_vehicle == "Rickshaw" else 0,
        "vtype_SUV":      1 if sel_vehicle == "SUV"      else 0,
    }
    input_df       = pd.DataFrame([input_dict])[feature_cols]
    predicted_fare = model.predict(input_df)[0]
    predicted_fare = max(30, round(predicted_fare / 10) * 10)

    db_avg  = avg_by_vehicle.get(sel_vehicle, 0)
    diff    = predicted_fare - db_avg
    diff_str = f"+PKR {abs(diff):,.0f} above avg" if diff >= 0 else f"PKR {abs(diff):,.0f} below avg"

    st.markdown("<br>", unsafe_allow_html=True)
    res_col, detail_col = st.columns([1, 2])

    with res_col:
        st.metric(
            label=f"Predicted Fare — {sel_vehicle}",
            value=f"PKR {predicted_fare:,.0f}",
            delta=diff_str,
        )

    with detail_col:
        conds = []
        if is_peak:      conds.append("🔴 Peak Hour (+30%)")
        if sel_raining:  conds.append("☔ Rain Surge (+50%)")
        if not conds:    conds.append("✅ Standard pricing")

        st.markdown(f"""
        | Parameter         | Value          |
        |-------------------|----------------|
        | Distance          | {sel_distance} km |
        | Est. Duration     | {duration_est:.0f} mins |
        | Surge Multiplier  | {surge}× |
        | Day               | {sel_day} |
        | Hour              | {sel_hour:02d}:00 |
        | Conditions        | {", ".join(conds)} |
        | DB Average Fare   | PKR {db_avg:,.0f} |
        """)

    # ── Section D: Actual vs Predicted Scatter ───────────────────
    st.markdown("<div class='section-title'>Actual vs Predicted Fares (Test Set)</div>",
                unsafe_allow_html=True)

    avp_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
    sample = avp_df.sample(min(800, len(avp_df)), random_state=42)

    fig_avp = px.scatter(
        sample, x="actual", y="predicted",
        opacity=0.5,
        color_discrete_sequence=["#a78bfa"],
        labels={"actual": "Actual Fare (PKR)", "predicted": "Predicted Fare (PKR)"},
    )
    min_val = min(sample["actual"].min(), sample["predicted"].min())
    max_val = max(sample["actual"].max(), sample["predicted"].max())
    fig_avp.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color="#ffcc2f", dash="dash", width=2),
    )
    fig_avp.add_annotation(
        x=max_val * 0.85, y=max_val * 0.9,
        text="Perfect Prediction",
        showarrow=False,
        font=dict(color="#ffcc2f", size=11),
    )
    apply_theme(fig_avp, "Actual vs Predicted Fares", height=420)
    st.plotly_chart(fig_avp, use_container_width=True)
