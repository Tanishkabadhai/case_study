"""
Logistics — Delivery Experience Decline During the Festive Surge

"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def two_proportion_p_value(count1, n1, count2, n2):
    """Two-sided p-value for a two-proportion z-test, normal approximation.
    No scipy dependency — implemented with math.erf so the app only needs
    numpy/pandas/plotly/streamlit."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = count1 / n1, count2 / n2
    pooled = (count1 + count2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Delivery Experience Decline — Festive Surge",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


INK = "#1A1A2E"
ACCENT = "#E6553F"
GOOD = "#2E8B57"
BAD = "#C0392B"
NEUTRAL_BLUE = "#5C6BC0"

st.markdown(f"""
<style>
    .hero {{
        padding: 1.6rem 2rem;
        border-radius: 12px;
        background: {INK};
        margin-bottom: 1.2rem;
    }}
    .hero h1 {{ color: #FFFFFF; margin-bottom: 0.3rem; font-size: 1.9rem; }}
    .hero p  {{ color: #D7D7E8; font-size: 1.02rem; margin: 0; }}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------
# DATA LOADING + CALCULATED FIELDS  (Phase 0)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR / "orders.csv",
                          parse_dates=["order_date", "promised_date", "delivery_date"])
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    nps = pd.read_csv(DATA_DIR / "nps.csv", parse_dates=["response_date"])
    complaints = pd.read_csv(DATA_DIR / "complaints.csv", parse_dates=["created_date"])
    hub = pd.read_csv(DATA_DIR / "hub_performance.csv")
    courier = pd.read_csv(DATA_DIR / "courier_performance.csv")

    # --- Delivery Delay & SLA Breach ---
  
    orders["delivery_delay"] = (orders["delivery_date"] - orders["promised_date"]).dt.days
    orders["sla_breach"] = np.where(orders["order_status"] == "Delivered",
                                     orders["delivery_delay"] > 0, np.nan)
    orders["order_month"] = orders["order_date"].dt.to_period("M").astype(str)
    orders["has_complaint"] = orders["order_id"].isin(complaints["order_id"])

    orders_c = orders.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")

    # --- NPS categorisation + join to the ORDER's month ---
 
    nps["category"] = pd.cut(nps["score"], bins=[-1, 6, 8, 10],
                              labels=["Detractor", "Passive", "Promoter"])
    nps_full = nps.merge(
        orders_c[["order_id", "order_month", "city", "courier_partner", "segment", "delivery_delay"]],
        on="order_id", how="left"
    )

    # --- Complaints enriched with order context ---
    comp_full = complaints.merge(
        orders_c[["order_id", "city", "courier_partner", "order_status", "order_month"]],
        on="order_id", how="left"
    )

    # --- Hub performance derived rates ---
    hub = hub.copy()
    hub["sla_breach_pct"] = (1 - hub["on_time_delivery"] / hub["total_orders"]) * 100
    hub["failed_attempt_pct"] = hub["failed_attempts"] / hub["total_orders"] * 100
    hub["rto_pct"] = hub["rto_count"] / hub["total_orders"] * 100

   
    delivered_only = orders_c[orders_c["order_status"] == "Delivered"]
    courier_computed = pd.DataFrame({
        "sla_breach_rate_computed": delivered_only.groupby("courier_partner")["sla_breach"].mean(),
        "complaint_rate_computed": orders_c.groupby("courier_partner")["has_complaint"].mean(),
    }).reset_index()

    return orders_c, customers, nps_full, comp_full, hub, courier, courier_computed


orders, customers, nps, complaints, hub, courier, courier_computed = load_data()
delivered = orders[orders["order_status"] == "Delivered"].copy()

MONTH_ORDER = sorted(orders["order_month"].unique())
MONTH_LABELS = {m: pd.Period(m).strftime("%b %Y") for m in MONTH_ORDER}
TIER_MAP = {"Mumbai": "Tier-1", "Pune": "Tier-1", "Nagpur": "Tier-2", "Indore": "Tier-2"}


def nps_score(df):
    """Standard NPS formula: %Promoters - %Detractors"""
    n = len(df)
    if n == 0:
        return 0.0
    return (df["category"].eq("Promoter").mean() - df["category"].eq("Detractor").mean()) * 100


def nps_by_group(df, group_col):
    """NPS score computed separately within each value of group_col —
    an explicit loop rather than groupby().apply() so behaviour is
    identical across pandas versions."""
    out = {}
    for g, sub in df.groupby(group_col, observed=True):
        out[g] = nps_score(sub)
    return pd.Series(out)


# --------------------------------------------------------------------------
# HERO BANNER + TOP-LEVEL KPIs
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📦 Delivery Experience Decline — Festive Surge (Oct–Dec 2025)</h1>
    <p>Why NPS is falling, complaints are rising, and what to fix before the next peak season.</p>
</div>
""", unsafe_allow_html=True)

overall_nps = nps_score(nps)
overall_sla_breach = delivered["sla_breach"].mean() * 100
complaint_rate = orders["has_complaint"].mean() * 100
rto_rate = (orders["order_status"] == "RTO").mean() * 100
nps_by_month = nps_by_group(nps, "order_month").reindex(MONTH_ORDER)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall NPS (3 mo.)", f"{overall_nps:.0f}")
k2.metric("SLA Breach Rate", f"{overall_sla_breach:.0f}%", "of delivered orders", delta_color="off")
k3.metric("Complaint Rate", f"{complaint_rate:.1f}%", "of all orders", delta_color="off")
k4.metric("RTO Rate", f"{rto_rate:.1f}%", "of all orders", delta_color="off")
k5.metric("Total Orders", f"{len(orders):,}", f"{orders['city'].nunique()} cities", delta_color="off")

with st.expander("Data notes & assumptions"):
    st.markdown(
        "- **SLA Breach Rate above (headline KPI) is a calculated field**: "
        "`delivery_date − promised_date > 0`, on Delivered orders only, exactly as the brief "
        "specifies. It is **not** the same figure as `sla_breach_rate` in the provided "
        "`courier_performance.csv` — that file is a separate, independently-supplied summary "
        "and its per-courier rates do not reconcile with the order-level calculation under any "
        "delay threshold we tested. Both are shown in Section B, each labeled with its source, "
        "rather than presented as if they measure the same thing.\n"
        "- **`hub_performance.csv` order counts also don't match `orders.csv`** city-level "
        "order counts (e.g. it reports 600 Mumbai orders; the order-level data has 458). Treat "
        "the hub file as management-reported summary stats, not a derivable aggregate of the "
        "order-level data.\n"
        "- **Tier-1/Tier-2 is an analyst assumption**, not a field in the data: Mumbai & Pune "
        "are treated as Tier-1, Nagpur & Indore as Tier-2, based on the cities named in the "
        "case context.\n"
        "- **`feedback_text` in the NPS data is a 5-value fixed field**, not free text — there's "
        "nothing to theme-extract. It also doesn't track the numeric score reliably (a "
        "meaningful share of Detractors have feedback_text = \"Satisfied\"), so it's used only "
        "as supporting color, never as the basis for a finding."
    )

st.write("")

tabs = st.tabs([
    " Executive Summary",
    "A · NPS & Customer Experience",
    "B · Operational Performance",
    "C · Tier-2 Deep Dive",
    "D · End-to-End Funnel",
    "E · Recommendations",
])

# ==========================================================================
# TAB 0 — EXECUTIVE SUMMARY
# ==========================================================================
with tabs[0]:
    st.subheader("Executive Summary")

    worst_courier = courier.loc[courier["sla_breach_rate"].idxmax(), "courier_partner"]
    worst_courier_rate = courier["sla_breach_rate"].max() * 100
    worst_hub_city = hub.loc[hub["sla_breach_pct"].idxmax(), "city"]
    worst_hub_rate = hub["sla_breach_pct"].max()

    st.info(
        f"Across Oct–Dec, **{overall_sla_breach:.0f}% of delivered orders were late.** "
        f"Overall NPS was **{overall_nps:.0f}**, showing that Detractors were much more common "
        f"than Promoters. **{worst_courier}** had the highest SLA breach rate among the three "
        f"couriers ({worst_courier_rate:.0f}%), while **{worst_hub_city}** had the highest "
        f"hub-level SLA breach rate ({worst_hub_rate:.0f}%)."
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[MONTH_LABELS[m] for m in MONTH_ORDER], y=nps_by_month.values,
            mode="lines+markers+text", text=[f"{v:.0f}" for v in nps_by_month.values],
            textposition="top center", line=dict(color=ACCENT, width=3), marker=dict(size=10),
        ))
        fig.update_layout(title="NPS trend, month on month", height=320,
                           yaxis_title="NPS score", margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hub_sorted = hub.sort_values("sla_breach_pct")
        fig2 = px.bar(hub_sorted, x="sla_breach_pct", y="city", orientation="h",
                       color="sla_breach_pct", color_continuous_scale=[GOOD, BAD],
                       labels={"sla_breach_pct": "SLA breach %", "city": ""})
        fig2.update_layout(title="SLA breach % by hub city", height=320, coloraxis_showscale=False,
                            margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Top 3  key findings")
    r1, r2, r3 = st.columns(3)
    r1.markdown(f"**1. {worst_courier} has the highest delay and complaint rates**")
    r1.caption(f"{worst_courier_rate:.0f}% SLA breach rate, which is the highest among the three couriers.")
    r2.markdown(f"**2. {worst_hub_city} has the highest SLA breach rate**")
    r2.caption(f"{worst_hub_rate:.0f}% SLA breach in the supplied hub data. "
    "Failed attempts and RTO are also relatively high.")
    r3.markdown("**3. Delays are strongly associated with complaints**")
    r3.caption("Most complaints are linked to delayed orders, and complaint-linked NPS responses "
    "are heavily concentrated among Detractors.")

# ==========================================================================
# TAB A — NPS & CUSTOMER EXPERIENCE
# ==========================================================================
with tabs[1]:
    st.subheader("A · NPS & Customer Experience")

    st.info(
         f"Overall NPS for the period is **{overall_nps:.0f}** — Detractors far outnumber Promoters. "
         f"NPS improves from October ({nps_by_month.iloc[0]:.0f}) to November "
         f"({nps_by_month.iloc[1]:.0f}), then slides back down in December "
         f"({nps_by_month.iloc[-1]:.0f})."
    )
    st.caption(
        "Note: NPS is trended by the month the underlying **order** was placed, not the month the "
        "customer happened to respond — a handful of responses to December orders arrive in early "
        "January, and grouping by response date would drop that feedback out of scope entirely."
    )

    c1, c2 = st.columns(2)
    with c1:
        cat_month = nps.groupby(["order_month", "category"], observed=True).size().unstack(fill_value=0).reindex(MONTH_ORDER)
        cat_month.index = [MONTH_LABELS[m] for m in cat_month.index]
        fig = px.bar(cat_month, barmode="stack",
                     color_discrete_map={"Promoter": GOOD, "Passive": "#D9A441", "Detractor": BAD})
        fig.update_layout(title="Promoter / Passive / Detractor mix by month", height=360,
                           legend_title="", yaxis_title="Responses")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_scores = nps_by_group(nps, "segment").sort_values()
        fig = px.bar(seg_scores, orientation="h", color=seg_scores.values,
                     color_continuous_scale=[BAD, GOOD])
        fig.update_layout(title="NPS by customer segment", height=360, coloraxis_showscale=False,
                           xaxis_title="NPS score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"**So what:** the **{seg_scores.index[0]}** segment posts the lowest NPS "
        f"({seg_scores.iloc[0]:.0f}) of any segment. The surge is hurting the customers the business "
        "can least afford to lose — not just casual, one-off buyers."
    )

    c3, c4 = st.columns(2)
    with c3:
      
        detractors = nps[nps["category"] == "Detractor"]
        det_issues = detractors.merge(
            complaints[["order_id", "issue_type"]].drop_duplicates("order_id"),
            on="order_id", how="left"
        )
        det_issue_counts = det_issues["issue_type"].value_counts(dropna=False).rename(
            index={np.nan: "No linked complaint ticket"}
        ).head(6)
        fig = px.bar(det_issue_counts, orientation="h", color_discrete_sequence=[BAD])
        fig.update_layout(title="Top drivers of Detractors (linked complaint issue_type)", height=340,
                           xaxis_title="Detractor responses", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        no_ticket_pct = det_issues["issue_type"].isna().mean() * 100
        st.caption(f"{no_ticket_pct:.0f}% of Detractors have no linked complaint ticket at all — "
                   "their dissatisfaction isn't showing up in the complaints channel.")
    with c4:
        st.markdown("**Does lateness drive the score down?**")
        delay_bucket = pd.cut(nps["delivery_delay"], bins=[-10, 0, 1, 3, 5, 100],
                              labels=["On time/early", "1 day late", "2-3 days late",
                                      "4-5 days late", "6+ days late"])
        bucket_scores = nps_by_group(nps.assign(_bucket=delay_bucket), "_bucket")
        fig = px.line(bucket_scores, markers=True, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=340, xaxis_title="Delivery delay bucket", yaxis_title="NPS score",
                           showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("NPS drops sharply once a delivery is even 1 day late, and keeps falling as the "
                   "delay grows.")

# ==========================================================================
# TAB B — OPERATIONAL PERFORMANCE
# ==========================================================================
with tabs[2]:
    st.subheader("B · Operational Performance")

    sla_city = delivered.groupby("city")["sla_breach"].mean().sort_values(ascending=False) * 100
    worst_city = sla_city.index[0]

    st.info(
        f"**{worst_city}** has the highest SLA breach rate at **{sla_city.iloc[0]:.0f}%** of delivered "
        f"orders. Across couriers, **{worst_courier}** is the clear outlier — "
        f"{courier.set_index('courier_partner').loc[worst_courier, 'sla_breach_rate']*100:.0f}% SLA "
        f"breach rate and {courier.set_index('courier_partner').loc[worst_courier, 'complaint_rate']*100:.0f}% "
        "complaint rate, both the highest of the three partners."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sla_city, color=sla_city.values, color_continuous_scale=[GOOD, BAD])
        fig.update_layout(title="SLA breach % by city (delivered orders)", height=340,
                           coloraxis_showscale=False, xaxis_title="", yaxis_title="SLA breach %",
                           showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cp = courier.set_index("courier_partner")[["sla_breach_rate", "complaint_rate"]] * 100
        fig = px.bar(cp, barmode="group", color_discrete_sequence=[NEUTRAL_BLUE, BAD])
        fig.update_layout(title="Courier partner: SLA breach % vs. complaint %", height=340,
                           xaxis_title="", yaxis_title="%", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(hub, x="failed_attempt_pct", y="rto_pct", size="total_orders", color="city",
                          text="city", size_max=45)
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Failed delivery attempts vs. RTO rate (by hub)", height=360,
                           xaxis_title="Failed attempt %", yaxis_title="RTO %")
        st.plotly_chart(fig, use_container_width=True)
        corr = hub["failed_attempt_pct"].corr(hub["rto_pct"])
        st.caption(f"Correlation across hubs: r = {corr:.2f} — hubs with more failed delivery "
                   "attempts also return more parcels to origin.")
    with c4:
        st.markdown("**Hub performance summary**")
        hub_disp = hub[["hub_id", "city", "total_orders", "sla_breach_pct",
                         "failed_attempt_pct", "rto_pct"]].copy()
        hub_disp.columns = ["Hub", "City", "Orders", "SLA Breach %", "Failed Attempt %", "RTO %"]
        for col in ["SLA Breach %", "Failed Attempt %", "RTO %"]:
            hub_disp[col] = hub_disp[col].round(1)
        st.dataframe(hub_disp.sort_values("SLA Breach %", ascending=False),
                     use_container_width=True, hide_index=True)

    with st.expander("Why don't these numbers match the top-of-page SLA Breach Rate KPI?"):
        cc_disp = courier.merge(courier_computed, on="courier_partner")
        cc_disp["sla_breach_rate"] = (cc_disp["sla_breach_rate"] * 100).round(1)
        cc_disp["complaint_rate"] = (cc_disp["complaint_rate"] * 100).round(1)
        cc_disp["sla_breach_rate_computed"] = (cc_disp["sla_breach_rate_computed"] * 100).round(1)
        cc_disp["complaint_rate_computed"] = (cc_disp["complaint_rate_computed"] * 100).round(1)
        cc_disp = cc_disp.rename(columns={
            "courier_partner": "Courier", "sla_breach_rate": "SLA Breach % (given file)",
            "complaint_rate": "Complaint % (given file)",
            "sla_breach_rate_computed": "SLA Breach % (computed from orders.csv)",
            "complaint_rate_computed": "Complaint % (computed from orders.csv)",
        })
        st.dataframe(cc_disp, use_container_width=True, hide_index=True)
        st.caption(
            "The provided `courier_performance.csv` and the delay/SLA calculated field built "
            "from `orders.csv` per the brief don't reconcile — absolute rates differ substantially "
            "on both metrics. **QuickShip is the worst courier under either source** (used "
            "throughout this app), but FastEx and ShipNow swap places between the two — the given "
            "file ranks ShipNow worse than FastEx, the order-level calculation ranks them the other "
            "way. Both tables are shown rather than picking one and hiding the disagreement; "
            "conclusions here only rely on the QuickShip finding, which holds either way."
        )

# ==========================================================================
# TAB C — TIER-2 DEEP DIVE
# ==========================================================================
with tabs[3]:
    st.subheader('C · Problem Deep Dive — "Tier-2 cities show higher complaint rates"')
    st.caption("City tiers aren't labeled in the data — we apply the standard classification: "
               "Mumbai & Pune = Tier-1, Nagpur & Indore = Tier-2.")

    

    comp_rate_city = orders.groupby("city")["has_complaint"].mean().sort_values(ascending=False) * 100
    courier_mix = pd.crosstab(orders["city"], orders["courier_partner"], normalize="index") * 100
    orders["tier"] = orders["city"].map(TIER_MAP)
    tier_counts = orders.groupby("tier")["has_complaint"].agg(["sum", "count"])
    tier_rate = (tier_counts["sum"] / tier_counts["count"]) * 100
    tier_p = two_proportion_p_value(
        tier_counts.loc["Tier-2", "sum"], tier_counts.loc["Tier-2", "count"],
        tier_counts.loc["Tier-1", "sum"], tier_counts.loc["Tier-1", "count"],
    )
    hub["tier"] = hub["city"].map(TIER_MAP)
    tier_hub_avg = hub.groupby("tier")[["sla_breach_pct", "failed_attempt_pct", "rto_pct"]].mean()

    st.info(
        f"**The complaint-rate difference between Tier-2 and Tier-1 is small and not statistically "
        f"significant in this dataset**: Tier-2 {tier_rate['Tier-2']:.1f}% vs Tier-1 "
        f"{tier_rate['Tier-1']:.1f}% (two-proportion z-test, p = {tier_p:.2f}). "
        f"However, Tier-2 hubs show higher SLA breach rates: "
        f"{tier_hub_avg.loc['Tier-2', 'sla_breach_pct']:.0f}% vs "
        f"{tier_hub_avg.loc['Tier-1', 'sla_breach_pct']:.0f}% in Tier-1. "
        f"Failed-attempt rates are {tier_hub_avg.loc['Tier-2','failed_attempt_pct']:.0f}% vs "
        f"{tier_hub_avg.loc['Tier-1','failed_attempt_pct']:.0f}%, while RTO rates are "
        f"{tier_hub_avg.loc['Tier-2','rto_pct']:.0f}% vs "
        f"{tier_hub_avg.loc['Tier-1','rto_pct']:.0f}%. "
        "Courier mix is broadly similar across the cities, so courier mix alone does not "
        "appear to explain the difference. This makes hub-level operational factors worth "
        "investigating further."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comp_rate_city, color=comp_rate_city.values, color_continuous_scale=[GOOD, BAD])
        fig.update_layout(title="Complaint rate % by city", height=340, coloraxis_showscale=False,
                           xaxis_title="", yaxis_title="Complaint %", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.imshow(courier_mix.round(0), text_auto=True, color_continuous_scale="Blues",
                         labels=dict(color="% of city volume"))
        fig.update_layout(title="Courier mix is even across cities", height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        hub[["city", "failed_attempt_pct", "rto_pct", "sla_breach_pct"]]
        .rename(columns={"failed_attempt_pct": "Failed Attempt %", "rto_pct": "RTO %",
                          "sla_breach_pct": "SLA Breach %", "city": "City"})
        .sort_values("SLA Breach %", ascending=False).round(1),
        use_container_width=True, hide_index=True,
    )

    st.success(
        "**Verdict:** the complaint-rate difference between Tier-2 and Tier-1 is not statistically "
        "significant in this dataset. However, Tier-2 hubs show higher SLA breach, failed-attempt, "
        "and RTO rates. Courier mix is broadly similar across cities, so hub-level operational "
        "factors are worth investigating further. The data supports prioritising hub operations "
        "alongside courier performance rather than treating courier selection as the only issue."
    )

# ==========================================================================
# TAB D — END-TO-END FUNNEL
# ==========================================================================
with tabs[4]:
    st.subheader("D · End-to-End Funnel: Orders → Delivery → Complaints → NPS → Repeat")

  
    delivered_c = delivered.copy()
    delayed_mask = delivered_c["sla_breach"] == True
    n_delayed = int(delayed_mask.sum())
    delayed_to_complaint = delivered_c.loc[delayed_mask, "has_complaint"].mean() * 100
    ontime_to_complaint = delivered_c.loc[~delayed_mask, "has_complaint"].mean() * 100

    delayed_complained_ids = delivered_c.loc[delayed_mask & delivered_c["has_complaint"], "order_id"]
    n_delayed_complained = len(delayed_complained_ids)
    dc_nps = nps[nps["order_id"].isin(delayed_complained_ids)]
    n_dc_detractor = int((dc_nps["category"] == "Detractor").sum())

    comp_nps = complaints.merge(nps[["order_id", "category"]], on="order_id", how="inner")
    pct_complaint_to_detractor = (comp_nps["category"] == "Detractor").mean() * 100 if len(comp_nps) else 0

   
    n_non_delivered_complaints = int((complaints["order_status"] != "Delivered").sum())
    pct_complaints_non_delivered = n_non_delivered_complaints / len(complaints) * 100

   
    orders_seq = orders.sort_values(["customer_id", "order_date"]).copy()
    orders_seq["has_later_order"] = orders_seq["order_date"] < orders_seq.groupby("customer_id")["order_date"].transform("max")
    reorder_after_complaint = orders_seq.loc[orders_seq["has_complaint"], "has_later_order"].mean() * 100
    reorder_after_clean = orders_seq.loc[~orders_seq["has_complaint"], "has_later_order"].mean() * 100

    st.info(
        f"Among **delivered** orders, **{delayed_to_complaint:.0f}% of delayed ones generate a "
        f"complaint**, versus {ontime_to_complaint:.0f}% of on-time ones. Of orders that were both "
        f"delayed and complained about, **{n_dc_detractor/n_delayed_complained*100 if n_delayed_complained else 0:.0f}% "
        f"came back as a Detractor score.** Delay is the trigger for most of the negative-experience "
        f"chain — but **{pct_complaints_non_delivered:.0f}% of all complaints are tied to RTO/"
        "Cancelled orders**, not delayed deliveries, so a delay-only funnel understates total "
        "complaint volume by design (shown separately, not folded into the funnel below)."
    )

    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure(go.Funnel(
            y=["Delivered orders", "Delayed (SLA breach)", "Delayed + complained",
               "Delayed + complained + Detractor"],
            x=[len(delivered_c), n_delayed, n_delayed_complained, n_dc_detractor],
            marker=dict(color=[NEUTRAL_BLUE, "#D9A441", BAD, "#8B0000"]),
        ))
        fig.update_layout(title="Negative-experience funnel (strict subset at each stage)", height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Separately: {n_non_delivered_complaints} of {len(complaints)} complaints "
                   f"({pct_complaints_non_delivered:.0f}%) come from RTO/Cancelled orders — a second, "
                   "parallel failure path outside this funnel.")
    with c2:
        st.markdown("**Repeat usage — did the customer order again?**")
        rep_df = pd.DataFrame({
            "Group": ["After a complaint-linked order", "After a clean order"],
            "Reorder rate %": [reorder_after_complaint, reorder_after_clean],
        })
        fig = px.bar(rep_df, x="Group", y="Reorder rate %", color="Group",
                     color_discrete_sequence=[BAD, GOOD])
        fig.update_layout(height=340, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{reorder_after_complaint:.0f}% of customers place another order after a "
                   f"complaint-linked one, vs {reorder_after_clean:.0f}% after a clean order — a real "
                   "but modest gap over a 3-month window. Repeat volume is a lagging signal here; "
                   "NPS is the leading one, and the gap would likely widen with a longer observation "
                   "window past the festive surge.")

# ==========================================================================
# TAB E — RECOMMENDATIONS
# ==========================================================================

with tabs[5]:

    st.subheader("E · Business Recommendations")
    st.caption("Primary goal: improve NPS and reduce complaints without significantly increasing cost.")

    st.markdown("#### Top 3 root causes")

    st.markdown(
        f"1. **{worst_courier} has the weakest performance** — "
        f"{courier.set_index('courier_partner').loc[worst_courier, 'sla_breach_rate']*100:.0f}% SLA "
        "breach rate and the highest complaint rate of any courier. This makes its performance "
        "a clear area for improvement.\n"
        f"2. **Hub-level operational gaps in {worst_city} and its Tier-2 peer** — SLA breach, failed "
        "attempts, and RTO are all higher in the weaker hubs. The similar courier mix across cities "
        "suggests that hub-level factors should be investigated.\n"
        f"3. **Delay is strongly linked to the CX chain** — {delayed_to_complaint:.0f}% of late "
        "orders become complaints, and complaint-linked NPS responses are heavily concentrated "
        "among Detractors. Reducing delays should therefore be a key priority."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🚀 Quick wins (0–4 weeks, low cost)")
        st.markdown(
            f"- Cap or reduce **{worst_courier}** volume allocation in the weaker cities; shift "
            "overflow to the better-performing couriers.\n"
            "- Add a **proactive delay notification** (SMS/WhatsApp) for any order projected to "
            "breach SLA — it doesn't fix the delay, but can help manage the customer experience.\n"
            "- Introduce a **second delivery-attempt window** in the two worst hubs to reduce failed "
            "attempts before they become RTOs.\n"
            "- Fast-track complaint resolution for the lowest-NPS customer segment specifically."
        )

    with c2:
        st.markdown("##### 🏗️ Long-term structural fixes")
        st.markdown(
            "- **Hub capacity review** for the weaker cities ahead of next festive season — "
            "staffing, last-mile fleet size, and sorting throughput.\n"
            f"- **Review {worst_courier} allocation and SLA performance** before the next peak period, "
            "and consider SLA-linked changes if the performance gap continues.\n"
            "- Use historical delay patterns to **flag orders that are likely to miss the promised "
            "date** (courier, hub load, city, season).\n"
            "- Follow up with customers who **complained and gave a low NPS score** to help protect "
            "repeat usage once the surge ends."
        )

    st.markdown("##### 📈 Suggested KPIs to track going forward")

    kpi_df = pd.DataFrame({
        "KPI": ["NPS (monthly)", "SLA breach % (delivered orders)", "Complaint rate %",
                "Complaint → Detractor %", "Failed attempt % (by hub)", "RTO %",
                "Courier SLA breach % (by partner)"],
        "Current (Oct–Dec)": [
            f"{overall_nps:.0f}", f"{overall_sla_breach:.0f}%", f"{complaint_rate:.1f}%",
            f"{pct_complaint_to_detractor:.0f}%", f"{hub['failed_attempt_pct'].mean():.0f}% avg",
            f"{rto_rate:.1f}%",
            f"{worst_courier} {courier.set_index('courier_partner').loc[worst_courier,'sla_breach_rate']*100:.0f}%",
        ],
        "Target": ["> 0 (positive)", "< 15%", "< 10%", "< 40%", "< 10%", "< 5%",
                   "< 20% for every partner"],
    })

    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    st.success(
        "**Reminder:** every recommendation above targets the delay → complaint → detractor chain "
        "directly, and most are process changes — courier re-allocation, attempt windows, proactive "
        "comms — not cost increases."
    )

st.caption("Built with Streamlit · Case Study")
