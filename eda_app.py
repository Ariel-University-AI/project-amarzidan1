import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA – תאונות דרכים", layout="wide", page_icon="🚗")
st.title("🔍 ניתוח חקרני (EDA) – תאונות דרכים בישראל")

@st.cache_data
def load_data():
    df = pd.read_csv("israel_road_accidents_simulated.csv")
    return df

df = load_data()

# ── 3 מדדים ─────────────────────────────────────────────────────────────────
missing_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("שורות", f"{df.shape[0]:,}")
col2.metric("עמודות", df.shape[1])
col3.metric("אחוז ערכים חסרים", f"{missing_pct}%")

st.markdown("---")

# ── סיכום סטטיסטי ────────────────────────────────────────────────────────────
st.subheader("📋 סיכום סטטיסטי")

tab_num, tab_cat = st.tabs(["עמודות מספריות", "עמודות קטגוריאליות"])

with tab_num:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        st.info("אין עמודות מספריות.")
    else:
        stats = numeric_df.agg(["mean", "median", "std", "min", "max"]).T
        stats.columns = ["ממוצע", "חציון", "סטיית תקן", "מינימום", "מקסימום"]
        stats = stats.round(2)
        st.dataframe(stats, use_container_width=True)

with tab_cat:
    cat_df = df.select_dtypes(include="object")
    if cat_df.empty:
        st.info("אין עמודות קטגוריאליות.")
    else:
        cat_stats = pd.DataFrame({
            "ערכים ייחודיים": cat_df.nunique(),
            "ערך נפוץ ביותר": cat_df.mode().iloc[0],
            "תדירות נפוצה": [df[c].value_counts().iloc[0] for c in cat_df.columns],
            "ערכים חסרים": cat_df.isnull().sum(),
        })
        st.dataframe(cat_stats, use_container_width=True)

st.markdown("---")

# ── Histogram ────────────────────────────────────────────────────────────────
st.subheader("📊 היסטוגרמה לפי עמודה")

hist_col = st.selectbox("בחר עמודה להיסטוגרמה:", df.columns.tolist(), key="hist")

if pd.api.types.is_numeric_dtype(df[hist_col]):
    fig_hist = px.histogram(
        df,
        x=hist_col,
        nbins=30,
        title=f"התפלגות {hist_col}",
        color_discrete_sequence=["#636EFA"],
    )
else:
    counts = df[hist_col].value_counts().reset_index()
    counts.columns = [hist_col, "count"]
    fig_hist = px.bar(
        counts,
        x=hist_col,
        y="count",
        title=f"התפלגות {hist_col}",
        color_discrete_sequence=["#636EFA"],
    )

fig_hist.update_layout(bargap=0.1, xaxis_title=hist_col, yaxis_title="תדירות")
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ── Scatter Plot ─────────────────────────────────────────────────────────────
st.subheader("🔵 Scatter Plot בין שתי עמודות")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 2:
    st.warning("אין מספיק עמודות מספריות ל-Scatter Plot.")
else:
    sc1, sc2, sc3 = st.columns(3)
    x_col = sc1.selectbox("ציר X:", numeric_cols, index=0, key="sc_x")
    y_col = sc2.selectbox("ציר Y:", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sc_y")

    color_options = ["ללא"] + df.select_dtypes(include="object").columns.tolist()
    color_col = sc3.selectbox("צבע לפי (אופציונלי):", color_options, key="sc_color")

    fig_scatter = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=None if color_col == "ללא" else color_col,
        title=f"{y_col} מול {x_col}",
        opacity=0.7,
    )
    fig_scatter.update_traces(marker=dict(size=6))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ── דשבורד ────────────────────────────────────────────────────────────────────
st.header("📈 דשבורד – מגמות לאורך זמן")

# זיהוי עמודות זמן אפשריות
date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "year", "month", "שנה", "חודש", "תאריך"])]

if not date_candidates:
    st.warning("לא נמצאה עמודת זמן. בחר עמודה ידנית:")
    date_candidates = df.columns.tolist()

d1, d2 = st.columns([1, 3])

with d1:
    time_col = st.selectbox("עמודת זמן:", date_candidates, key="dash_time")
    count_by = st.selectbox(
        "קבץ לפי (אופציונלי):",
        ["ללא"] + df.select_dtypes(include="object").columns.tolist(),
        key="dash_group",
    )

with d2:
    if count_by == "ללא":
        ts = df.groupby(time_col).size().reset_index(name="תאונות")
        fig_ts = px.line(
            ts,
            x=time_col,
            y="תאונות",
            title=f"מספר תאונות לאורך {time_col}",
            markers=True,
            color_discrete_sequence=["#EF553B"],
        )
    else:
        ts = df.groupby([time_col, count_by]).size().reset_index(name="תאונות")
        fig_ts = px.line(
            ts,
            x=time_col,
            y="תאונות",
            color=count_by,
            title=f"מספר תאונות לאורך {time_col} לפי {count_by}",
            markers=True,
        )

    fig_ts.update_layout(xaxis_title=time_col, yaxis_title="מספר תאונות")
    st.plotly_chart(fig_ts, use_container_width=True)

# KPI cards בתחתית הדשבורד
st.markdown("### מדדי מפתח")
k1, k2, k3, k4 = st.columns(4)
k1.metric("סה\"כ תאונות", f"{len(df):,}")
k2.metric("מספר שנים", df[time_col].nunique() if time_col in df.columns else "—")

if "severity" in df.columns.str.lower().tolist():
    sev_col = [c for c in df.columns if c.lower() == "severity"][0]
    fatal = df[sev_col].value_counts().get("Fatal", df[sev_col].value_counts().get("קטלנית", 0))
    k3.metric("תאונות קטלניות", f"{fatal:,}")
else:
    k3.metric("עמודות", df.shape[1])

k4.metric("ממוצע תאונות לתקופה", f"{len(df) / max(df[time_col].nunique(), 1):.0f}" if time_col in df.columns else "—")
