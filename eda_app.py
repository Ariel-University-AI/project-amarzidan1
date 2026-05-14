import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA – תאונות דרכים בישראל", layout="wide", page_icon="🚗")

@st.cache_data
def load_data():
    df = pd.read_csv("israel_road_accidents_simulated.csv")
    df.columns = df.columns.str.strip()
    df["תאריך"] = pd.to_datetime(df["תאריך"], dayfirst=True, errors="coerce")
    df["חודש"] = df["תאריך"].dt.month
    df["שנה"] = df["תאריך"].dt.year
    df["יום בשבוע"] = df["תאריך"].dt.day_name()
    df["שעה_מספר"] = df["שעה"].str.split(":").str[0].astype(int, errors="ignore")
    return df

df = load_data()

DAYS_HE = {"Monday":"שני","Tuesday":"שלישי","Wednesday":"רביעי",
           "Thursday":"חמישי","Friday":"שישי","Saturday":"שבת","Sunday":"ראשון"}
df["יום בשבוע עב"] = df["יום בשבוע"].map(DAYS_HE)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 סינון נתונים")
    cities = ["הכל"] + sorted(df["עיר או כביש"].dropna().unique().tolist())
    city_sel = st.selectbox("עיר / כביש:", cities)
    severity_sel = st.multiselect("חומרת תאונה:", df["חומרת התאונה"].dropna().unique().tolist(),
                                   default=df["חומרת התאונה"].dropna().unique().tolist())
    weather_sel = st.multiselect("מזג אוויר:", df["מזג_אוויר"].dropna().unique().tolist(),
                                  default=df["מזג_אוויר"].dropna().unique().tolist())

filtered = df.copy()
if city_sel != "הכל":
    filtered = filtered[filtered["עיר או כביש"] == city_sel]
if severity_sel:
    filtered = filtered[filtered["חומרת התאונה"].isin(severity_sel)]
if weather_sel:
    filtered = filtered[filtered["מזג_אוויר"].isin(weather_sel)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 ניתוח חקרני (EDA) – תאונות דרכים בישראל")
st.caption(f"מציג {len(filtered):,} תאונות מתוך {len(df):,}")
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("סה\"כ תאונות", f"{len(filtered):,}")
k2.metric("סה\"כ נפגעים", f"{filtered['מספר_נפגעים'].sum():,}")
k3.metric("תאונות קטלניות", f"{(filtered['חומרת התאונה']=='קטלנית').sum():,}")
k4.metric("תאונות קשות", f"{(filtered['חומרת התאונה']=='קשה').sum():,}")
k5.metric("ממוצע רכבים לתאונה", f"{filtered['מספר הרכבים המשתטפים'].mean():.1f}")

st.markdown("---")

# ── Row 1: חומרה + מזג אוויר ──────────────────────────────────────────────────
st.subheader("📊 התפלגות לפי קטגוריות")
c1, c2, c3 = st.columns(3)

with c1:
    sev = filtered["חומרת התאונה"].value_counts().reset_index()
    sev.columns = ["חומרה", "כמות"]
    colors = {"קלה": "#2ecc71", "קשה": "#e67e22", "קטלנית": "#e74c3c"}
    fig = px.pie(sev, names="חומרה", values="כמות", title="חומרת תאונות",
                 color="חומרה", color_discrete_map=colors, hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    weather = filtered["מזג_אוויר"].value_counts().reset_index()
    weather.columns = ["מזג אוויר", "כמות"]
    fig = px.bar(weather, x="מזג אוויר", y="כמות", title="תאונות לפי מזג אוויר",
                 color="כמות", color_continuous_scale="Blues")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    traffic = filtered["עומס_תנועה"].value_counts().reset_index()
    traffic.columns = ["עומס", "כמות"]
    fig = px.bar(traffic, x="עומס", y="כמות", title="תאונות לפי עומס תנועה",
                 color="כמות", color_continuous_scale="Oranges")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 2: זמן ───────────────────────────────────────────────────────────────
st.subheader("⏰ ניתוח זמני")
t1, t2 = st.columns(2)

with t1:
    hour_data = filtered.groupby("שעה_מספר").size().reset_index(name="כמות")
    fig = px.bar(hour_data, x="שעה_מספר", y="כמות",
                 title="תאונות לפי שעה ביום",
                 labels={"שעה_מספר": "שעה", "כמות": "מספר תאונות"},
                 color="כמות", color_continuous_scale="Reds")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    day_order = ["ראשון","שני","שלישי","רביעי","חמישי","שישי","שבת"]
    day_data = filtered.groupby("יום בשבוע עב").size().reset_index(name="כמות")
    day_data["יום בשבוע עב"] = pd.Categorical(day_data["יום בשבוע עב"], categories=day_order, ordered=True)
    day_data = day_data.sort_values("יום בשבוע עב")
    fig = px.bar(day_data, x="יום בשבוע עב", y="כמות",
                 title="תאונות לפי יום בשבוע",
                 labels={"יום בשבוע עב": "יום", "כמות": "מספר תאונות"},
                 color="כמות", color_continuous_scale="Purples")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 3: חודשי + סוג דרך ───────────────────────────────────────────────────
m1, m2 = st.columns(2)

with m1:
    month_names = {1:"ינואר",2:"פברואר",3:"מרץ",4:"אפריל",5:"מאי",6:"יוני",
                   7:"יולי",8:"אוגוסט",9:"ספטמבר",10:"אוקטובר",11:"נובמבר",12:"דצמבר"}
    month_data = filtered.groupby("חודש").size().reset_index(name="כמות")
    month_data["שם חודש"] = month_data["חודש"].map(month_names)
    fig = px.line(month_data, x="חודש", y="כמות",
                  title="תאונות לפי חודש",
                  labels={"חודש": "חודש", "כמות": "מספר תאונות"},
                  markers=True, color_discrete_sequence=["#3498db"])
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=list(month_names.values()))
    st.plotly_chart(fig, use_container_width=True)

with m2:
    road_sev = filtered.groupby(["סוג_דרך", "חומרת התאונה"]).size().reset_index(name="כמות")
    fig = px.bar(road_sev, x="סוג_דרך", y="כמות", color="חומרת התאונה",
                 title="סוג דרך לפי חומרה",
                 color_discrete_map={"קלה":"#2ecc71","קשה":"#e67e22","קטלנית":"#e74c3c"},
                 barmode="stack")
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 4: ערים + נפגעים ─────────────────────────────────────────────────────
st.subheader("🏙️ ניתוח לפי מיקום")
ci1, ci2 = st.columns(2)

with ci1:
    top_cities = filtered["עיר או כביש"].value_counts().head(10).reset_index()
    top_cities.columns = ["עיר", "כמות"]
    fig = px.bar(top_cities, x="כמות", y="עיר", orientation="h",
                 title="10 מיקומים עם הכי הרבה תאונות",
                 color="כמות", color_continuous_scale="Reds")
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with ci2:
    victims = filtered.groupby("עיר או כביש")["מספר_נפגעים"].sum().sort_values(ascending=False).head(10).reset_index()
    victims.columns = ["עיר", "נפגעים"]
    fig = px.bar(victims, x="נפגעים", y="עיר", orientation="h",
                 title="10 מיקומים עם הכי הרבה נפגעים",
                 color="נפגעים", color_continuous_scale="Oranges")
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── מפה ──────────────────────────────────────────────────────────────────────
st.subheader("🗺️ מפת תאונות")
map_data = filtered.dropna(subset=["קו_רוחב", "קו_אורך"])
color_map = {"קלה": "green", "קשה": "orange", "קטלנית": "red"}
fig_map = px.scatter_mapbox(
    map_data, lat="קו_רוחב", lon="קו_אורך",
    color="חומרת התאונה",
    color_discrete_map=color_map,
    hover_name="עיר או כביש",
    hover_data={"מספר_נפגעים": True, "מזג_אוויר": True, "עומס_תנועה": True,
                "קו_רוחב": False, "קו_אורך": False},
    size="מספר_נפגעים", size_max=15,
    zoom=7, center={"lat": 31.8, "lon": 35.0},
    mapbox_style="carto-positron",
    title="פיזור תאונות על מפת ישראל",
    height=500,
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ── סיכום סטטיסטי ────────────────────────────────────────────────────────────
st.subheader("📋 סיכום סטטיסטי")
tab_num, tab_cat = st.tabs(["עמודות מספריות", "עמודות קטגוריאליות"])

with tab_num:
    num_cols = ["מספר_נפגעים", "מספר הרכבים המשתטפים", "שעה_מספר", "חודש"]
    stats = filtered[num_cols].agg(["mean","median","std","min","max"]).T.round(2)
    stats.columns = ["ממוצע","חציון","סטיית תקן","מינימום","מקסימום"]
    st.dataframe(stats, use_container_width=True)

with tab_cat:
    cat_cols = ["חומרת התאונה","מזג_אוויר","עומס_תנועה","סוג_דרך","חלק מהיום"]
    rows = []
    for c in cat_cols:
        vc = filtered[c].value_counts()
        rows.append({"עמודה": c,
                     "ערכים ייחודיים": filtered[c].nunique(),
                     "ערך נפוץ": vc.index[0] if len(vc) else "—",
                     "תדירות": int(vc.iloc[0]) if len(vc) else 0,
                     "ערכים חסרים": int(filtered[c].isnull().sum())})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")

# ── המלצת כביש בטוח ──────────────────────────────────────────────────────────
st.header("🛡️ המלצת כביש בטוח")
st.write("בחר עיר או כביש מוצא ויעד — המערכת תדרג את כל האפשרויות לפי ציון סיכון.")

SEVERITY_WEIGHT = {"קלה": 1, "קשה": 5, "קטלנית": 20}

@st.cache_data
def calc_risk(data):
    data = data.copy()
    data["ציון_חומרה"] = data["חומרת התאונה"].map(SEVERITY_WEIGHT).fillna(1)
    risk = data.groupby("עיר או כביש").agg(
        תאונות=("מזהה תאונה", "count"),
        נפגעים=("מספר_נפגעים", "sum"),
        ציון_סיכון_גולמי=("ציון_חומרה", "sum"),
        lat=("קו_רוחב", "mean"),
        lon=("קו_אורך", "mean"),
    ).reset_index()
    max_score = risk["ציון_סיכון_גולמי"].max()
    risk["ציון_סיכון"] = (risk["ציון_סיכון_גולמי"] / max_score * 100).round(1)
    risk["דירוג_בטיחות"] = risk["ציון_סיכון"].apply(
        lambda x: "🟢 בטוח" if x < 20 else ("🟡 בינוני" if x < 50 else "🔴 מסוכן")
    )
    return risk.sort_values("ציון_סיכון")

risk_df = calc_risk(df)

all_locations = sorted(df["עיר או כביש"].dropna().unique().tolist())

rec1, rec2 = st.columns(2)
with rec1:
    origin = st.selectbox("📍 מוצא:", all_locations, key="origin")
with rec2:
    dest_options = [l for l in all_locations if l != origin]
    destination = st.selectbox("🏁 יעד:", dest_options, key="dest")

# הצג ציון כביש מוצא ויעד
orig_row = risk_df[risk_df["עיר או כביש"] == origin]
dest_row = risk_df[risk_df["עיר או כביש"] == destination]

r1, r2, r3 = st.columns(3)
if not orig_row.empty:
    r1.metric(f"ציון סיכון — {origin}",
              f"{orig_row['ציון_סיכון'].values[0]}/100",
              orig_row['דירוג_בטיחות'].values[0])
if not dest_row.empty:
    r2.metric(f"ציון סיכון — {destination}",
              f"{dest_row['ציון_סיכון'].values[0]}/100",
              dest_row['דירוג_בטיחות'].values[0])

avg_score = 0
if not orig_row.empty and not dest_row.empty:
    avg_score = round((orig_row['ציון_סיכון'].values[0] + dest_row['ציון_סיכון'].values[0]) / 2, 1)
    if avg_score < 20:
        verdict = "✅ המסלול בטוח יחסית"
        color = "success"
    elif avg_score < 50:
        verdict = "⚠️ המסלול בסיכון בינוני — נסע בזהירות"
        color = "warning"
    else:
        verdict = "🚨 המסלול מסוכן — שקול מסלול חלופי"
        color = "error"
    r3.metric("ציון סיכון ממוצע למסלול", f"{avg_score}/100")
    getattr(st, color)(verdict)

st.markdown("#### 📊 דירוג כל הכבישים לפי בטיחות")

show_cols = ["עיר או כביש", "תאונות", "נפגעים", "ציון_סיכון", "דירוג_בטיחות"]
risk_display = risk_df[show_cols].copy()
risk_display.columns = ["כביש / עיר", "מס' תאונות", "מס' נפגעים", "ציון סיכון (0-100)", "דירוג"]

st.dataframe(
    risk_display.style.background_gradient(subset=["ציון סיכון (0-100)"], cmap="RdYlGn_r"),
    use_container_width=True,
    height=400,
)

# מפת סיכון
st.markdown("#### 🗺️ מפת ציוני סיכון")
map_risk = risk_df.dropna(subset=["lat", "lon"])
fig_risk = px.scatter_mapbox(
    map_risk, lat="lat", lon="lon",
    color="ציון_סיכון",
    size="תאונות",
    size_max=20,
    hover_name="עיר או כביש",
    hover_data={"תאונות": True, "נפגעים": True, "ציון_סיכון": True, "lat": False, "lon": False},
    color_continuous_scale="RdYlGn_r",
    zoom=7, center={"lat": 31.8, "lon": 35.0},
    mapbox_style="carto-positron",
    title="מפת סיכון — ירוק=בטוח, אדום=מסוכן",
    height=500,
)
fig_risk.update_coloraxes(colorbar_title="ציון סיכון")
st.plotly_chart(fig_risk, use_container_width=True)
