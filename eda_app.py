import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="מוקדי סיכון — תשתיות תחבורה בישראל",
    layout="wide",
    page_icon="🚗",
)

# ── Decode tables (CBS PUF documentation) ─────────────────────────────────────
_SEV    = {1: "קטלנית", 2: "קשה", 3: "קלה"}
_ROAD   = {1: "עירוני - בצומת", 2: "עירוני - לא בצומת",
           3: "בין-עירוני - בצומת", 4: "בין-עירוני - לא בצומת",
           5: "חניון / כיכר", 9: "אחר"}
_WTHR   = {1: "בהיר", 2: "גשם קל", 3: "גשם", 4: "ערפל",
           5: "חול", 7: "שלג", 8: "סופה", 9: "אחר"}
_SURF   = {1: "יבש", 2: "רטוב", 3: "קפוא", 4: "שלג", 9: "אחר"}
_DNTM   = {1: "יום", 5: "לילה"}
_DWEEK  = {1: "ראשון", 2: "שני", 3: "שלישי", 4: "רביעי",
           5: "חמישי", 6: "שישי", 7: "שבת"}
_DIST   = {1: "ירושלים", 2: "צפון", 3: "חיפה",
           4: "מרכז", 5: "תל אביב", 6: "דרום", 7: 'יו"ש'}
_SPD    = {1: "30", 2: "40", 3: "50", 4: "60", 5: "70",
           6: "80", 7: "90", 8: "100", 9: "110"}
_ACCTYP = {1: "חזיתית", 2: "אחורית", 3: "צידית", 4: "הולך רגל",
           5: "התהפכות", 6: "פגיעה בעמוד", 7: "נפילה מרכב", 8: "אחר"}
_MONTHS = {1: "ינואר", 2: "פברואר", 3: "מרץ", 4: "אפריל", 5: "מאי",
           6: "יוני", 7: "יולי", 8: "אוגוסט", 9: "ספטמבר",
           10: "אוקטובר", 11: "נובמבר", 12: "דצמבר"}
_DAY_ORDER = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]

# ── Coordinate conversion: ITM (EPSG:2039) → WGS84 ───────────────────────────
def _itm_to_wgs84(x_s: pd.Series, y_s: pd.Series):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x_s.values, y_s.values)
        return pd.Series(lat.round(6), index=x_s.index), pd.Series(lon.round(6), index=x_s.index)
    except Exception:
        lat = ((y_s - 626907) / 111320 + 31.5).round(5)
        lon = ((x_s - 219529) / (111320 * 0.857) + 35.21).round(5)
        return lat, lon

# ── City names lookup ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _city_map() -> dict:
    try:
        r = requests.get(
            "https://data.gov.il/api/3/action/datastore_search"
            "?resource_id=5c78e9fa-c2e2-4771-93ff-7f400a12f7ba&limit=2000",
            timeout=10,
        )
        rows = r.json()["result"]["records"]
        return {int(c["סמל_ישוב"]): c["שם_ישוב"] for c in rows if c["סמל_ישוב"]}
    except Exception:
        return {}

# ── Data loading & decoding ───────────────────────────────────────────────────
@st.cache_data(show_spinner="טוען נתוני תאונות (למ\"ס PUF 2021)…")
def load_data() -> pd.DataFrame:
    city_map = _city_map()

    df = pd.read_csv("data/accidents_israel_2021_raw.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    num_cols = [
        "HUMRAT_TEUNA", "SUG_DEREH", "SUG_TEUNA", "MEZEG_AVIR", "PNE_KVISH",
        "YOM_LAYLA", "YOM_BASHAVUA", "MAHOZ", "MEHIRUT_MUTERET",
        "SEMEL_YISHUV", "KVISH1", "HODESH_TEUNA", "SHAA", "X", "Y",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["חומרת_תאונה"]  = df["HUMRAT_TEUNA"].map(_SEV)
    df["סוג_דרך"]      = df["SUG_DEREH"].map(_ROAD)
    df["סוג_תאונה"]    = df["SUG_TEUNA"].map(_ACCTYP)
    df["מזג_אוויר"]    = df["MEZEG_AVIR"].map(_WTHR)
    df["מצב_כביש"]     = df["PNE_KVISH"].map(_SURF)
    df["חלק_יממה"]     = df["YOM_LAYLA"].map(_DNTM)
    df["יום_בשבוע"]    = df["YOM_BASHAVUA"].map(_DWEEK)
    df["מחוז"]         = df["MAHOZ"].map(_DIST)
    df["מהירות_מותרת"] = df["MEHIRUT_MUTERET"].map(_SPD)
    df["חודש"]         = df["HODESH_TEUNA"].astype("Int64")
    df["שעה"]          = (df["SHAA"].fillna(0) // 4).clip(0, 23).astype("Int64")
    df["שם_חודש"]      = df["חודש"].map(_MONTHS)
    df["יום_בשבוע"]    = pd.Categorical(df["יום_בשבוע"], categories=_DAY_ORDER, ordered=True)

    # Location
    df["שם_ישוב"] = df["SEMEL_YISHUV"].map(city_map)
    df["כביש"]    = df["KVISH1"].where(df["KVISH1"].notna() & (df["KVISH1"] > 0))
    df["מיקום"]   = df.apply(
        lambda r: r["שם_ישוב"]
        if pd.notna(r["שם_ישוב"]) and r["שם_ישוב"]
        else (f"כביש {int(r['כביש'])}" if pd.notna(r["כביש"]) else "לא ידוע"),
        axis=1,
    )
    df["אתר"] = df["מיקום"] + " – " + df["סוג_דרך"].fillna("לא ידוע")

    # Coordinates (ITM → WGS84)
    mask = df["X"].notna() & df["Y"].notna()
    df["קו_רוחב"] = np.nan
    df["קו_אורך"] = np.nan
    lat, lon = _itm_to_wgs84(df.loc[mask, "X"], df.loc[mask, "Y"])
    df.loc[mask, "קו_רוחב"] = lat.values
    df.loc[mask, "קו_אורך"] = lon.values

    return df

# ── Intervention recommendation engine ───────────────────────────────────────
INTERVENTIONS = {
    "כיכר תנועה": {
        "desc":    "מחליפה צומת רגיל — מבטלת התנגשויות חזיתיות ומאלצת האטה",
        "fatal":   0.82, "serious": 0.55, "cost": "גבוה",
        "tags":    {"חזיתית", "התהפכות", "מהירות_גבוהה"},
    },
    "רמזור חכם": {
        "desc":    "מסדיר תנועה עם עדיפות להולכי רגל ובשעות עומס",
        "fatal":   0.45, "serious": 0.32, "cost": "בינוני",
        "tags":    {"הולך רגל", "עירוני", "עומס"},
    },
    "מעבר חצייה מוגן + תאורה": {
        "desc":    "הגבהה, תמרור, תאורה ממוקדת — יעיל לאזורים עם הולכי רגל",
        "fatal":   0.55, "serious": 0.38, "cost": "נמוך",
        "tags":    {"הולך רגל", "עירוני"},
    },
    "תאורת לד מוגברת": {
        "desc":    "שיפור תאורה בצמתים ולאורך הכביש — יעיל במיוחד בתאונות לילה",
        "fatal":   0.32, "serious": 0.22, "cost": "נמוך",
        "tags":    {"לילה", "ערפל"},
    },
    "מצלמת אכיפה + מד-מהירות": {
        "desc":    "אכיפה אוטומטית — מורידה מהירות ממוצעת ב-7 קמ\"ש",
        "fatal":   0.25, "serious": 0.18, "cost": "נמוך-בינוני",
        "tags":    {"מהירות_גבוהה", "בין-עירוני"},
    },
    "פסי האטה / מוקפצים": {
        "desc":    "מאלצים האטה פיזית — יעיל בכניסות לישובים ואזורי מגורים",
        "fatal":   0.30, "serious": 0.25, "cost": "נמוך",
        "tags":    {"אחורית", "מהירות_בינונית"},
    },
}

def _accident_profile(site_df: pd.DataFrame) -> dict:
    n = max(len(site_df), 1)
    spd_mode = site_df["מהירות_מותרת"].mode()
    spd_val  = spd_mode.iloc[0] if len(spd_mode) else "50"
    hr_mode  = site_df["שעה"].mode()
    return {
        "total":          len(site_df),
        "fatal_pct":      (site_df["חומרת_תאונה"] == "קטלנית").sum() / n,
        "serious_pct":    (site_df["חומרת_תאונה"] == "קשה").sum() / n,
        "night_pct":      (site_df["חלק_יממה"] == "לילה").sum() / n,
        "rain_pct":       site_df["מזג_אוויר"].isin(["גשם", "גשם קל"]).sum() / n,
        "pedestrian_pct": (site_df["סוג_תאונה"] == "הולך רגל").sum() / n,
        "frontal_pct":    (site_df["סוג_תאונה"] == "חזיתית").sum() / n,
        "rear_pct":       (site_df["סוג_תאונה"] == "אחורית").sum() / n,
        "rollover_pct":   (site_df["סוג_תאונה"] == "התהפכות").sum() / n,
        "high_speed":     str(spd_val) in {"70", "80", "90", "100", "110"},
        "peak_hour":      int(hr_mode.iloc[0]) if len(hr_mode) else 0,
        "road_type":      site_df["סוג_דרך"].mode().iloc[0] if len(site_df) else "—",
    }

def _rank_interventions(prof: dict, approaches: int, traffic: int, current: str) -> list:
    scores = {}
    for name, info in INTERVENTIONS.items():
        s = (
            prof["fatal_pct"]   * info["fatal"]   +
            prof["serious_pct"] * info["serious"]
        ) * prof["total"] * 10

        if prof["pedestrian_pct"] > 0.12 and "הולך רגל" in info["tags"]: s += 25
        if prof["frontal_pct"]    > 0.18 and "חזיתית"   in info["tags"]: s += 25
        if prof["night_pct"]      > 0.35 and "לילה"      in info["tags"]: s += 20
        if prof["high_speed"]               and "מהירות_גבוהה" in info["tags"]: s += 20
        if prof["rear_pct"]       > 0.20 and "אחורית"    in info["tags"]: s += 15
        if name == "כיכר תנועה"    and approaches in {3, 4}:  s += 10
        if name == "רמזור חכם"     and traffic > 10000:        s += 10
        if name == "רמזור חכם"     and current == "רמזור קיים": s -= 35
        if name == "כיכר תנועה"   and current == "כיכר קיימת": s -= 35
        scores[name] = max(0.0, s)

    mx = max(scores.values()) or 1
    return sorted(
        [(n, round(v / mx * 100), INTERVENTIONS[n]) for n, v in scores.items()],
        key=lambda x: x[1], reverse=True,
    )

# ── ML model ──────────────────────────────────────────────────────────────────
_FEAT_COLS = ["סוג_דרך", "מזג_אוויר", "מצב_כביש", "מהירות_מותרת",
              "חלק_יממה", "מחוז", "סוג_תאונה"]

@st.cache_resource(show_spinner="מאמן מודל Random Forest…")
def _train_model():
    df = load_data()
    sub = df.dropna(subset=_FEAT_COLS + ["חומרת_תאונה"]).copy()

    encs = {c: LabelEncoder() for c in _FEAT_COLS}
    X = np.column_stack([encs[c].fit_transform(sub[c].astype(str)) for c in _FEAT_COLS])
    y = sub["חומרת_תאונה"].map({"קלה": 0, "קשה": 1, "קטלנית": 2}).values

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    importance = pd.Series(clf.feature_importances_, index=_FEAT_COLS).sort_values(ascending=False)
    return clf, encs, importance

@st.cache_data(show_spinner="מחשב ציוני סיכון לאתרי תשתית…")
def _score_sites(_v: str) -> pd.DataFrame:
    df  = load_data()
    clf, encs, _ = _train_model()

    sub = df.dropna(subset=_FEAT_COLS).copy()
    X_parts = []
    for c in _FEAT_COLS:
        col_str = sub[c].fillna("לא ידוע").astype(str)
        col_str[~col_str.isin(encs[c].classes_)] = encs[c].classes_[0]
        X_parts.append(encs[c].transform(col_str))
    X = np.column_stack(X_parts)

    proba = clf.predict_proba(X)
    weights = np.array([0, 5, 20])
    sub = sub.copy()
    sub["ציון_גלמי"] = (proba * weights).sum(axis=1)

    agg = (
        sub.groupby("אתר")
        .agg(
            תאונות        = ("אתר",         "count"),
            ציון_סיכון_גלמי = ("ציון_גלמי", "mean"),
            lat           = ("קו_רוחב",     "mean"),
            lon           = ("קו_אורך",     "mean"),
            מחוז          = ("מחוז",        lambda x: x.mode().iloc[0] if len(x) else "—"),
            סוג_דרך       = ("סוג_דרך",     lambda x: x.mode().iloc[0] if len(x) else "—"),
        )
        .reset_index()
    )

    mx = agg["ציון_סיכון_גלמי"].max()
    agg["ציון_סיכון"] = (agg["ציון_סיכון_גלמי"] / mx * 100).round(1)
    agg["דירוג"] = agg["ציון_סיכון"].apply(
        lambda s: "🔴 גבוה" if s >= 60 else ("🟡 בינוני" if s >= 30 else "🟢 נמוך")
    )
    agg = agg.sort_values("ציון_סיכון", ascending=False).reset_index(drop=True)
    agg.index += 1
    return agg

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 סינון נתונים")
    dist_opts = ["הכל"] + sorted(df["מחוז"].dropna().unique().tolist())
    dist_sel  = st.selectbox("מחוז:", dist_opts)
    sev_opts  = sorted(df["חומרת_תאונה"].dropna().unique().tolist())
    sev_sel   = st.multiselect("חומרת תאונה:", sev_opts, default=sev_opts)
    wthr_opts = sorted(df["מזג_אוויר"].dropna().unique().tolist())
    wthr_sel  = st.multiselect("מזג אוויר:", wthr_opts, default=wthr_opts)

flt = df.copy()
if dist_sel != "הכל":
    flt = flt[flt["מחוז"] == dist_sel]
if sev_sel:
    flt = flt[flt["חומרת_תאונה"].isin(sev_sel)]
if wthr_sel:
    flt = flt[flt["מזג_אוויר"].isin(wthr_sel)]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_eda, tab_mgr, tab_pred = st.tabs([
    "📊 ניתוח חקרני (EDA)",
    "🎯 ממשק מנהל — מוקדי סיכון",
    "🔮 חיזוי פתרון לצומת",
])

# ══════════════════════════════ EDA TAB ═══════════════════════════════════════
with tab_eda:
    st.title("🔍 ניתוח חקרני — תאונות דרכים ישראל 2021")
    st.caption(f"מציג {len(flt):,} תאונות מתוך {len(df):,} | מקור: למ\"ס PUF 2021")
    st.markdown("---")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("סה\"כ תאונות",       f"{len(flt):,}")
    k2.metric("קטלניות",            f"{(flt['חומרת_תאונה'] == 'קטלנית').sum():,}")
    k3.metric("קשות",               f"{(flt['חומרת_תאונה'] == 'קשה').sum():,}")
    k4.metric("קלות",               f"{(flt['חומרת_תאונה'] == 'קלה').sum():,}")
    pct = (flt["חומרת_תאונה"].isin(["קטלנית", "קשה"])).sum() / max(len(flt), 1) * 100
    k5.metric("% קשות + קטלניות",   f"{pct:.1f}%")

    st.markdown("---")

    # ── Row 1: severity + weather + road surface ──
    st.subheader("📊 התפלגות לפי קטגוריות")
    c1, c2, c3 = st.columns(3)

    with c1:
        sev = flt["חומרת_תאונה"].value_counts().reset_index()
        sev.columns = ["חומרה", "כמות"]
        fig = px.pie(sev, names="חומרה", values="כמות", title="חומרת תאונות",
                     color="חומרה",
                     color_discrete_map={"קלה": "#2ecc71", "קשה": "#e67e22", "קטלנית": "#e74c3c"},
                     hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        wthr = flt["מזג_אוויר"].value_counts().reset_index()
        wthr.columns = ["מזג אוויר", "כמות"]
        fig = px.bar(wthr, x="מזג אוויר", y="כמות", title="לפי מזג אוויר",
                     color="כמות", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        surf = flt["מצב_כביש"].value_counts().reset_index()
        surf.columns = ["מצב כביש", "כמות"]
        fig = px.bar(surf, x="מצב כביש", y="כמות", title="לפי מצב כביש",
                     color="כמות", color_continuous_scale="Oranges")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 2: Time ──
    st.subheader("⏰ ניתוח זמני")
    t1, t2 = st.columns(2)

    with t1:
        hour_data = flt.groupby("שעה", observed=True).size().reset_index(name="כמות")
        fig = px.bar(hour_data, x="שעה", y="כמות", title="תאונות לפי שעה ביום",
                     color="כמות", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        day_data = flt.groupby("יום_בשבוע", observed=True).size().reset_index(name="כמות")
        fig = px.bar(day_data, x="יום_בשבוע", y="כמות", title="תאונות לפי יום בשבוע",
                     color="כמות", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Month + road type ──
    m1, m2 = st.columns(2)

    with m1:
        month_data = flt.groupby("חודש", observed=True).size().reset_index(name="כמות")
        month_data["שם"] = month_data["חודש"].map(_MONTHS)
        fig = px.line(month_data, x="חודש", y="כמות", title="תאונות לפי חודש",
                      markers=True, color_discrete_sequence=["#3498db"])
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(_MONTHS.values()))
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        rd = flt.groupby(["סוג_דרך", "חומרת_תאונה"], observed=True).size().reset_index(name="כמות")
        fig = px.bar(rd, x="סוג_דרך", y="כמות", color="חומרת_תאונה",
                     title="סוג דרך לפי חומרה",
                     color_discrete_map={"קלה": "#2ecc71", "קשה": "#e67e22", "קטלנית": "#e74c3c"},
                     barmode="stack")
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 4: District + speed ──
    st.subheader("🏙️ ניתוח לפי מיקום ומהירות")
    d1, d2 = st.columns(2)

    with d1:
        dist_data = flt["מחוז"].value_counts().reset_index()
        dist_data.columns = ["מחוז", "כמות"]
        fig = px.bar(dist_data, x="כמות", y="מחוז", orientation="h",
                     title="תאונות לפי מחוז",
                     color="כמות", color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        spd = flt.groupby(["מהירות_מותרת", "חומרת_תאונה"], observed=True).size().reset_index(name="כמות")
        fig = px.bar(spd, x="מהירות_מותרת", y="כמות", color="חומרת_תאונה",
                     title="חומרה לפי מהירות מותרת (קמ\"ש)",
                     color_discrete_map={"קלה": "#2ecc71", "קשה": "#e67e22", "קטלנית": "#e74c3c"},
                     barmode="stack",
                     category_orders={"מהירות_מותרת": ["30","40","50","60","70","80","90","100","110"]})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Map ──
    st.subheader("🗺️ מפת תאונות")
    map_data = flt.dropna(subset=["קו_רוחב", "קו_אורך"])
    if len(map_data) > 0:
        sample = map_data.sample(min(3000, len(map_data)), random_state=1)
        fig_map = px.scatter_map(
            sample,
            lat="קו_רוחב", lon="קו_אורך",
            color="חומרת_תאונה",
            color_discrete_map={"קלה": "green", "קשה": "orange", "קטלנית": "red"},
            hover_data={"מיקום": True, "מזג_אוויר": True, "סוג_דרך": True,
                        "קו_רוחב": False, "קו_אורך": False},
            zoom=7, center={"lat": 31.8, "lon": 35.0},
            map_style="carto-positron",
            title="פיזור תאונות על מפת ישראל",
            height=520,
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("אין נתוני מיקום לתצוגה בסינון הנוכחי")

    st.markdown("---")

    # ── Statistical summary ──
    st.subheader("📋 סיכום סטטיסטי")
    tab_cat, = st.tabs(["עמודות קטגוריאליות"])
    with tab_cat:
        cat_cols = ["חומרת_תאונה", "סוג_דרך", "מזג_אוויר", "מצב_כביש", "חלק_יממה"]
        rows = []
        for c in cat_cols:
            vc = flt[c].value_counts()
            rows.append({
                "עמודה":           c,
                "ערכים ייחודיים": flt[c].nunique(),
                "ערך נפוץ":        vc.index[0] if len(vc) else "—",
                "תדירות":          int(vc.iloc[0]) if len(vc) else 0,
                "ערכים חסרים":     int(flt[c].isnull().sum()),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════ MANAGER TAB ═══════════════════════════════════
with tab_mgr:
    st.title("🎯 ממשק מנהל — דירוג מוקדי סיכון בתשתיות תחבורה")
    st.caption("מודל Random Forest | נתוני למ\"ס PUF 2021 — 11,554 תאונות אמיתיות")
    st.markdown("---")

    clf, encs, importance = _train_model()
    sites = _score_sites("v1")

    sites_flt = sites[sites["מחוז"] == dist_sel] if dist_sel != "הכל" else sites

    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("סה\"כ אתרי תשתית",   f"{len(sites_flt):,}")
    m2.metric("🔴 סיכון גבוה",       f"{(sites_flt['דירוג'] == '🔴 גבוה').sum():,}")
    m3.metric("🟡 סיכון בינוני",     f"{(sites_flt['דירוג'] == '🟡 בינוני').sum():,}")
    m4.metric("🟢 סיכון נמוך",       f"{(sites_flt['דירוג'] == '🟢 נמוך').sum():,}")

    st.markdown("---")

    # Ranked table
    st.subheader("📋 טבלת עדיפויות לטיפול תשתיתי")

    risk_filter = st.multiselect(
        "סנן לפי רמת סיכון:",
        ["🔴 גבוה", "🟡 בינוני", "🟢 נמוך"],
        default=["🔴 גבוה", "🟡 בינוני", "🟢 נמוך"],
    )
    tbl = sites_flt[sites_flt["דירוג"].isin(risk_filter)].copy()

    show = tbl[["אתר", "מחוז", "סוג_דרך", "תאונות", "ציון_סיכון", "דירוג"]].copy()
    show.columns = ["אתר תשתית", "מחוז", "סוג דרך", "מס' תאונות", "ציון סיכון (0–100)", "דירוג"]

    st.dataframe(
        show,
        use_container_width=True,
        height=460,
        column_config={
            "ציון סיכון (0–100)": st.column_config.ProgressColumn(
                "ציון סיכון (0–100)", min_value=0, max_value=100, format="%.1f"
            )
        },
    )

    csv_bytes = show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ ייצוא לCSV", csv_bytes, "hotspots_risk.csv", "text/csv")

    st.markdown("---")

    # Feature importance
    st.subheader("🔍 גורמי סיכון מרכזיים — Feature Importance")
    imp_df = importance.reset_index()
    imp_df.columns = ["גורם", "חשיבות"]
    fig_imp = px.bar(
        imp_df, x="חשיבות", y="גורם", orientation="h",
        color="חשיבות", color_continuous_scale="Reds",
        title="אילו גורמים מסבירים את חומרת התאונה?",
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # Risk map
    st.subheader("🗺️ מפת חום — ריכוזי סיכון")
    map_sites = tbl.dropna(subset=["lat", "lon"])
    if len(map_sites) > 0:
        fig_risk = px.scatter_map(
            map_sites,
            lat="lat", lon="lon",
            color="ציון_סיכון",
            size="תאונות",
            size_max=28,
            hover_name="אתר",
            hover_data={"תאונות": True, "ציון_סיכון": True, "דירוג": True,
                        "lat": False, "lon": False},
            color_continuous_scale="RdYlGn_r",
            zoom=7, center={"lat": 31.8, "lon": 35.0},
            map_style="carto-positron",
            title="מפת סיכון — אדום=גבוה, ירוק=נמוך",
            height=560,
        )
        fig_risk.update_coloraxes(colorbar_title="ציון סיכון")
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("אין נתוני מיקום לתצוגה בסינון הנוכחי")

# ══════════════════════════════ PREDICTION TAB ════════════════════════════════
with tab_pred:
    st.title("🔮 חיזוי פתרון תשתיתי לצומת")
    st.caption("המערכת מנתחת את פרופיל התאונות ומדרגת פתרונות לפי צפי להפחתת קטלניות")
    st.markdown("---")

    # ── Step 1: site selection ──
    st.subheader("1️⃣  בחר אתר לניתוח")
    all_sites = sorted(df["אתר"].dropna().unique().tolist())
    selected_site = st.selectbox("בחר צומת / אתר:", all_sites)

    site_df = df[df["אתר"] == selected_site]
    prof    = _accident_profile(site_df)

    st.markdown("---")

    # ── Step 2: accident profile ──
    st.subheader("2️⃣  פרופיל התאונות באתר")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("סה\"כ תאונות",        prof["total"])
    p2.metric("% קשות + קטלניות",    f"{(prof['fatal_pct']+prof['serious_pct'])*100:.1f}%")
    p3.metric("% תאונות לילה",       f"{prof['night_pct']*100:.1f}%")
    p4.metric("שעת שיא",             f"{prof['peak_hour']:02d}:00")

    pc1, pc2 = st.columns(2)

    with pc1:
        type_data = site_df["סוג_תאונה"].value_counts().reset_index()
        type_data.columns = ["סוג", "כמות"]
        fig = px.pie(type_data, names="סוג", values="כמות",
                     title="סוגי תאונות באתר", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with pc2:
        sev_data = site_df["חומרת_תאונה"].value_counts().reset_index()
        sev_data.columns = ["חומרה", "כמות"]
        fig = px.bar(sev_data, x="חומרה", y="כמות",
                     title="חומרת תאונות באתר",
                     color="חומרה",
                     color_discrete_map={"קלה": "#2ecc71", "קשה": "#e67e22", "קטלנית": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Step 3: site characteristics (user input) ──
    st.subheader("3️⃣  מאפייני הצומת")

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        approaches = st.slider("מספר כניסות לצומת:", 2, 6, 4)
    with ch2:
        traffic = st.select_slider(
            "נפח תנועה יומי משוער (כלי רכב):",
            options=[1000, 3000, 5000, 10000, 20000, 40000, 80000],
            value=10000,
        )
    with ch3:
        current = st.selectbox(
            "בקרה קיימת בצומת:",
            ["ללא בקרה", "תמרורי עצור / כניעה", "רמזור קיים", "כיכר קיימת"],
        )

    st.markdown("---")

    # ── Step 4: recommendations ──
    st.subheader("4️⃣  המלצות מדורגות לפתרון תשתיתי")

    ranked = _rank_interventions(prof, approaches, traffic, current)

    # Top recommendation
    top_name, top_score, top_info = ranked[0]
    st.success(
        f"**המלצה ראשית: {top_name}**  \n"
        f"{top_info['desc']}  \n"
        f"צפי להפחתת תאונות קטלניות: **{int(top_info['fatal']*100)}%** | "
        f"קשות: **{int(top_info['serious']*100)}%** | "
        f"עלות יחסית: **{top_info['cost']}**"
    )

    # All ranked alternatives
    st.markdown("#### השוואת כל הפתרונות")
    rank_df = pd.DataFrame([
        {
            "פתרון":              name,
            "ציון התאמה (0–100)": score,
            "הפחתת קטלניות":      f"{int(info['fatal']*100)}%",
            "הפחתת קשות":         f"{int(info['serious']*100)}%",
            "עלות":               info["cost"],
            "תיאור":              info["desc"],
        }
        for name, score, info in ranked
    ])

    st.dataframe(
        rank_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ציון התאמה (0–100)": st.column_config.ProgressColumn(
                "ציון התאמה (0–100)", min_value=0, max_value=100, format="%d"
            )
        },
    )

    # Expected impact chart
    fig_imp = px.bar(
        rank_df,
        x="פתרון", y="ציון התאמה (0–100)",
        color="ציון התאמה (0–100)",
        color_continuous_scale="RdYlGn",
        title="ציון התאמה לפי פתרון",
    )
    fig_imp.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚠️ ההמלצות מבוססות על פרופיל התאונות ההיסטורי ונתוני ספרות בינלאומית. "
        "לא כוללות נתוני נפח תנועה אמיתיים או מצב תשתית פיזית."
    )
