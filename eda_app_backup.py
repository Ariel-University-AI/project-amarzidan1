"""
מערכת תמיכה בהחלטות לשדרוג צמתים — משרד התחבורה
==================================================
נתונים: למ"ס PUF 2021 | עלויות: נתיבי ישראל + מכרזים רשמיים
"""

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="שדרוג צמתים — משרד התחבורה",
    layout="wide",
    page_icon="🚦",
)

# ── CBS decode tables ──────────────────────────────────────────────────────────
_SEV    = {1:"קטלנית",2:"קשה",3:"קלה"}
_ROAD   = {1:"עירוני - בצומת",2:"עירוני - לא בצומת",
           3:"בין-עירוני - בצומת",4:"בין-עירוני - לא בצומת",
           5:"חניון / כיכר",9:"אחר"}
_WTHR   = {1:"בהיר",2:"גשם קל",3:"גשם",4:"ערפל",5:"חול",7:"שלג",8:"סופה",9:"אחר"}
_SURF   = {1:"יבש",2:"רטוב",3:"קפוא",4:"שלג",9:"אחר"}
_DNTM   = {1:"יום",5:"לילה"}
_DWEEK  = {1:"ראשון",2:"שני",3:"שלישי",4:"רביעי",5:"חמישי",6:"שישי",7:"שבת"}
_DIST   = {1:"ירושלים",2:"צפון",3:"חיפה",4:"מרכז",5:"תל אביב",6:"דרום",7:'יו"ש'}
_SPD    = {1:"30",2:"40",3:"50",4:"60",5:"70",6:"80",7:"90",8:"100",9:"110"}
_ACCTYP = {1:"חזיתית",2:"אחורית",3:"צידית",4:"הולך רגל",
           5:"התהפכות",6:"פגיעה בעמוד",7:"נפילה מרכב",8:"אחר"}
_MONTHS = {1:"ינואר",2:"פברואר",3:"מרץ",4:"אפריל",5:"מאי",6:"יוני",
           7:"יולי",8:"אוגוסט",9:"ספטמבר",10:"אוקטובר",11:"נובמבר",12:"דצמבר"}
_PEAK   = {7,8,9,16,17,18,19}

# ── עלות תאונה רשמית (למ"ס / משרד התחבורה) ────────────────────────────────────
ACCIDENT_COST = {"קטלנית":3_500_000,"קשה":700_000,"קלה":50_000}

# ── הגדרת התערבויות + עלויות מאומתות ──────────────────────────────────────────
INTERVENTIONS = {
    "כיכר תנועה": {
        "desc":         "מעגל תנועה — מבטל התנגשויות חזיתיות ומאט תנועה",
        "fatal_r":      0.82, "serious_r": 0.55, "cong_r": 0.30,
        "cost_min":     400_000, "cost_avg": 500_000, "cost_max": 600_000,
        "cost_src":     "עיריית ירושלים 2023 — 9 כיכרות / 3.7M ₪",
        "cost_label":   "נמוך-בינוני",
        "tags":         {"חזיתית","התהפכות","מהירות_גבוהה"},
    },
    "רמזור חכם": {
        "desc":         "מערכת רמזורים אדפטיבית — עדיפות הולכי רגל ושעות עומס",
        "fatal_r":      0.45, "serious_r": 0.32, "cong_r": 0.25,
        "cost_min":     5_000_000, "cost_avg": 7_000_000, "cost_max": 10_000_000,
        "cost_src":     "נתיבי ישראל — פרויקטי אחזקה",
        "cost_label":   "גבוה",
        "tags":         {"הולך רגל","עירוני","עומס"},
    },
    "מעבר חצייה מוגן + תאורה": {
        "desc":         "הגבהה, תמרור, תאורה ממוקדת — יעיל לאזורים עם הולכי רגל",
        "fatal_r":      0.55, "serious_r": 0.38, "cong_r": 0.05,
        "cost_min":     50_000, "cost_avg": 120_000, "cost_max": 200_000,
        "cost_src":     "הערכה מקצועית",
        "cost_label":   "נמוך",
        "tags":         {"הולך רגל","עירוני"},
    },
    "תאורת לד מוגברת": {
        "desc":         "שיפור תאורה בצמתים ולאורך הכביש — יעיל בתאונות לילה",
        "fatal_r":      0.32, "serious_r": 0.22, "cong_r": 0.05,
        "cost_min":     9_000_000, "cost_avg": 9_500_000, "cost_max": 10_000_000,
        "cost_src":     "נתיבי ישראל — שחת-קציר",
        "cost_label":   "גבוה",
        "tags":         {"לילה","ערפל"},
    },
    "מצלמת אכיפה + מד-מהירות": {
        "desc":         'אכיפה אוטומטית — מורידה מהירות ממוצעת ב-7 קמ"ש',
        "fatal_r":      0.25, "serious_r": 0.18, "cong_r": 0.10,
        "cost_min":     100_000, "cost_avg": 120_000, "cost_max": 150_000,
        "cost_src":     "מכרז משטרת ישראל דצמבר 2025",
        "cost_label":   "נמוך",
        "tags":         {"מהירות_גבוהה","בין-עירוני"},
    },
    "פסי האטה / מוקפצים": {
        "desc":         "האטה פיזית — כניסות לישובים ואזורי מגורים",
        "fatal_r":      0.30, "serious_r": 0.25, "cong_r": 0.05,
        "cost_min":     50_000, "cost_avg": 70_000, "cost_max": 90_000,
        "cost_src":     "הערכה מקצועית (4 נתיבים)",
        "cost_label":   "נמוך",
        "tags":         {"אחורית","מהירות_בינונית"},
    },
    "הוספת נתיב נסיעה": {
        "desc":         "הרחבת הכביש בנתיב נוסף — מפחית עומסים ותאונות עורפיות",
        "fatal_r":      0.20, "serious_r": 0.35, "cong_r": 0.45,
        "cost_min":     3_000_000, "cost_avg": 8_000_000, "cost_max": 20_000_000,
        "cost_src":     "כביש 6 + כביש עוקף קריית אתא (2025)",
        "cost_label":   "גבוה מאוד",
        "tags":         {"אחורית","עומס","בין-עירוני"},
    },
}

_ROAD_ENC = {"עירוני - בצומת":1,"עירוני - לא בצומת":2,
             "בין-עירוני - בצומת":3,"בין-עירוני - לא בצומת":4,"חניון / כיכר":5}

# ── קואורדינטות ITM → WGS84 ────────────────────────────────────────────────────
def _itm_to_wgs84(x_s, y_s):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:2039","EPSG:4326",always_xy=True)
        lon,lat = tr.transform(x_s.values,y_s.values)
        return pd.Series(lat.round(6),index=x_s.index), pd.Series(lon.round(6),index=x_s.index)
    except Exception:
        lat = ((y_s-626907)/111320+31.5).round(5)
        lon = ((x_s-219529)/(111320*0.857)+35.21).round(5)
        return lat, lon

# ── שמות ישובים ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _city_map():
    try:
        r = requests.get(
            "https://data.gov.il/api/3/action/datastore_search"
            "?resource_id=5c78e9fa-c2e2-4771-93ff-7f400a12f7ba&limit=2000",
            timeout=8)
        return {int(c["סמל_ישוב"]):c["שם_ישוב"] for c in r.json()["result"]["records"] if c["סמל_ישוב"]}
    except Exception:
        return {}

# ── טעינת נתוני CBS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='טוען נתוני תאונות (למ"ס PUF 2021)…')
def load_data():
    city_map = _city_map()
    df = pd.read_csv("data/accidents_israel_2021_raw.csv",low_memory=False)
    df.columns = df.columns.str.strip()
    for c in ["HUMRAT_TEUNA","SUG_DEREH","SUG_TEUNA","MEZEG_AVIR","PNE_KVISH",
              "YOM_LAYLA","YOM_BASHAVUA","MAHOZ","MEHIRUT_MUTERET",
              "SEMEL_YISHUV","KVISH1","KM","ZOMET_IRONI","ZOMET_LO_IRONI",
              "HODESH_TEUNA","SHAA","X","Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c],errors="coerce")

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
    df["שעה"]          = (df["SHAA"].fillna(0)//4).clip(0,23).astype("Int64")
    df["שעה_מספר"]     = df["SHAA"].fillna(0).astype(int)
    df["שם_ישוב"] = df["SEMEL_YISHUV"].map(city_map)
    df["כביש"]    = df["KVISH1"].where(df["KVISH1"].notna()&(df["KVISH1"]>0))

    # ── זיהוי צומת ייחודי לפי קוד CBS ─────────────────────────────────────────
    def _site_id(r):
        road_type = str(r.get("SUG_DEREH",""))
        # עירוני בצומת: קוד ישוב + קוד צומת
        if road_type in {"1","5"}:
            zi = r.get("ZOMET_IRONI")
            city = r.get("שם_ישוב") or f"ישוב {int(r['SEMEL_YISHUV'])}" if pd.notna(r.get("SEMEL_YISHUV")) else "עירוני"
            if pd.notna(zi) and zi > 0:
                return f"{city} – צומת {int(zi)}"
            return f"{city} – עירוני כללי"
        # בין-עירוני בצומת: כביש + קמ
        if road_type == "3":
            k = r.get("כביש")
            km = r.get("KM")
            if pd.notna(k) and pd.notna(km):
                km_seg = round(float(km)/0.5)*0.5
                return f"כביש {int(k)} – ק\"מ {km_seg:.1f}"
            if pd.notna(k):
                return f"כביש {int(k)} – בצומת"
            zi2 = r.get("ZOMET_LO_IRONI")
            if pd.notna(zi2) and zi2 > 0:
                return f"כביש בין-עירוני – צומת {int(zi2)}"
        # כיכר / חניון
        city2 = r.get("שם_ישוב") or "כיכר"
        return f"{city2} – כיכר/חניון"

    df["מיקום"] = df.apply(
        lambda r: r["שם_ישוב"] if pd.notna(r["שם_ישוב"]) and r["שם_ישוב"]
                  else (f"כביש {int(r['כביש'])}" if pd.notna(r["כביש"]) else "לא ידוע"),
        axis=1)
    df["אתר"] = df.apply(_site_id, axis=1)
    mask = df["X"].notna()&df["Y"].notna()
    df["קו_רוחב"]=np.nan; df["קו_אורך"]=np.nan
    lat,lon = _itm_to_wgs84(df.loc[mask,"X"],df.loc[mask,"Y"])
    df.loc[mask,"קו_רוחב"]=lat.values; df.loc[mask,"קו_אורך"]=lon.values
    df["בשעת_עומס"] = df["שעה_מספר"].isin(_PEAK)
    return df

# ── פרופיל אתר + ציוני סיכון ──────────────────────────────────────────────────
@st.cache_data(show_spinner="מחשב ציוני סיכון ועומס לכל האתרים…")
def score_sites(_v="v4"):
    df = load_data()
    # סינון: רק צמתים ומחלפים
    df = df[df["סוג_דרך"].isin({"עירוני - בצומת","בין-עירוני - בצומת","חניון / כיכר"})].copy()
    records = []
    for site, grp in df.groupby("אתר"):
        n = max(len(grp),1)
        spd   = grp["מהירות_מותרת"].mode()
        spd_v = spd.iloc[0] if len(spd) else "50"
        road  = grp["סוג_דרך"].mode()
        road_v= road.iloc[0] if len(road) else "אחר"
        rec = {
            "אתר":           site,
            "תאונות":        n,
            "fatal_pct":     (grp["חומרת_תאונה"]=="קטלנית").sum()/n,
            "serious_pct":   (grp["חומרת_תאונה"]=="קשה").sum()/n,
            "night_pct":     (grp["חלק_יממה"]=="לילה").sum()/n,
            "rain_pct":      grp["מזג_אוויר"].isin(["גשם","גשם קל"]).sum()/n,
            "pedestrian_pct":(grp["סוג_תאונה"]=="הולך רגל").sum()/n,
            "frontal_pct":   (grp["סוג_תאונה"]=="חזיתית").sum()/n,
            "rear_pct":      (grp["סוג_תאונה"]=="אחורית").sum()/n,
            "rollover_pct":  (grp["סוג_תאונה"]=="התהפכות").sum()/n,
            "peak_pct":      grp["בשעת_עומס"].sum()/n,
            "high_speed":    1 if str(spd_v) in {"70","80","90","100","110"} else 0,
            "סוג_דרך":       road_v,
            "מחוז":          grp["מחוז"].mode().iloc[0] if grp["מחוז"].notna().any() else "—",
            "lat":           grp["קו_רוחב"].mean(),
            "lon":           grp["קו_אורך"].mean(),
        }
        # ציון סיכון תאונות
        rec["ציון_תאונות_גלמי"] = (rec["fatal_pct"]*20+rec["serious_pct"]*5)*n
        # מדד עומס: % אחוריות × 0.6 + % שעות עומס × 0.4
        rec["מדד_עומס_גלמי"] = rec["rear_pct"]*0.6 + rec["peak_pct"]*0.4

        # התערבות מומלצת (rule-based)
        scores = {}
        for name,info in INTERVENTIONS.items():
            s = (rec["fatal_pct"]*info["fatal_r"]+rec["serious_pct"]*info["serious_r"])*n*10
            if rec["pedestrian_pct"]>0.12 and "הולך רגל"      in info["tags"]: s+=25
            if rec["frontal_pct"]   >0.18 and "חזיתית"        in info["tags"]: s+=25
            if rec["night_pct"]     >0.35 and "לילה"          in info["tags"]: s+=20
            if rec["high_speed"]          and "מהירות_גבוהה"  in info["tags"]: s+=20
            if rec["rear_pct"]      >0.20 and "אחורית"        in info["tags"]: s+=15
            if name=="הוספת נתיב נסיעה"  and rec["high_speed"]:               s+=15
            if name=="הוספת נתיב נסיעה"  and rec["rear_pct"]>0.25:            s+=20
            scores[name] = max(0.0,s)
        rec["התערבות_מומלצת"] = max(scores,key=scores.get)
        info_top = INTERVENTIONS[rec["התערבות_מומלצת"]]
        rec["עלות_ממוצעת_₪"]  = info_top["cost_avg"]
        rec["עלות_label"]      = info_top["cost_label"]
        records.append(rec)

    agg = pd.DataFrame(records)
    # נרמול לציון 0-100
    mx_acc = agg["ציון_תאונות_גלמי"].max() or 1
    mx_cng = agg["מדד_עומס_גלמי"].max()   or 1
    agg["ציון_תאונות"]  = (agg["ציון_תאונות_גלמי"]/mx_acc*100).round(1)
    agg["מדד_עומס"]     = (agg["מדד_עומס_גלמי"]   /mx_cng*100).round(1)
    agg["ציון_משולב"]   = (agg["ציון_תאונות"]*0.65+agg["מדד_עומס"]*0.35).round(1)
    agg["דירוג"] = agg["ציון_משולב"].apply(
        lambda s:"🔴 גבוה" if s>=60 else("🟡 בינוני" if s>=30 else "🟢 נמוך"))

    # חישוב ROI שנתי
    def _roi(row):
        n     = row["תאונות"]
        fatal = row["fatal_pct"]*n*ACCIDENT_COST["קטלנית"]
        ser   = row["serious_pct"]*n*ACCIDENT_COST["קשה"]
        light = (1-row["fatal_pct"]-row["serious_pct"])*n*ACCIDENT_COST["קלה"]
        annual_cost = fatal+ser+light
        iv = INTERVENTIONS[row["התערבות_מומלצת"]]
        saving = annual_cost*(iv["fatal_r"]*0.7+iv["serious_r"]*0.3)
        inv    = iv["cost_avg"]
        return round(saving*5/inv*100,1) if inv>0 else 0
    agg["ROI_5yr_%"] = agg.apply(_roi,axis=1)

    return agg.sort_values("ציון_משולב",ascending=False).reset_index(drop=True)

# snap_to_roads הוסר — pyproj מספיק מדויק (דיוק ~50מ')
def snap_to_roads(_v="v2"):
    return score_sites("v4")

# ── דירוג התערבויות לאתר ──────────────────────────────────────────────────────
def rank_interventions(prof, approaches=4, traffic=10000, current="ללא בקרה"):
    results=[]
    n = prof.get("total",prof.get("תאונות",1))
    for name,info in INTERVENTIONS.items():
        s=(prof.get("fatal_pct",0)*info["fatal_r"]+
           prof.get("serious_pct",0)*info["serious_r"])*n*10
        if prof.get("pedestrian_pct",0)>0.12 and "הולך רגל"     in info["tags"]: s+=25
        if prof.get("frontal_pct",0)   >0.18 and "חזיתית"       in info["tags"]: s+=25
        if prof.get("night_pct",0)     >0.35 and "לילה"         in info["tags"]: s+=20
        if prof.get("high_speed",0)          and "מהירות_גבוהה" in info["tags"]: s+=20
        if prof.get("rear_pct",0)      >0.20 and "אחורית"       in info["tags"]: s+=15
        if name=="כיכר תנועה"          and approaches in {3,4}:                   s+=10
        if name=="רמזור חכם"           and traffic>10000:                          s+=10
        if name=="הוספת נתיב נסיעה"   and traffic>20000:                          s+=25
        if name=="הוספת נתיב נסיעה"   and prof.get("high_speed",0):               s+=15
        if name=="הוספת נתיב נסיעה"   and prof.get("rear_pct",0)>0.25:            s+=20
        if name=="רמזור חכם"           and current=="רמזור קיים":                 s-=35
        if name=="כיכר תנועה"         and current=="כיכר קיימת":                 s-=35
        scores_entry = max(0.0,s)

        # ROI
        annual_cost = (prof.get("fatal_pct",0)*n*ACCIDENT_COST["קטלנית"]+
                       prof.get("serious_pct",0)*n*ACCIDENT_COST["קשה"]+
                       (1-prof.get("fatal_pct",0)-prof.get("serious_pct",0))*n*ACCIDENT_COST["קלה"])
        saving = annual_cost*(info["fatal_r"]*0.7+info["serious_r"]*0.3)
        roi5   = round(saving*5/info["cost_avg"]*100,1) if info["cost_avg"]>0 else 0

        results.append({
            "פתרון":              name,
            "raw_score":          scores_entry,
            "הפחתת קטלניות":      f"{int(info['fatal_r']*100)}%",
            "הפחתת קשות":         f"{int(info['serious_r']*100)}%",
            "הפחתת עומס":         f"{int(info['cong_r']*100)}%",
            "עלות מינימום":        f"₪{info['cost_min']:,}",
            "עלות ממוצעת":         f"₪{info['cost_avg']:,}",
            "עלות מקסימום":        f"₪{info['cost_max']:,}",
            "מקור עלות":          info["cost_src"],
            "ROI 5 שנים":         f"{roi5:.0f}%",
            "תיאור":              info["desc"],
            "cost_avg_num":       info["cost_avg"],
            "roi_num":            roi5,
        })
    mx = max((r["raw_score"] for r in results),default=1) or 1
    for r in results:
        r["ציון התאמה"] = round(r["raw_score"]/mx*100)
    return sorted(results,key=lambda x:x["ציון התאמה"],reverse=True)

# ── מודל ML ───────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_MDIR = os.path.join(_BASE,"models")

@st.cache_resource(show_spinner="טוען מודל ML…")
def load_ml_model(_v="v2"):
    mp = os.path.join(_MDIR,"model.pkl")
    ep = os.path.join(_MDIR,"encoders.pkl")
    if os.path.exists(mp) and os.path.exists(ep):
        clf = joblib.load(mp)
        enc = joblib.load(ep)
        return clf,enc
    return None,None

@st.cache_data(show_spinner=False)
def load_metrics(_v="v2"):
    p = os.path.realpath(os.path.join(_MDIR,"metrics.json"))
    base = os.path.realpath(_MDIR)
    if not p.startswith(base):
        return None
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False, ttl=3600)
def detect_existing_infra(lat: float, lon: float) -> dict:
    """שאילתת Overpass API לזיהוי תשתית קיימת בצומת."""
    try:
        q = f"""
[out:json][timeout:8];
(
  way["junction"="roundabout"](around:80,{lat},{lon});
  node["highway"="traffic_signals"](around:60,{lat},{lon});
  node["highway"="stop"](around:60,{lat},{lon});
  node["highway"="give_way"](around:60,{lat},{lon});
  node["speed_camera"](around:80,{lat},{lon});
  node["highway"="speed_camera"](around:80,{lat},{lon});
  way["highway"="living_street"](around:60,{lat},{lon});
);
out body;
"""
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data=q, timeout=10)
        elements = r.json().get("elements", [])
        found = {"כיכר": False, "רמזור": False, "עצור": False, "מצלמה": False}
        for el in elements:
            tags = el.get("tags", {})
            if tags.get("junction") == "roundabout":          found["כיכר"]   = True
            if tags.get("highway") == "traffic_signals":      found["רמזור"]  = True
            if tags.get("highway") in ("stop","give_way"):    found["עצור"]   = True
            if "speed_camera" in str(tags):                   found["מצלמה"]  = True
        return found
    except Exception:
        return {}

@st.cache_resource(show_spinner="בונה מודל חיזוי מרחבי…")
def build_planning_model():
    sites = snap_to_roads("v2").dropna(subset=["lat","lon"]).copy()
    sites["road_enc"] = sites["סוג_דרך"].map(_ROAD_ENC).fillna(0)
    X = sites[["lat","lon","road_enc","high_speed"]].values
    y = sites["ציון_משולב"].values
    model = RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=-1)
    model.fit(X,y)
    nn = NearestNeighbors(n_neighbors=5,algorithm="ball_tree")
    nn.fit(sites[["lat","lon"]].values)
    return model,nn,sites

# ── טעינה ────────────────────────────────────────────────────────────────────
df    = load_data()
sites = snap_to_roads("v2")   # קואורדינטות מדויקות לכביש (OSRM)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 סינון")
    dist_opts = ["הכל"]+sorted(df["מחוז"].dropna().unique().tolist())
    dist_sel  = st.selectbox("מחוז:",dist_opts)
    risk_sel  = st.multiselect("רמת סיכון:",["🔴 גבוה","🟡 בינוני","🟢 נמוך"],
                               default=["🔴 גבוה","🟡 בינוני","🟢 נמוך"])
    st.markdown("---")
    st.caption('נתונים: למ"ס PUF 2021 | עלויות: נתיבי ישראל + מכרזים')

sites_flt = sites.copy()
if dist_sel!="הכל":
    sites_flt = sites_flt[sites_flt["מחוז"]==dist_sel]
if risk_sel:
    sites_flt = sites_flt[sites_flt["דירוג"].isin(risk_sel)]

# ── דף יחיד — ללא טאבים ──────────────────────────────────────────────────────
tab1 = tab2 = tab3 = tab4 = tab5 = st.container()

st.title("🚦 מערכת ניתוח וחיזוי צמתים — ישראל 2021")
st.caption('משרד התחבורה | נתונים: למ"ס PUF 2021 | עלויות: נתיבי ישראל + מכרזים')

with st.container():
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 M1 — ניתוח תיאורי (Descriptive)")
    st.caption("מה קרה? סקירת מצב הצמתים הבעייתיים בישראל לפי תאונות ועומסי תנועה")

    df_junc = df[df["סוג_דרך"].isin({"עירוני - בצומת","בין-עירוני - בצומת","חניון / כיכר"})]

    # ── בלוק 1: תאונות דרכים ──────────────────────────────────────────────────
    st.markdown("#### 🚗 תאונות דרכים בצמתים")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("סה\"כ תאונות בצמתים",   f"{len(df_junc):,}")
    k2.metric("קטלניות ☠️",             f"{(df_junc['חומרת_תאונה']=='קטלנית').sum():,}")
    k3.metric("קשות 🟠",                f"{(df_junc['חומרת_תאונה']=='קשה').sum():,}")
    k4.metric("קלות 🟢",                f"{(df_junc['חומרת_תאונה']=='קלה').sum():,}")
    k5.metric("עלות שנתית כלכלית",
              f"₪{int(sum(df_junc['חומרת_תאונה'].map(ACCIDENT_COST).dropna())/1e6):.0f}M")

    st.markdown("---")

    # ── בלוק 2: עומסי תנועה (מדדים מבוססי CBS) ────────────────────────────────
    st.markdown("#### 🚦 עומסי תנועה בצמתים")
    st.caption("מדד עומס = % תאונות עורפיות (פקק) × 0.6 + % תאונות בשעות עומס × 0.4")
    peak_total      = int(df_junc["בשעת_עומס"].sum())
    peak_pct        = df_junc["בשעת_עומס"].mean() * 100
    rear_total      = int((df_junc["סוג_תאונה"]=="אחורית").sum())
    rear_pct_avg    = (df_junc["סוג_תאונה"]=="אחורית").mean() * 100
    high_cong_sites = int((sites_flt["מדד_עומס"] >= 50).sum())
    avg_cong        = sites_flt["מדד_עומס"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("תאונות בשעות עומס",   f"{peak_total:,}",
              f"{peak_pct:.1f}% מסך התאונות בצמתים")
    c2.metric("תאונות עורפיות (פקק)", f"{rear_total:,}",
              f"{rear_pct_avg:.1f}% — מדד צפיפות תנועה")
    c3.metric("צמתים עם עומס גבוה",     f"{high_cong_sites:,}",
              "מדד עומס ≥ 50/100")
    c4.metric("מדד עומס ממוצע לצומת",  f"{avg_cong:.1f} / 100")
    c5.metric("שעות שיא",               "07:00–09:00 | 16:00–19:00",
              "מקור: CBS — נתוני שעת תאונה")

    st.markdown("---")

    # ── בלוק 3: ציון סיכון משולב ──────────────────────────────────────────────
    st.markdown("#### 🎯 ציון סיכון משולב (תאונות 65% + עומס 35%)")
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("אתרים 🔴 סיכון גבוה",   f"{(sites_flt['דירוג']=='🔴 גבוה').sum():,}")
    r2.metric("אתרים 🟡 סיכון בינוני", f"{(sites_flt['דירוג']=='🟡 בינוני').sum():,}")
    r3.metric("אתרים 🟢 סיכון נמוך",  f"{(sites_flt['דירוג']=='🟢 נמוך').sum():,}")
    r4.metric("סה\"כ צמתים מנותחים",   f"{len(sites_flt):,}")

    st.markdown("---")

    # ── מפה אינטראקטיבית — לחץ לניתוח ────────────────────────────────────────
    st.info("👆 **לחץ על צומת במפה** לקבלת ניתוח מלא מתחת")
    map_data = sites_flt.dropna(subset=["lat","lon"])
    if len(map_data)>0:
        map_data = map_data.copy()
        map_data["custom_site"] = map_data["אתר"]
        fig_map = px.scatter_map(
            map_data, lat="lat", lon="lon",
            color="ציון_משולב", size="תאונות", size_max=28,
            custom_data=["custom_site"],
            hover_name="אתר",
            hover_data={"תאונות":True,"ציון_תאונות":True,"מדד_עומס":True,
                        "ציון_משולב":True,"דירוג":True,"התערבות_מומלצת":True,
                        "lat":False,"lon":False,"custom_site":False},
            color_continuous_scale="RdYlGn_r",
            zoom=7, center={"lat":31.8,"lon":35.0},
            map_style="open-street-map",
            height=520,
            title="🔴 אדום = סיכון גבוה | גודל נקודה = מספר תאונות | לחץ לניתוח",
        )
        fig_map.update_coloraxes(colorbar_title="ציון משולב")
        map_event = st.plotly_chart(
            fig_map, use_container_width=True,
            on_select="rerun", selection_mode=["points"],
            key="main_map"
        )
        # עדכון צומת נבחר בלחיצה
        if map_event and map_event.selection and map_event.selection.points:
            cd = map_event.selection.points[0].get("customdata",[None])
            if cd and cd[0]:
                st.session_state["selected_site"] = cd[0]
    else:
        st.info("אין נתוני מיקום בסינון הנוכחי")

    st.markdown("---")

    # טבלת דירוג
    st.subheader("📋 טבלת צמתים מדורגת")
    show_cols = ["אתר","מחוז","סוג_דרך","תאונות","ציון_תאונות","מדד_עומס",
                 "ציון_משולב","דירוג","התערבות_מומלצת","עלות_ממוצעת_₪","ROI_5yr_%"]
    tbl = sites_flt[show_cols].copy()
    tbl.columns = ["אתר","מחוז","סוג דרך","תאונות","ציון תאונות","מדד עומס",
                   "ציון משולב","דירוג","התערבות מומלצת","עלות ממוצעת ₪","ROI 5 שנים %"]
    st.dataframe(tbl, use_container_width=True, height=400,
        column_config={
            "ציון משולב": st.column_config.ProgressColumn("ציון משולב",min_value=0,max_value=100,format="%.1f"),
            "ציון תאונות": st.column_config.ProgressColumn("ציון תאונות",min_value=0,max_value=100,format="%.1f"),
            "מדד עומס": st.column_config.ProgressColumn("מדד עומס",min_value=0,max_value=100,format="%.1f"),
        })

    csv = tbl.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ ייצוא לCSV",csv,"hotspots.csv","text/csv")

    st.markdown("---")
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 M3 — מנחה (Prescriptive)")
    st.caption("מה כדאי לעשות? ניתוח צומת, המלצת פתרון, עלויות וחיזוי השפעה")
    st.subheader("🔍 ניתוח צומת נבחר")

    junction_mask_t1 = df["סוג_דרך"].isin({"עירוני - בצומת","בין-עירוני - בצומת","חניון / כיכר"})
    all_sites_t1 = sorted(df[junction_mask_t1]["אתר"].dropna().unique().tolist())

    if "selected_site" not in st.session_state or st.session_state["selected_site"] not in all_sites_t1:
        st.session_state["selected_site"] = all_sites_t1[0] if all_sites_t1 else ""

    selected = st.selectbox(
        "בחר צומת (או לחץ על נקודה במפה למעלה):",
        all_sites_t1,
        index=all_sites_t1.index(st.session_state["selected_site"]) if st.session_state["selected_site"] in all_sites_t1 else 0,
        key="t1_site_select"
    )
    st.session_state["selected_site"] = selected
    site_df = df[df["אתר"]==selected]
    n       = max(len(site_df),1)

    # ── זיהוי OSM אוטומטי בשינוי צומת ───────────────────────────────────────
    site_lat_t1 = site_df["קו_רוחב"].dropna()
    site_lon_t1 = site_df["קו_אורך"].dropna()

    if len(site_lat_t1)>0 and f"osm_{selected}" not in st.session_state:
        raw_lat = float(site_lat_t1.mean())
        raw_lon = float(site_lon_t1.mean())
        with st.spinner("🔍 מדייק מיקום ומזהה תשתית..."):
            # Snap center to nearest road (1 request only)
            try:
                r = requests.get(
                    f"http://router.project-osrm.org/nearest/v1/driving/{raw_lon:.6f},{raw_lat:.6f}",
                    timeout=5)
                d = r.json()
                if d.get("code") == "Ok":
                    loc = d["waypoints"][0]["location"]
                    st.session_state[f"snap_{selected}"] = (loc[1], loc[0])
            except Exception:
                st.session_state[f"snap_{selected}"] = (raw_lat, raw_lon)
            st.session_state[f"osm_{selected}"] = detect_existing_infra(
                st.session_state[f"snap_{selected}"][0],
                st.session_state[f"snap_{selected}"][1])

    osm_i = st.session_state.get(f"osm_{selected}", {})

    # ── תג תשתית קיימת ───────────────────────────────────────────────────────
    if osm_i:
        tags = []
        if osm_i.get("כיכר"):  tags.append("🔵 כיכר תנועה קיימת")
        if osm_i.get("רמזור"): tags.append("🚦 רמזור קיים")
        if osm_i.get("עצור"):  tags.append("🛑 תמרורי עצור")
        if osm_i.get("מצלמה"):tags.append("📷 מצלמת אכיפה")
        if tags:
            st.info("**תשתית קיימת בצומת:** " + " | ".join(tags))

    # ── מפה עם נקודה מדויקת לכביש + עיגול מסמן ────────────────────────────
    if len(site_lat_t1)>0:
        # השתמש בנקודה הקרובה לכביש אם זמינה
        snap = st.session_state.get(f"snap_{selected}")
        clat = snap[0] if snap else float(site_lat_t1.mean())
        clon = snap[1] if snap else float(site_lon_t1.mean())

        pts = site_df.dropna(subset=["קו_רוחב","קו_אורך"]).copy()
        fig_loc = px.scatter_map(pts, lat="קו_רוחב", lon="קו_אורך",
            color="חומרת_תאונה",
            color_discrete_map={"קלה":"#2ecc71","קשה":"#e67e22","קטלנית":"#e74c3c"},
            hover_data={"חומרת_תאונה":True,"סוג_תאונה":True,"חלק_יממה":True,
                        "קו_רוחב":False,"קו_אורך":False},
            zoom=16, center={"lat":clat,"lon":clon},
            map_style="open-street-map", height=400,
            title=f"📍 {selected}")
        fig_loc.update_traces(marker=dict(size=12, opacity=0.9))
        # עיגול שקוף — מסגרת הצומת
        fig_loc.add_trace(go.Scattermap(
            lat=[clat], lon=[clon], mode="markers",
            marker=dict(size=55, color="rgba(0,100,255,0.15)", symbol="circle"),
            hoverinfo="skip", showlegend=False, name=""))
        # נקודה + תווית
        fig_loc.add_trace(go.Scattermap(
            lat=[clat], lon=[clon], mode="markers+text",
            marker=dict(size=16, color="#1a6bff"),
            text=["📍 מיקום הצומת"], textposition="top right",
            textfont=dict(size=12, color="#1a6bff"),
            hovertext=f"{selected}", hoverinfo="text",
            showlegend=False, name=""))
        st.plotly_chart(fig_loc, use_container_width=True, key="pc2")

    # KPIs
    spd_m=site_df["מהירות_מותרת"].mode(); spd_v=spd_m.iloc[0] if len(spd_m) else "50"
    hr_m=site_df["שעה_מספר"].mode()
    prof={
        "total":n,
        "fatal_pct":      (site_df["חומרת_תאונה"]=="קטלנית").sum()/n,
        "serious_pct":    (site_df["חומרת_תאונה"]=="קשה").sum()/n,
        "night_pct":      (site_df["חלק_יממה"]=="לילה").sum()/n,
        "rain_pct":       site_df["מזג_אוויר"].isin(["גשם","גשם קל"]).sum()/n,
        "pedestrian_pct": (site_df["סוג_תאונה"]=="הולך רגל").sum()/n,
        "frontal_pct":    (site_df["סוג_תאונה"]=="חזיתית").sum()/n,
        "rear_pct":       (site_df["סוג_תאונה"]=="אחורית").sum()/n,
        "rollover_pct":   (site_df["סוג_תאונה"]=="התהפכות").sum()/n,
        "peak_pct":       site_df["בשעת_עומס"].sum()/n,
        "high_speed":     1 if str(spd_v) in {"70","80","90","100","110"} else 0,
        "road_type":      site_df["סוג_דרך"].mode().iloc[0] if len(site_df) else "—",
    }
    p1,p2,p3,p4,p5 = st.columns(5)
    p1.metric("סה\"כ תאונות",        prof["total"])
    p2.metric("% קטלניות",          f"{prof['fatal_pct']*100:.1f}%")
    p3.metric("% תאונות לילה",      f"{prof['night_pct']*100:.1f}%")
    p4.metric("% עורפיות (פקק) 🚦", f"{prof['rear_pct']*100:.1f}%")
    p5.metric("% שעות עומס",        f"{prof['peak_pct']*100:.1f}%")

    n_fatal=round(prof["fatal_pct"]*n); n_serious=round(prof["serious_pct"]*n); n_light=n-n_fatal-n_serious
    annual=(n_fatal*ACCIDENT_COST["קטלנית"]+n_serious*ACCIDENT_COST["קשה"]+n_light*ACCIDENT_COST["קלה"])
    with st.expander(f"💰 עלות כלכלית שנתית: ₪{annual:,.0f} — לחץ לפירוט"):
        st.markdown(f"""
| חומרה | מספר | עלות ליחידה | סה"כ |
|-------|------|------------|------|
| קטלנית ☠️ | {n_fatal} | ₪3,500,000 | ₪{n_fatal*3_500_000:,} |
| קשה 🟠 | {n_serious} | ₪700,000 | ₪{n_serious*700_000:,} |
| קלה 🟢 | {n_light} | ₪50,000 | ₪{n_light*50_000:,} |
| **סה"כ** | **{n}** | | **₪{annual:,}** |
*מקור: משרד התחבורה*""")

    # גרפים + מצב קיים + המלצה
    ga,gb,gc = st.columns(3)
    with ga:
        td=site_df["סוג_תאונה"].value_counts().reset_index(); td.columns=["סוג","כמות"]
        fig=px.pie(td,names="סוג",values="כמות",title="סוגי תאונות",hole=0.35)
        st.plotly_chart(fig,use_container_width=True, key="pc3")
    with gb:
        hd=site_df.groupby("שעה_מספר").size().reset_index(name="כמות")
        hd["סוג_שעה"]=hd["שעה_מספר"].apply(lambda h:"🚦 עומס" if h in _PEAK else "רגיל")
        fig=px.bar(hd,x="שעה_מספר",y="כמות",color="סוג_שעה",
                   color_discrete_map={"🚦 עומס":"#e74c3c","רגיל":"#95a5a6"},
                   title="תאונות לפי שעה — עומס מסומן")
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig,use_container_width=True, key="pc4")
    with gc:
        cong_sc=round((prof["rear_pct"]*0.6+prof["peak_pct"]*0.4)*100,1)
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number",value=cong_sc,
            title={"text":"מדד עומס תנועה"},number={"suffix":" / 100"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#e74c3c" if cong_sc>=60 else("#e67e22" if cong_sc>=30 else "#2ecc71")},
                   "steps":[{"range":[0,30],"color":"#d5f5e3"},{"range":[30,60],"color":"#fdebd0"},
                             {"range":[60,100],"color":"#fadbd8"}]}))
        fig_g.update_layout(height=240)
        st.plotly_chart(fig_g,use_container_width=True, key="pc5")

    # מצב קיים + המלצה
    st.markdown("**🔍 מצב קיים + המלצה:**")
    osm_i = st.session_state.get(f"osm_{selected}", {})
    if osm_i.get("כיכר"):    def_cur="כיכר קיימת"
    elif osm_i.get("רמזור"): def_cur="רמזור קיים"
    elif osm_i.get("עצור"):  def_cur="תמרורי עצור / כניעה"
    else:                     def_cur="ללא בקרה"

    cur_opts=["ללא בקרה","תמרורי עצור / כניעה","רמזור קיים","כיכר קיימת","מצלמת אכיפה קיימת","פסי האטה קיימים","תאורה מוגברת קיימת"]
    hc1,hc2,hc3=st.columns(3)
    current_t1=hc1.selectbox("בקרה קיימת:",cur_opts,index=cur_opts.index(def_cur),key="t1_current")
    approaches_t1=hc2.slider("מספר כניסות:",2,6,4,key="t1_app")
    traffic_t1=hc3.select_slider("נפח תנועה:",options=[1000,3000,5000,10000,20000,40000,80000],value=10000,key="t1_traffic")

    ranked_t1=rank_interventions(prof,approaches_t1,traffic_t1,current_t1)
    cur_map={"כיכר קיימת":"כיכר תנועה","רמזור קיים":"רמזור חכם",
             "מצלמת אכיפה קיימת":"מצלמת אכיפה + מד-מהירות","פסי האטה קיימים":"פסי האטה / מוקפצים"}

    st.markdown("---")
    st.subheader("🏆 פתרונות מדורגים")

    rank_disp=pd.DataFrame([{
        "פתרון":r["פתרון"],"ציון":r["ציון התאמה"],
        "הפחתת קטלניות":r["הפחתת קטלניות"],
        "הפחתת עומס":r["הפחתת עומס"],
        "עלות ממוצעת":r["עלות ממוצעת"],
        "ROI 5 שנים":r["ROI 5 שנים"],
    } for r in ranked_t1])
    st.dataframe(rank_disp, use_container_width=True, hide_index=True,
        column_config={"ציון":st.column_config.ProgressColumn(
            "ציון התאמה",min_value=0,max_value=100,format="%d")})

    st.markdown("---")
    st.subheader("🔮 חיזוי השפעת הפתרון הנבחר")

    chosen=st.selectbox(
        "בחר פתרון לחיזוי:",
        [r["פתרון"] for r in ranked_t1],
        key="t1_chosen_intervention"
    )
    chosen_info = INTERVENTIONS.get(chosen, {})
    chosen_rank = next((r for r in ranked_t1 if r["פתרון"]==chosen), ranked_t1[0])

    if cur_map.get(current_t1)==chosen:
        st.warning(f"⚠️ {chosen} כבר קיים בצומת! בחר פתרון אחר.")
    else:
        # ── חישוב חיזוי ────────────────────────────────────────────────────
        fatal_r   = chosen_info.get("fatal_r", 0)
        serious_r = chosen_info.get("serious_r", 0)
        cong_r    = chosen_info.get("cong_r", 0)
        cost_avg  = chosen_info.get("cost_avg", 0)

        n_fatal_now   = round(prof["fatal_pct"]   * n)
        n_serious_now = round(prof["serious_pct"] * n)
        n_light_now   = n - n_fatal_now - n_serious_now

        n_fatal_after   = round(n_fatal_now   * (1 - fatal_r))
        n_serious_after = round(n_serious_now * (1 - serious_r))
        n_light_after   = n_light_now  # קלות פחות מושפעות

        cost_now  = (n_fatal_now*ACCIDENT_COST["קטלנית"] +
                     n_serious_now*ACCIDENT_COST["קשה"]   +
                     n_light_now*ACCIDENT_COST["קלה"])
        cost_after= (n_fatal_after*ACCIDENT_COST["קטלנית"] +
                     n_serious_after*ACCIDENT_COST["קשה"]  +
                     n_light_after*ACCIDENT_COST["קלה"])
        annual_saving = cost_now - cost_after
        roi5 = round(annual_saving*5/cost_avg*100,1) if cost_avg>0 else 0

        cong_now   = round((prof["rear_pct"]*0.6+prof["peak_pct"]*0.4)*100,1)
        cong_after = round(cong_now*(1-cong_r),1)

        # ── תצוגה ──────────────────────────────────────────────────────────
        f1,f2,f3,f4 = st.columns(4)
        f1.metric("💰 עלות ההשקעה",          f"₪{cost_avg:,}",
                  chosen_info.get("cost_src",""))
        f2.metric("📉 חיסכון שנתי צפוי",     f"₪{annual_saving:,.0f}",
                  f"ROI ל-5 שנים: {roi5:.0f}%")
        f3.metric("🚗 הפחתת תאונות קטלניות", f"{int(fatal_r*100)}%",
                  f"{n_fatal_now} ← {n_fatal_after} קטלניות/שנה")
        f4.metric("🚦 הפחתת עומס תנועה",     f"{int(cong_r*100)}%",
                  f"מדד עומס: {cong_now} ← {cong_after}")

        # ── גרף לפני / אחרי ────────────────────────────────────────────────
        comp_df = pd.DataFrame({
            "מצב":      ["לפני שדרוג","אחרי שדרוג","לפני שדרוג","אחרי שדרוג","לפני שדרוג","אחרי שדרוג"],
            "חומרה":    ["קטלנית","קטלנית","קשה","קשה","קלה","קלה"],
            "תאונות":   [n_fatal_now,n_fatal_after,n_serious_now,n_serious_after,n_light_now,n_light_after],
        })
        fig_comp=px.bar(comp_df, x="חומרה", y="תאונות", color="מצב",
            barmode="group",
            color_discrete_map={"לפני שדרוג":"#e74c3c","אחרי שדרוג":"#2ecc71"},
            title=f"תאונות לפני ואחרי — {chosen}",
            labels={"תאונות":"מספר תאונות לשנה"})
        st.plotly_chart(fig_comp, use_container_width=True, key="pc_comp")

        # ── גרף עומס לפני / אחרי ───────────────────────────────────────────
        cong_comp=pd.DataFrame({
            "מצב":["לפני","אחרי"],
            "מדד עומס":[cong_now,cong_after]
        })
        fig_cong=px.bar(cong_comp,x="מצב",y="מדד עומס",
            color="מצב",color_discrete_map={"לפני":"#e74c3c","אחרי":"#2ecc71"},
            title=f"מדד עומס תנועה לפני ואחרי — {chosen}",
            text="מדד עומס")
        fig_cong.update_traces(textposition="outside")
        fig_cong.update_layout(yaxis_range=[0,100],showlegend=False)
        st.plotly_chart(fig_cong, use_container_width=True, key="pc_cong")

        st.info(
            f"📋 **{chosen}** — {chosen_info.get('desc','')}  \n"
            f"מקור עלות: {chosen_info.get('cost_src','')}"
        )

    st.markdown("---")
    st.subheader("📊 ניתוח EDA — נתונים כלליים")
    e1,e2 = st.columns(2)
    with e1:
        sv = df["חומרת_תאונה"].value_counts().reset_index()
        sv.columns=["חומרה","כמות"]
        fig=px.pie(sv,names="חומרה",values="כמות",title="התפלגות חומרת תאונות",
                   color="חומרה",color_discrete_map={"קלה":"#2ecc71","קשה":"#e67e22","קטלנית":"#e74c3c"},hole=0.4)
        st.plotly_chart(fig,use_container_width=True, key="pc6")
    with e2:
        hr = df.groupby("שעה_מספר").size().reset_index(name="כמות")
        fig=px.bar(hr,x="שעה_מספר",y="כמות",title="תאונות לפי שעה",
                   color="כמות",color_continuous_scale="Reds")
        fig.add_vrect(x0=6.5,x1=9.5,fillcolor="orange",opacity=0.15,annotation_text="עומס בוקר")
        fig.add_vrect(x0=15.5,x1=19.5,fillcolor="orange",opacity=0.15,annotation_text="עומס ערב")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True, key="pc7")

    e3,e4 = st.columns(2)
    with e3:
        dist=df_junc["מחוז"].value_counts().reset_index(); dist.columns=["מחוז","כמות"]
        fig=px.bar(dist,x="כמות",y="מחוז",orientation="h",title="תאונות לפי מחוז (צמתים בלבד)",
                   color="כמות",color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True, key="pc8")
    with e4:
        acc_type=df_junc["סוג_תאונה"].value_counts().reset_index(); acc_type.columns=["סוג","כמות"]
        fig=px.bar(acc_type,x="סוג",y="כמות",title="סוגי תאונות בצמתים",
                   color="כמות",color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False,xaxis_tickangle=-20)
        st.plotly_chart(fig,use_container_width=True, key="pc9")

    st.markdown("---")

    # ── ניתוח עומסי תנועה ────────────────────────────────────────────────────
    st.subheader("🚦 ניתוח עומסי תנועה בצמתים")
    st.caption("מבוסס על תאונות עורפיות (מדד פקק) ותאונות בשעות עומס (07-09, 16-19)")

    # גרף 1: עומס vs לא עומס לפי יום בשבוע
    f1,f2 = st.columns(2)
    with f1:
        peak_by_hour = (
            df_junc.groupby(["שעה_מספר","בשעת_עומס"])
            .size().reset_index(name="כמות")
        )
        peak_by_hour["סוג_שעה"] = peak_by_hour["בשעת_עומס"].map(
            {True:"🚦 שעת עומס", False:"🕐 שעה רגילה"})
        fig = px.bar(
            peak_by_hour, x="שעה_מספר", y="כמות", color="סוג_שעה",
            color_discrete_map={"🚦 שעת עומס":"#e74c3c","🕐 שעה רגילה":"#95a5a6"},
            title="תאונות בצמתים — שעות עומס מול רגילות",
            labels={"שעה_מספר":"שעה","כמות":"תאונות"},
            barmode="stack"
        )
        fig.update_layout(legend_title="סוג שעה")
        st.plotly_chart(fig, use_container_width=True, key="pc10")

    with f2:
        # תאונות עורפיות (פקק) לפי מחוז — ביחס לסה"כ
        rear_by_dist = df_junc.groupby("מחוז").apply(
            lambda g: pd.Series({
                "עורפיות (פקק)":  (g["סוג_תאונה"]=="אחורית").sum(),
                "אחרות": (g["סוג_תאונה"]!="אחורית").sum(),
            })
        ).reset_index()
        rear_melt = rear_by_dist.melt(id_vars="מחוז", var_name="סוג", value_name="כמות")
        fig = px.bar(
            rear_melt, x="מחוז", y="כמות", color="סוג",
            color_discrete_map={"עורפיות (פקק)":"#e74c3c","אחרות":"#bdc3c7"},
            title="תאונות עורפיות (מדד פקק) לפי מחוז",
            barmode="stack"
        )
        fig.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True, key="pc11")

    # גרף 2: פיזור מדד עומס + ציון תאונות לפי צומת (scatter)
    f3,f4 = st.columns(2)
    with f3:
        scatter_data = sites_flt.dropna(subset=["מדד_עומס","ציון_תאונות"])
        fig = px.scatter(
            scatter_data,
            x="ציון_תאונות", y="מדד_עומס",
            color="דירוג",
            color_discrete_map={"🔴 גבוה":"red","🟡 בינוני":"orange","🟢 נמוך":"green"},
            size="תאונות", size_max=20,
            hover_name="אתר",
            title="ציון תאונות מול מדד עומס — כל צומת",
            labels={"ציון_תאונות":"ציון תאונות (0-100)","מדד_עומס":"מדד עומס (0-100)"}
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                      annotation_text="סף עומס גבוה")
        fig.add_vline(x=50, line_dash="dot", line_color="gray",
                      annotation_text="סף סיכון גבוה")
        st.plotly_chart(fig, use_container_width=True, key="pc12")

    with f4:
        # TOP 10 צמתים לפי מדד עומס
        top_cong = sites_flt.nlargest(10,"מדד_עומס")[["אתר","מדד_עומס","ציון_תאונות","ציון_משולב"]]
        fig = px.bar(
            top_cong, x="מדד_עומס", y="אתר", orientation="h",
            color="מדד_עומס", color_continuous_scale="RdYlGn_r",
            title="10 הצמתים עם עומס התנועה הגבוה ביותר",
            labels={"מדד_עומס":"מדד עומס","אתר":"צומת"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, key="pc13")

st.markdown("---")
st.markdown("## 🔮 M2 — חיזוי (Predictive)")
st.caption("מה יקרה? חיזוי סיכון לצומת חדש + מודל ML")
st.subheader("🏗️ חיזוי סיכון לתשתית חדשה בתכנון")
st.caption("חזה סיכון לצומת חדש לפני שנבנה, או העלה קובץ תכנון לבדיקה")

mode = st.radio("בחר מצב:",["📝 הזנה ידנית","📁 העלאת קובץ תכנון"],horizontal=True)

plan_model,plan_nn,plan_sites = build_planning_model()

if mode=="📝 הזנה ידנית":
    st.subheader("הזן פרטי הצומת המתוכנן")
    st.info("💡 ירושלים ≈ 31.78, 35.22 | תל אביב ≈ 32.07, 34.78 | חיפה ≈ 32.82, 34.99")

    pc1,pc2 = st.columns(2)
    with pc1:
        plan_lat  = st.number_input("קו רוחב (Latitude):",29.5,33.5,31.78,0.001,format="%.4f",key="t3_lat")
        plan_road = st.selectbox("סוג דרך מתוכנן:",list(_ROAD_ENC.keys()),key="t3_road")
        plan_app  = st.slider("מספר כניסות לצומת:",2,6,4,key="t3_app")
    with pc2:
        plan_lon   = st.number_input("קו אורך (Longitude):",34.0,36.5,35.21,0.001,format="%.4f",key="t3_lon")
        plan_speed = st.select_slider('מהירות מותרת (קמ"ש):',
                       options=[30,40,50,60,70,80,90,100,110],value=50,key="t3_speed")
        plan_traf  = st.select_slider("נפח תנועה יומי:",
                       options=[1000,3000,5000,10000,20000,40000,80000],value=10000,key="t3_traf")
    plan_ctrl = st.selectbox("בקרה מתוכננת:",
                  ["ללא בקרה","תמרורי עצור / כניעה","רמזור קיים","כיכר קיימת"],key="t3_ctrl")

    if st.button("🔮 חזה סיכון",type="primary",use_container_width=True):
        hs  = 1 if plan_speed>=70 else 0
        X_n = np.array([[plan_lat,plan_lon,_ROAD_ENC.get(plan_road,0),hs]])
        pred= float(plan_model.predict(X_n)[0])
        pred= round(min(max(pred,0),100),1)
        rlvl= "🔴 גבוה" if pred>=60 else("🟡 בינוני" if pred>=30 else "🟢 נמוך")

        dist_,idx_ = plan_nn.kneighbors([[plan_lat,plan_lon]])
        near = plan_sites.iloc[idx_[0]].copy()
        near["מרחק (ק\"מ)"] = (dist_[0]*111).round(2)

        syn = {
            "total":          int(near["תאונות"].mean()),
            "fatal_pct":      float(near["fatal_pct"].mean()),
            "serious_pct":    float(near["serious_pct"].mean()),
            "night_pct":      float(near["night_pct"].mean()),
            "rain_pct":       float(near["rain_pct"].mean()),
            "pedestrian_pct": float(near["pedestrian_pct"].mean()),
            "frontal_pct":    float(near["frontal_pct"].mean()),
            "rear_pct":       float(near["rear_pct"].mean()),
            "rollover_pct":   float(near["rollover_pct"].mean()),
            "peak_pct":       float(near["peak_pct"].mean()),
            "high_speed":     hs,
        }
        ranked_p = rank_interventions(syn,plan_app,plan_traf,plan_ctrl)
        top_p    = ranked_p[0]

        r1,r2,r3 = st.columns(3)
        r1.metric("ציון סיכון חזוי",f"{pred:.1f}/100")
        r2.metric("רמת סיכון",rlvl)
        r3.metric("התערבות מומלצת",top_p["פתרון"])

        st.success(
            f"**{top_p['פתרון']}** — {top_p['תיאור']}  \n"
            f"עלות ממוצעת: **{top_p['עלות ממוצעת']}** | "
            f"הפחתת קטלניות: **{top_p['הפחתת קטלניות']}** | "
            f"ROI 5 שנים: **{top_p['ROI 5 שנים']}**"
        )

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",value=pred,
            title={"text":"ציון סיכון חזוי"},number={"suffix":" / 100"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#e74c3c" if pred>=60 else("#e67e22" if pred>=30 else "#2ecc71")},
                   "steps":[{"range":[0,30],"color":"#d5f5e3"},
                            {"range":[30,60],"color":"#fdebd0"},
                            {"range":[60,100],"color":"#fadbd8"}]}))
        fig_g.update_layout(height=260)
        st.plotly_chart(fig_g,use_container_width=True, key="pc19")

        st.markdown("#### אתרים דומים קרובים — בסיס החיזוי")
        st.dataframe(near[['אתר','מרחק (ק"מ)','ציון_משולב','דירוג','התערבות_מומלצת','תאונות']
                          ].reset_index(drop=True),use_container_width=True,hide_index=True)

        # מפה
        viz = near[["lat","lon","אתר","ציון_משולב","תאונות","דירוג"]].copy()
        viz["סוג"]="אתר ידוע"
        pred_row=pd.DataFrame([{"lat":plan_lat,"lon":plan_lon,"אתר":"📍 צומת מתוכנן",
                                 "ציון_משולב":pred,"תאונות":8,"דירוג":rlvl,"סוג":"צומת מתוכנן"}])
        viz_all=pd.concat([viz,pred_row],ignore_index=True)
        fig_m=px.scatter_map(viz_all,lat="lat",lon="lon",color="סוג",
            color_discrete_map={"אתר ידוע":"orange","צומת מתוכנן":"blue"},
            size="תאונות",size_max=20,hover_name="אתר",
            hover_data={"ציון_משולב":True,"דירוג":True,"lat":False,"lon":False,"תאונות":False,"סוג":False},
            zoom=10,center={"lat":plan_lat,"lon":plan_lon},
            map_style="open-street-map",height=400,
            title="צומת מתוכנן (כחול) לעומת אתרים קיימים (כתום)")
        st.plotly_chart(fig_m,use_container_width=True, key="pc20")

else:  # העלאת קובץ
    st.subheader("📁 העלאת קובץ תכנון")
    st.info(
        "**פורמט נדרש (CSV/Excel):**  \n"
        "עמודות חובה: `שם_צומת, קו_רוחב, קו_אורך, סוג_דרך, מהירות_מותרת, מספר_כניסות`  \n"
        "עמודה אופציונלית: `פתרון_מוצע` — המערכת תשווה להמלצה ותאשר/תדחה"
    )

    # הורדת תבנית
    template = pd.DataFrame([
        {"שם_צומת":"צומת לדוגמה 1","קו_רוחב":32.07,"קו_אורך":34.78,
         "סוג_דרך":"עירוני - בצומת","מהירות_מותרת":50,"מספר_כניסות":4,"פתרון_מוצע":"כיכר תנועה"},
        {"שם_צומת":"צומת לדוגמה 2","קו_רוחב":31.78,"קו_אורך":35.22,
         "סוג_דרך":"בין-עירוני - בצומת","מהירות_מותרת":80,"מספר_כניסות":3,"פתרון_מוצע":""},
    ])
    st.download_button("⬇️ הורד תבנית CSV",
                       template.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
                       "template_junctions.csv","text/csv")

    uploaded = st.file_uploader("העלה קובץ תכנון:",type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                up_df = pd.read_excel(uploaded)
            else:
                up_df = pd.read_csv(uploaded,encoding="utf-8-sig")

            required = {"שם_צומת","קו_רוחב","קו_אורך","סוג_דרך","מהירות_מותרת","מספר_כניסות"}
            if not required.issubset(set(up_df.columns)):
                st.error(f"חסרות עמודות: {required - set(up_df.columns)}")
            else:
                st.success(f"✅ נטען בהצלחה — {len(up_df)} צמתים")
                results_list=[]
                for _,row in up_df.iterrows():
                    hs=1 if row["מהירות_מותרת"]>=70 else 0
                    X_r=np.array([[row["קו_רוחב"],row["קו_אורך"],_ROAD_ENC.get(str(row["סוג_דרך"]),0),hs]])
                    pred_r=float(plan_model.predict(X_r)[0])
                    pred_r=round(min(max(pred_r,0),100),1)
                    rlvl_r="🔴 גבוה" if pred_r>=60 else("🟡 בינוני" if pred_r>=30 else "🟢 נמוך")

                    dist_r,idx_r=plan_nn.kneighbors([[row["קו_רוחב"],row["קו_אורך"]]])
                    near_r=plan_sites.iloc[idx_r[0]]
                    syn_r={
                        "total":int(near_r["תאונות"].mean()),
                        "fatal_pct":float(near_r["fatal_pct"].mean()),
                        "serious_pct":float(near_r["serious_pct"].mean()),
                        "night_pct":float(near_r["night_pct"].mean()),
                        "rain_pct":float(near_r["rain_pct"].mean()),
                        "pedestrian_pct":float(near_r["pedestrian_pct"].mean()),
                        "frontal_pct":float(near_r["frontal_pct"].mean()),
                        "rear_pct":float(near_r["rear_pct"].mean()),
                        "rollover_pct":float(near_r["rollover_pct"].mean()),
                        "peak_pct":float(near_r["peak_pct"].mean()),
                        "high_speed":hs,
                    }
                    ranked_r=rank_interventions(syn_r,int(row["מספר_כניסות"]))
                    rec_r   =ranked_r[0]["פתרון"]
                    prop    =str(row.get("פתרון_מוצע","")).strip()

                    # ── חישוב יעילות הפתרון המוצע מול המומלץ ──────────────
                    def _eff(name):
                        iv=INTERVENTIONS.get(name,{})
                        n_=syn_r["total"]
                        fatal_save=syn_r["fatal_pct"]*n_*ACCIDENT_COST["קטלנית"]*iv.get("fatal_r",0)
                        ser_save  =syn_r["serious_pct"]*n_*ACCIDENT_COST["קשה"]*iv.get("serious_r",0)
                        annual_save=fatal_save+ser_save
                        cong_reduce=iv.get("cong_r",0)*100
                        cost_=iv.get("cost_avg",1)
                        roi5=round(annual_save*5/cost_*100,1) if cost_>0 else 0
                        return annual_save,cong_reduce,roi5,iv.get("cost_avg",0)

                    rec_save,rec_cong,rec_roi,rec_cost=_eff(rec_r)
                    if prop and prop in INTERVENTIONS:
                        prop_save,prop_cong,prop_roi,prop_cost=_eff(prop)
                        if prop==rec_r:
                            verdict="✅ מאושר — פתרון אופטימלי"
                            eff_score=100
                        else:
                            eff_score=round(prop_roi/rec_roi*100) if rec_roi>0 else 0
                            if eff_score>=80:
                                verdict=f"✅ מקובל ({eff_score}% מהאופטימלי)"
                            elif eff_score>=50:
                                verdict=f"⚠️ פחות יעיל ({eff_score}%) — מומלץ: {rec_r}"
                            else:
                                verdict=f"❌ לא יעיל ({eff_score}%) — החלף ל: {rec_r}"
                    else:
                        prop_save,prop_cong,prop_roi,prop_cost=None,None,None,None
                        verdict="— לא הוגדר פתרון"
                        eff_score=None

                    results_list.append({
                        "שם צומת":                  row["שם_צומת"],
                        "ציון סיכון נוכחי":         pred_r,
                        "רמת סיכון":                 rlvl_r,
                        "פתרון מוצע":               prop if prop else "—",
                        "המלצת המערכת":             rec_r,
                        "הערכה":                    verdict,
                        "יעילות %":                 eff_score if eff_score is not None else "—",
                        "חיסכון תאונות/שנה (מוצע)": f"₪{prop_save:,.0f}" if prop_save else "—",
                        "הפחתת עומס (מוצע)":        f"{prop_cong:.0f}%" if prop_cong else "—",
                        "ROI מוצע 5 שנים":          f"{prop_roi:.0f}%" if prop_roi else "—",
                        "חיסכון תאונות/שנה (מומלץ)":f"₪{rec_save:,.0f}",
                        "הפחתת עומס (מומלץ)":       f"{rec_cong:.0f}%",
                        "ROI מומלץ 5 שנים":         f"{rec_roi:.0f}%",
                        "עלות מומלץ":               f"₪{rec_cost:,}",
                    })

                res_df=pd.DataFrame(results_list)

                # ── ציון כללי לתכנון ───────────────────────────────────────
                approved  = res_df["הערכה"].str.startswith("✅").sum()
                warning   = res_df["הערכה"].str.startswith("⚠️").sum()
                rejected  = res_df["הערכה"].str.startswith("❌").sum()
                sc1,sc2,sc3,sc4=st.columns(4)
                sc1.metric("✅ מאושר",   f"{approved}/{len(res_df)}")
                sc2.metric("⚠️ לשיפור",  f"{warning}/{len(res_df)}")
                sc3.metric("❌ לדחייה",  f"{rejected}/{len(res_df)}")
                avg_eff=[x for x in res_df["יעילות %"] if isinstance(x,(int,float))]
                sc4.metric("יעילות ממוצעת",f"{sum(avg_eff)/len(avg_eff):.0f}%" if avg_eff else "—")

                st.dataframe(res_df,use_container_width=True,hide_index=True,
                    column_config={"ציון סיכון נוכחי":st.column_config.ProgressColumn(
                        "ציון סיכון נוכחי",min_value=0,max_value=100,format="%.1f")})

                # ── מפה ──────────────────────────────────────────────────
                up_with_res=up_df.copy()
                up_with_res["הערכה"]=res_df["הערכה"].values
                up_with_res["צבע"]=up_with_res["הערכה"].apply(
                    lambda v: "green" if str(v).startswith("✅") else
                              ("orange" if str(v).startswith("⚠️") else "red"))
                fig_up=px.scatter_map(up_with_res,lat="קו_רוחב",lon="קו_אורך",
                    color="צבע",
                    color_discrete_map={"green":"green","orange":"orange","red":"red"},
                    hover_name="שם_צומת",
                    hover_data={"הערכה":True,"צבע":False,"קו_רוחב":False,"קו_אורך":False},
                    zoom=7,center={"lat":31.8,"lon":35.0},
                    map_style="open-street-map",height=420,
                    title="🟢 מאושר | 🟠 לשיפור | 🔴 לדחייה")
                st.plotly_chart(fig_up,use_container_width=True, key="pc21")

                out=res_df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("⬇️ ייצוא דוח הערכה",out,"junction_evaluation.csv","text/csv")
        except Exception as e:
            st.error(f"שגיאה בטעינת הקובץ: {e}")

st.markdown("---")
st.subheader("🤖 מודל ML — מטריקות וחיזוי אינטראקטיבי")
st.caption('Random Forest Classifier | נתוני למ"ס PUF 2021 | 7 סוגי התערבות')

clf,enc = load_ml_model()
metrics = load_metrics()

if clf is None:
    st.warning("⚠️ מודל לא נמצא. הרץ: `python train_model.py`")
else:
    if metrics:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")
        m3.metric("Precision",f"{metrics['precision']*100:.1f}%")
        m4.metric("Recall",   f"{metrics['recall']*100:.1f}%")

        mc1,mc2=st.columns(2)
        with mc1:
            cm=np.array(metrics["confusion_matrix"])
            lbls=metrics["labels"]
            fig_cm=px.imshow(cm,x=lbls,y=lbls,text_auto=True,
                             color_continuous_scale="Blues",
                             labels={"x":"חיזוי","y":"אמיתי"},title="Confusion Matrix")
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm,use_container_width=True, key="pc24")
        with mc2:
            fi=metrics["feature_importance"]
            fi_df=pd.DataFrame(sorted(fi.items(),key=lambda x:x[1],reverse=True),
                               columns=["פיצ'ר","חשיבות"])
            fig_fi=px.bar(fi_df,x="חשיבות",y="פיצ'ר",orientation="h",
                          color="חשיבות",color_continuous_scale="Reds",
                          title="Feature Importance")
            fig_fi.update_layout(yaxis=dict(autorange="reversed"),
                                 coloraxis_showscale=False,height=380)
            st.plotly_chart(fig_fi,use_container_width=True, key="pc25")

    st.markdown("---")
    st.subheader("🔮 חיזוי אינטראקטיבי")
    _FEAT = ["תאונות","fatal_pct","serious_pct","night_pct","rain_pct",
             "pedestrian_pct","frontal_pct","rear_pct","rollover_pct","high_speed","peak_pct"]
    c1,c2,c3=st.columns(3)
    with c1:
        v_acc  =st.number_input("תאונות היסטוריות:",1,500,15,key="t5_acc")
        v_fatal=st.slider("% קטלניות:",0,100,10,key="t5_fatal")/100
        v_ser  =st.slider("% קשות:",0,100,25,key="t5_ser")/100
        v_night=st.slider("% לילה:",0,100,30,key="t5_night")/100
    with c2:
        v_rain =st.slider("% גשם:",0,100,15,key="t5_rain")/100
        v_ped  =st.slider("% הולכי רגל:",0,100,10,key="t5_ped")/100
        v_front=st.slider("% חזיתיות:",0,100,20,key="t5_front")/100
    with c3:
        v_rear =st.slider("% אחוריות:",0,100,15,key="t5_rear")/100
        v_roll =st.slider("% התהפכות:",0,100,5,key="t5_roll")/100
        v_peak =st.slider("% שעות עומס (07-09, 16-19):",0,100,35,key="t5_peak")/100
        v_hs   =st.checkbox('מהירות ≥ 70 קמ"ש',value=False,key="t5_hs")

    if st.button("🚀 חזה",type="primary",use_container_width=True):
        X_in=np.array([[v_acc,v_fatal,v_ser,v_night,v_rain,v_ped,v_front,v_rear,v_roll,1 if v_hs else 0,v_peak]])
        pred_cls=clf.predict(X_in)[0]
        probs=clf.predict_proba(X_in)[0]
        classes=list(clf.classes_)
        info_p=INTERVENTIONS.get(pred_cls,{})
        st.success(
            f"**התערבות מומלצת: {pred_cls}**  \n"
            f"{info_p.get('desc','')}  \n"
            f"עלות ממוצעת: **₪{info_p.get('cost_avg',0):,}** | "
            f"מקור: {info_p.get('cost_src','')}"
        )
        prob_df=pd.DataFrame({"התערבות":classes,"הסתברות":probs,
                              "אחוז":[f"{p*100:.1f}%" for p in probs]})
        fig_prob=px.bar(prob_df.sort_values("הסתברות",ascending=False),
                        x="התערבות",y="הסתברות",color="הסתברות",
                        color_continuous_scale="Blues",text="אחוז",
                        title="הסתברות לכל התערבות")
        fig_prob.update_layout(yaxis_tickformat=".0%",coloraxis_showscale=False,
                               xaxis_tickangle=-20)
        fig_prob.update_traces(textposition="outside")
        st.plotly_chart(fig_prob,use_container_width=True, key="pc26")

st.markdown("---")
st.caption('⚠️ המודל מאומן על נתוני למ"ס PUF 2021. Labels נוצרו מלוגיקה עסקית.')
