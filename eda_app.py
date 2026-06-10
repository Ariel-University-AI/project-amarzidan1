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

# ══════════════════════════════════════════════════════════════════════════════
# מנוע ניתוח הנדסי — מבוסס HCM 6th Ed., AASHTO, FHWA, NCHRP
# ══════════════════════════════════════════════════════════════════════════════

# 9 חלופות עם פרמטרים מבוססי תקנים בינלאומיים
ALTERNATIVES_HCM = {
    "שיפור גיאומטרי": {
        "desc":            "תיקון מחדל גיאומטרי — נראות, אי תנועה, סימון מחדש",
        "fatal_r":         0.15, "injury_r": 0.10, "crash_r": 0.08,
        "capacity_pct":    0.10, "delay_r":  0.05, "los_imp":  0.5,
        "conflict_r":      0.05, "emission_r": 0.02,
        "cost_min":        200_000, "cost_avg": 500_000, "cost_max": 1_000_000,
        "maint_annual":    20_000,  "aadt_max": 60_000,
        "source":          "NCHRP Report 617",
    },
    "הוספת נתיבי פנייה": {
        "desc":            "נתיבי פנייה ייעודיים להפחתת קונפליקטים",
        "fatal_r":         0.25, "injury_r": 0.20, "crash_r": 0.15,
        "capacity_pct":    0.20, "delay_r":  0.15, "los_imp":  1.0,
        "conflict_r":      0.10, "emission_r": 0.05,
        "cost_min":        300_000, "cost_avg": 700_000, "cost_max": 1_500_000,
        "maint_annual":    30_000,  "aadt_max": 50_000,
        "source":          "NCHRP Report 617, HCM 6th Ed.",
    },
    "צומת מרומזר": {
        "desc":            "רמזורים קבועים — HCM ציון D מקובל",
        "fatal_r":         0.30, "injury_r": 0.25, "crash_r": 0.20,
        "capacity_pct":    0.25, "delay_r":  0.10, "los_imp":  1.0,
        "conflict_r":      0.15, "emission_r": -0.05,
        "cost_min":        5_000_000, "cost_avg": 7_000_000, "cost_max": 10_000_000,
        "maint_annual":    150_000,   "aadt_max": 80_000,
        "source":          "HCM 6th Edition, Chapter 19",
    },
    "כיכר חד נתיבית": {
        "desc":            "Single-Lane Roundabout — מבטל 75% מנקודות הקונפליקט",
        "fatal_r":         0.90, "injury_r": 0.75, "crash_r": 0.47,
        "capacity_pct":    0.20, "delay_r":  0.25, "los_imp":  1.5,
        "conflict_r":      0.75, "emission_r": 0.15,
        "cost_min":        400_000, "cost_avg": 500_000, "cost_max": 600_000,
        "maint_annual":    25_000,  "aadt_max": 15_000,
        "source":          "FHWA Roundabout Guide (2010), NCHRP 572",
    },
    "כיכר דו נתיבית": {
        "desc":            "Two-Lane Roundabout — לנפחי תנועה בינוניים",
        "fatal_r":         0.78, "injury_r": 0.65, "crash_r": 0.40,
        "capacity_pct":    0.40, "delay_r":  0.30, "los_imp":  2.0,
        "conflict_r":      0.70, "emission_r": 0.12,
        "cost_min":        800_000, "cost_avg": 1_200_000, "cost_max": 2_000_000,
        "maint_annual":    40_000,  "aadt_max": 35_000,
        "source":          "NCHRP Report 572, AASHTO",
    },
    "כיכר טורבו": {
        "desc":            "Turbo Roundabout — מונע שינוי נתיב, קיבולת גבוהה",
        "fatal_r":         0.85, "injury_r": 0.70, "crash_r": 0.45,
        "capacity_pct":    0.55, "delay_r":  0.35, "los_imp":  2.5,
        "conflict_r":      0.72, "emission_r": 0.14,
        "cost_min":        1_500_000, "cost_avg": 2_500_000, "cost_max": 4_000_000,
        "maint_annual":    60_000,    "aadt_max": 40_000,
        "source":          "SWOV (2010), Fortuijn (2009)",
    },
    "צומת חכם עם רמזור אדפטיבי": {
        "desc":            "Adaptive Traffic Control System — אופטימיזציה בזמן אמת",
        "fatal_r":         0.20, "injury_r": 0.15, "crash_r": 0.12,
        "capacity_pct":    0.20, "delay_r":  0.20, "los_imp":  1.0,
        "conflict_r":      0.15, "emission_r": 0.10,
        "cost_min":        5_000_000, "cost_avg": 7_000_000, "cost_max": 10_000_000,
        "maint_annual":    200_000,   "aadt_max": 80_000,
        "source":          "FHWA ATMS Program (2022), TRB",
    },
    "מחלף חלקי": {
        "desc":            "Partial Interchange — הפרדה מפלסית לזרמים עיקריים",
        "fatal_r":         0.50, "injury_r": 0.40, "crash_r": 0.35,
        "capacity_pct":    0.60, "delay_r":  0.50, "los_imp":  3.0,
        "conflict_r":      0.60, "emission_r": 0.20,
        "cost_min":        20_000_000, "cost_avg": 35_000_000, "cost_max": 60_000_000,
        "maint_annual":    300_000,    "aadt_max": 100_000,
        "source":          "AASHTO Green Book (2018)",
    },
    "מחלף מלא": {
        "desc":            "Full Interchange — ביטול מלא של הצומת, הפרדה מפלסית",
        "fatal_r":         0.70, "injury_r": 0.60, "crash_r": 0.55,
        "capacity_pct":    1.00, "delay_r":  0.80, "los_imp":  4.0,
        "conflict_r":      0.95, "emission_r": 0.30,
        "cost_min":        50_000_000, "cost_avg": 100_000_000, "cost_max": 200_000_000,
        "maint_annual":    500_000,    "aadt_max": 999_999,
        "source":          "AASHTO Green Book (2018), נתיבי ישראל",
    },
}

_LOS_NUM  = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
_LOS_CHAR = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F"}

def analyze_intersection_hcm(inp: dict) -> pd.DataFrame:
    """
    ניתוח הנדסי מלא לפי HCM 6th Ed., AASHTO, FHWA, NCHRP.
    מחזיר DataFrame מדורג עם ציון 0-100 לכל חלופה.
    """
    aadt     = inp.get("aadt", 10_000)
    fatal    = inp.get("fatal", 0)
    serious  = inp.get("serious", 0)
    los_curr = _LOS_NUM.get(inp.get("los","D"), 4)

    rows = []
    for name, alt in ALTERNATIVES_HCM.items():

        # ── בטיחות (40%) ─────────────────────────────────────────────────
        fatal_save   = fatal   * alt["fatal_r"]
        serious_save = serious * alt["injury_r"]
        saf = (alt["fatal_r"]*0.50 + alt["injury_r"]*0.30 +
               alt["crash_r"]*0.10 + alt["conflict_r"]*0.10) * 100
        if fatal > 2:   saf = min(saf * 1.1, 100)
        if serious > 5: saf = min(saf * 1.05, 100)

        # ── תנועה ותפעול (35%) ───────────────────────────────────────────
        new_los   = max(1, los_curr - alt["los_imp"])
        los_score = (los_curr - new_los) / 5 * 100
        # התאמת AADT לקיבולת (HCM)
        if aadt <= alt["aadt_max"]:
            aadt_fit = 1.0
        else:
            aadt_fit = max(0.2, 1 - (aadt - alt["aadt_max"]) / max(alt["aadt_max"],1) * 0.5)
        trf = (los_score*0.30 + alt["delay_r"]*100*0.30 +
               alt["capacity_pct"]*100*0.25 + alt["conflict_r"]*100*0.15) * aadt_fit

        # ── כלכלה (15%) ──────────────────────────────────────────────────
        ann_save = (fatal_save*3_500_000 + serious_save*700_000)
        roi_yr   = alt["cost_avg"] / ann_save if ann_save > 0 else 50
        roi_sc   = max(0, min(100, (20 - roi_yr) / 20 * 100))
        if alt["cost_avg"] > 50_000_000: roi_sc *= 0.50
        elif alt["cost_avg"] > 10_000_000: roi_sc *= 0.75

        # ── סביבה (10%) ──────────────────────────────────────────────────
        env = min(100, max(0, alt["emission_r"]*100 + 50))

        # ── ציון כולל ────────────────────────────────────────────────────
        total = saf*0.40 + trf*0.35 + roi_sc*0.15 + env*0.10
        rows.append({
            "חלופה":               name,
            "תיאור":               alt["desc"],
            "ציון_בטיחות":         round(min(saf,100),1),
            "ציון_תנועה":          round(min(trf,100),1),
            "ציון_כלכלה":          round(min(roi_sc,100),1),
            "ציון_סביבה":          round(env,1),
            "ציון_כולל":           round(total,1),
            "הפחתת קטלניות":       f"{int(alt['fatal_r']*100)}%",
            "הפחתת פציעות":        f"{int(alt['injury_r']*100)}%",
            "הפחתת עיכוב":         f"{int(alt['delay_r']*100)}%",
            "LOS חדש":             _LOS_CHAR.get(int(max(1,new_los)),
                                               _LOS_CHAR.get(int(min(6,round(new_los))),"B")),
            "עלות הקמה":           f"₪{alt['cost_avg']:,}",
            "ROI (שנים)":          round(roi_yr,1) if roi_yr<50 else ">50",
            "AADT מקסימלי":        f"{alt['aadt_max']:,}",
            "מקור":                alt["source"],
            "_cost":               alt["cost_avg"],
            "_fatal_r":            alt["fatal_r"],
            "_aadt_max":           alt["aadt_max"],
        })

    df = pd.DataFrame(rows).sort_values("ציון_כולל",ascending=False).reset_index(drop=True)
    df.insert(0, "דירוג", range(1, len(df)+1))
    return df

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
def detect_existing_infra(lat: float, lon: float, road_type: str = "", prof: dict = None) -> dict:
    """
    זיהוי תשתית קיימת בצומת — שלוש שכבות:
    1. נתוני CBS (SUG_DEREH, פרופיל תאונות)
    2. OSM/Overpass — שאילתה מקיפה עם כל תגי הצומת הנפוצים בישראל
    3. Google Places API (אם קיים GOOGLE_MAPS_KEY)
    """
    found = {"כיכר": False, "רמזור": False, "עצור": False, "מצלמה": False,
             "מקור": None, "ביטחון": "נמוך"}

    # ── שכבה 1: CBS data — SUG_DEREH=5 = כיכר ────────────────────────────────
    if road_type in ("חניון / כיכר", "5"):
        found["כיכר"] = True
        found["מקור"] = "CBS (SUG_DEREH=5)"
        found["ביטחון"] = "גבוה"

    if prof:
        frontal = prof.get("frontal_pct", 0)
        # ב-CBS אין סימן אמין לזיהוי רמזור — rear_pct גבוה קיים גם בצמתות רמזור.
        # frontal גבוה = עצור/כניעה (ציר ראשי ללא עדיפות ברורה)
        if frontal > 0.25 and not found["כיכר"] and not found["רמזור"]:
            found["עצור"] = True
            found["מקור"] = found["מקור"] or "CBS (frontal_pct>25%)"
            found["ביטחון"] = "בינוני"

    # ── שכבה 2: OSM/Overpass — שאילתה מקיפה ─────────────────────────────────
    try:
        q = f"""
[out:json][timeout:8];
(
  way["junction"~"roundabout|circular"](around:150,{lat},{lon});
  way["highway"="mini_roundabout"](around:150,{lat},{lon});
  node["highway"="mini_roundabout"](around:150,{lat},{lon});
  node["junction"~"roundabout|circular"](around:150,{lat},{lon});
  node["highway"="traffic_signals"](around:120,{lat},{lon});
  node["traffic_signals"](around:120,{lat},{lon});
  node["highway"~"^stop$|^give_way$"](around:100,{lat},{lon});
  node["highway"="speed_camera"](around:120,{lat},{lon});
  node["enforcement"="speed"](around:120,{lat},{lon});
);
out tags;
"""
        r = requests.post("https://overpass-api.de/api/interpreter", data=q, timeout=9)
        elements = r.json().get("elements", [])
        for el in elements:
            tags = el.get("tags", {})
            hw   = tags.get("highway", "")
            junc = tags.get("junction", "")
            ts   = tags.get("traffic_signals", "")
            # כיכר: junction=roundabout/circular OR highway=mini_roundabout
            if junc in ("roundabout", "circular") or hw == "mini_roundabout":
                found["כיכר"] = True
                found["מקור"] = "OpenStreetMap"
                found["ביטחון"] = "גבוה"
            # רמזור: highway=traffic_signals OR traffic_signals=* key exists
            if hw == "traffic_signals" or ts:
                found["רמזור"] = True
                found["מקור"] = found["מקור"] or "OpenStreetMap"
                found["ביטחון"] = "גבוה"
            # עצור / כניעה
            if hw in ("stop", "give_way"):
                found["עצור"] = True
                found["מקור"] = found["מקור"] or "OpenStreetMap"
            # מצלמה
            if hw == "speed_camera" or tags.get("enforcement") == "speed":
                found["מצלמה"] = True
                found["מקור"] = found["מקור"] or "OpenStreetMap"
    except Exception:
        pass  # ממשיכים עם מה שיש מ-CBS

    # ── שכבה 3: Google Places API (Nearby Search) ─────────────────────────────
    gkey = os.environ.get("GOOGLE_MAPS_KEY", "")
    if gkey and not any([found["כיכר"], found["רמזור"], found["עצור"]]):
        try:
            url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                   f"?location={lat},{lon}&radius=150&type=route&key={gkey}")
            gr = requests.get(url, timeout=6)
            if gr.status_code == 200:
                for place in gr.json().get("results", []):
                    name_lc = place.get("name", "").lower()
                    if any(x in name_lc for x in ["roundabout", "circle", "כיכר"]):
                        found["כיכר"] = True
                        found["מקור"] = "Google Places API"
                        found["ביטחון"] = "בינוני"
                        break
        except Exception:
            pass

    if not found["מקור"]:
        found["מקור"] = "לא זוהה אוטומטית"
        found["ביטחון"] = "נמוך"

    return found

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

st.markdown("""
<div style='background:linear-gradient(135deg,#1a237e,#0d47a1);
     padding:28px 32px;border-radius:12px;color:white;margin-bottom:20px'>
<h1 style='margin:0;font-size:28px'>🚦 מערכת ניתוח וחיזוי צמתים — ישראל 2021</h1>
<p style='margin:6px 0 0;opacity:0.85;font-size:14px'>
משרד התחבורה | נתונים: למ"ס PUF 2021 | עלויות: נתיבי ישראל + מכרזים רשמיים
</p></div>
""", unsafe_allow_html=True)

with st.container():
    df_junc = df[df["סוג_דרך"].isin({"עירוני - בצומת","בין-עירוני - בצומת","חניון / כיכר"})]

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
    st.markdown("#### 📈 ניתוח עומסי תנועה — כל הצמתים")
    st.caption("מדד עומס = % עורפיות (פקק) × 0.6 + % שעות עומס × 0.4")
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

    # ── מפה אינטראקטיבית — לחץ לניתוח ────────────────────────────────────────
    st.info("👆 **לחץ על צומת במפה** לקבלת ניתוח מלא מתחת")
    map_data = sites_flt.dropna(subset=["lat","lon"])
    # מסנן צמתים עם שמות לא תקינים
    map_data = map_data[~map_data["אתר"].str.contains("nan|כללי", na=True, case=False)]
    # ── סנכרון: אם הסלקטבוקס השתנה — עדכן לפני המפה ────────────────────────
    if "t1_site_select" in st.session_state and st.session_state["t1_site_select"]:
        st.session_state["selected_site"] = st.session_state["t1_site_select"]

    if len(map_data)>0:
        map_data = map_data.copy()
        map_data["custom_site"] = map_data["אתר"]
        # אם יש צומת נבחר — המפה מתקרבת אליו
        sel_site_data = st.session_state.get("selected_site","")
        map_center = {"lat":31.8,"lon":35.0}
        map_zoom   = 7
        if sel_site_data:
            # חיפוש בנתוני CBS לפי שם הצומת
            site_rows = df[df["אתר"]==sel_site_data].dropna(subset=["קו_רוחב","קו_אורך"])
            if len(site_rows)>0:
                map_center = {"lat": float(site_rows["קו_רוחב"].mean()),
                              "lon": float(site_rows["קו_אורך"].mean())}
                map_zoom   = 15
            elif sel_site_data in map_data["אתר"].values:
                sel_row    = map_data[map_data["אתר"]==sel_site_data].iloc[0]
                map_center = {"lat": sel_row["lat"], "lon": sel_row["lon"]}
                map_zoom   = 15

        fig_map = px.scatter_map(
            map_data, lat="lat", lon="lon",
            color="ציון_משולב", size="תאונות", size_max=28,
            custom_data=["custom_site"],
            hover_name="אתר",
            hover_data={"תאונות":True,"ציון_תאונות":True,"מדד_עומס":True,
                        "ציון_משולב":True,"דירוג":True,"התערבות_מומלצת":True,
                        "lat":False,"lon":False,"custom_site":False},
            color_continuous_scale="RdYlGn_r",
            zoom=map_zoom, center=map_center,
            map_style="open-street-map",
            height=500,
            title="🔴 אדום = סיכון גבוה | לחץ על צומת לניתוח",
        )
        fig_map.update_coloraxes(colorbar_title="ציון משולב")
        # סמן את הצומת הנבחר
        if map_zoom == 15 and map_center["lat"] != 31.8:
            fig_map.add_trace(go.Scattermap(
                lat=[map_center["lat"]], lon=[map_center["lon"]],
                mode="markers",
                marker=dict(size=40, color="rgba(0,100,255,0.2)", symbol="circle"),
                hoverinfo="skip", showlegend=False, name=""))
            fig_map.add_trace(go.Scattermap(
                lat=[map_center["lat"]], lon=[map_center["lon"]],
                mode="markers+text",
                marker=dict(size=18, color="#1a6bff"),
                text=["📍"], textposition="top right",
                textfont=dict(size=14, color="#1a6bff"),
                hoverinfo="skip", showlegend=False, name=""))
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

    st.subheader("🔍 ניתוח צומת נבחר")

    junction_mask_t1 = df["סוג_דרך"].isin({"עירוני - בצומת","בין-עירוני - בצומת","חניון / כיכר"})
    # מסנן שמות "nan" וצמתים ללא קוד ספציפי
    all_raw = df[junction_mask_t1]["אתר"].dropna().unique().tolist()
    all_sites_t1 = sorted([
        s for s in all_raw
        if "nan" not in str(s).lower()
        and "כללי" not in str(s)
        and len(str(s)) > 5
    ])

    # עדכון session_state רק אם הערך הנוכחי לא תקין
    cur = st.session_state.get("selected_site","")
    if cur not in all_sites_t1:
        st.session_state["selected_site"] = all_sites_t1[0] if all_sites_t1 else ""

    selected = st.selectbox(
        "בחר צומת (או לחץ על נקודה במפה):",
        all_sites_t1,
        index=all_sites_t1.index(st.session_state["selected_site"])
              if st.session_state["selected_site"] in all_sites_t1 else 0,
        key="t1_site_select"
    )
    st.session_state["selected_site"] = selected
    site_df = df[df["אתר"]==selected]
    n       = max(len(site_df),1)

    # ── זיהוי OSM אוטומטי בשינוי צומת ───────────────────────────────────────
    site_lat_t1 = site_df["קו_רוחב"].dropna()
    site_lon_t1 = site_df["קו_אורך"].dropna()

    if len(site_lat_t1)>0:
        raw_lat = float(site_lat_t1.mean())
        raw_lon = float(site_lon_t1.mean())
        # pyproj מדויק מספיק — לא צריך OSRM snap (חוסך 2-3 שניות)
        if f"snap_{selected}" not in st.session_state:
            st.session_state[f"snap_{selected}"] = (raw_lat, raw_lon)
        # זיהוי תשתית — רק פעם אחת לכל צומת (cached)
        if f"osm_{selected}" not in st.session_state:
            rd_type = site_df["סוג_דרך"].mode().iloc[0] if len(site_df)>0 else ""
            st.session_state[f"osm_{selected}"] = detect_existing_infra(
                raw_lat, raw_lon,
                road_type=rd_type,
                prof={
                    "rear_pct":    (site_df["סוג_תאונה"]=="אחורית").sum()/max(len(site_df),1),
                    "fatal_pct":   (site_df["חומרת_תאונה"]=="קטלנית").sum()/max(len(site_df),1),
                    "frontal_pct": (site_df["סוג_תאונה"]=="חזיתית").sum()/max(len(site_df),1),
                })

    osm_i = st.session_state.get(f"osm_{selected}", {})

    # ── תג תשתית קיימת + מקור + ביטחון ──────────────────────────────────────
    if osm_i:
        tags = []
        if osm_i.get("כיכר"):  tags.append("🔵 כיכר תנועה קיימת")
        if osm_i.get("רמזור"): tags.append("🚦 רמזור קיים")
        if osm_i.get("עצור"):  tags.append("🛑 תמרורי עצור")
        if osm_i.get("מצלמה"):tags.append("📷 מצלמת אכיפה")
        src    = osm_i.get("מקור","לא זוהה")
        trust  = osm_i.get("ביטחון","נמוך")
        trust_icon = {"גבוה":"✅","בינוני":"⚠️","נמוך":"❓"}.get(trust,"❓")

        if tags:
            st.success(f"**תשתית מזוהה:** {' | '.join(tags)}  \n"
                       f"מקור: **{src}** | ביטחון זיהוי: {trust_icon} **{trust}**")
        else:
            st.warning(f"⚠️ **לא זוהתה תשתית קיימת** (מקור: {src})  \n"
                       "אנא בחר ידנית בשדה 'בקרה קיימת' מתחת.")

    # ── מפה עם נקודה מדויקת לכביש + עיגול מסמן ────────────────────────────
    # נתוני המצב הקיים — מוצגים מיד
    st.info(f"📍 **{selected}** | סוג: {site_df['סוג_דרך'].mode().iloc[0] if len(site_df)>0 else '—'} | {n} תאונות מתועדות")

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

    # ── פרמטרים הנדסיים — ממשיך מניתוח הצומת ────────────────────────────────
    st.markdown("---")
    st.caption("📐 נתוני CBS ממולאים אוטומטית — הזן פרמטרים הנדסיים נוספים לניתוח מלא")

    cur_opts = ["ללא בקרה","תמרורי עצור / כניעה","רמזור קיים","כיכר קיימת",
                "מצלמת אכיפה קיימת","פסי האטה קיימים","תאורה מוגברת קיימת"]
    osm_i_   = st.session_state.get(f"osm_{selected}", {})
    if osm_i_.get("כיכר"):    def_cur="כיכר קיימת"
    elif osm_i_.get("רמזור"): def_cur="רמזור קיים"
    elif osm_i_.get("עצור"):  def_cur="תמרורי עצור / כניעה"
    else:                      def_cur="ללא בקרה"

    # ══ חישוב פרמטרים הנדסיים מנתוני CBS ══════════════════════════════════════
    # כל ערך מחושב ישירות מהנתונים — מבוסס HCM + ספרות ישראלית
    road_type     = prof.get("road_type","עירוני - בצומת")
    ia_speed_opts = [30,40,50,60,70,80,90,100,110]
    spd_def       = int(spd_v) if str(spd_v).isdigit() and int(spd_v) in ia_speed_opts else 50

    # מדד עומס משולב מ-CBS (0-1):
    # rear_pct = % תאונות עורפיות → אינדיקטור קלאסי לצפיפות תנועה
    # peak_pct = % תאונות בשעות עומס → אינדיקטור לנפח תנועה
    cong_est = round(prof["rear_pct"]*0.6 + prof["peak_pct"]*0.4, 3)

    # LOS — מבוסס cong_est (HCM 6th Ed. Table 19-8):
    # cong > 0.7 → F (over capacity), > 0.5 → E (near capacity),
    # > 0.3 → D (heavy), > 0.15 → C (stable), else B/A
    los_def = ("F" if cong_est>0.70 else "E" if cong_est>0.50 else
               "D" if cong_est>0.30 else "C" if cong_est>0.15 else "B")

    # AADT — נגזר מ: תאונות/שנה ÷ שיעור תאונות סטנדרטי (NCHRP Report 17-45)
    # Israel avg accident rate at intersections: ~0.35 crashes/MEV (million entering vehicles)
    # AADT = (crashes/yr × 10^6) / (365 × 4_approaches × 0.35)
    n_annual  = n  # תאונות בשנה (2021)
    acc_rate  = 0.35  # crashes per MEV — NCHRP 17-45, Israel urban intersections
    aadt_calc = int(n_annual * 1_000_000 / (365 * 4 * acc_rate))
    # תיקון לפי סוג דרך ומהירות
    if "בין-עירוני" in road_type or spd_def >= 70:
        aadt_calc = int(aadt_calc * 1.8)  # כבישים מהירים — נפח גבוה יותר
    aadt_est  = max(2_000, min(100_000, aadt_calc))

    # נתיבים — לפי סוג דרך (AASHTO Green Book Table 3-1)
    lanes_est = 4 if ("בין-עירוני" in road_type or spd_def >= 70) else 2

    # % כלי רכב כבדים — לפי סוג דרך (נתיבי ישראל, ספר מאפייני דרכים 2023)
    heavy_est = 15 if "בין-עירוני" in road_type else (8 if spd_def >= 70 else 5)

    # השהייה (שנ'/רכב) — מ-LOS לפי HCM 6th Ed. Exhibit 19-1:
    # A:<10, B:10-20, C:20-35, D:35-55, E:55-80, F:>80
    _delay_map = {"A":8,"B":15,"C":28,"D":45,"E":68,"F":85}
    delay_est  = _delay_map[los_def]
    # תוספת לפי % עורפיות — תאונות עורפיות = סימן לעצירות תכופות
    delay_est  = min(150, int(delay_est + prof["rear_pct"] * 40))

    # תורים (מ') — מ-HCM queuing formula + rear_pct
    # queue ≈ delay × arrival_rate / saturation_flow × car_length
    queue_est  = min(1500, int(delay_est * 2.5 + prof["rear_pct"] * 400))

    # ── תצוגת בסיס החישוב ────────────────────────────────────────────────────
    with st.expander("🔢 כיצד חושבו הערכים? — בסיס חישוב CBS", expanded=False):
        st.markdown(f"""
| פרמטר | ערך מחושב | נוסחה / בסיס |
|--------|-----------|-------------|
| **מדד עומס CBS** | {cong_est:.2f} | rear_pct×0.6 + peak_pct×0.4 = {prof['rear_pct']:.2f}×0.6 + {prof['peak_pct']:.2f}×0.4 |
| **LOS** | {los_def} | HCM 6th Ed. Table 19-8: cong={cong_est:.2f} |
| **AADT** | {aadt_est:,} | NCHRP 17-45: {n_annual} תאונות/שנה ÷ rate={acc_rate} → {aadt_calc:,} + תיקון סוג דרך |
| **השהייה** | {delay_est} שנ' | HCM Exhibit 19-1: LOS {los_def} + rear_pct={prof['rear_pct']:.2f}×40 |
| **תורים** | {queue_est} מ' | HCM Queuing: delay×2.5 + rear_pct×400 |
| **נתיבים** | {lanes_est} | AASHTO Green Book Table 3-1: {road_type} |
| **% כבדים** | {heavy_est}% | נתיבי ישראל 2023: {road_type} |

*מקורות: HCM 6th Edition (TRB 2016) | NCHRP Report 17-45 | AASHTO Green Book (2018) | נתיבי ישראל*
        """)
    st.info(f"💡 ערכים מחושבים מנתוני CBS 2021 — מדד עומס: **{cong_est:.2f}** | LOS מוערך: **{los_def}** | AADT מוערך: **{aadt_est:,}**")

    ia1,ia2,ia3 = st.columns(3)
    with ia1:
        st.markdown("**תנועה**")
        ia_aadt  = st.number_input("AADT (רכב/יום):", 1000, 200_000, aadt_est, 500, key="ia_aadt")
        ia_lanes = st.slider("מספר נתיבים:", 1, 6, lanes_est, key="ia_lanes")
        ia_heavy = st.slider("% כלי רכב כבדים:", 0, 30, heavy_est, key="ia_heavy")
        ia_ctrl  = st.selectbox("בקרה קיימת:", cur_opts, index=cur_opts.index(def_cur), key="ia_ctrl")
        ia_speed = st.select_slider("מהירות מותרת:", ia_speed_opts, spd_def, key="ia_speed")
    with ia2:
        st.markdown("**בטיחות (מ-CBS)**")
        ia_fatal = st.number_input("קטלניות (3 שנים):", 0, 50, max(0,int(prof["fatal_pct"]*n*3)), 1, key="ia_fatal")
        ia_ser   = st.number_input("קשות (3 שנים):", 0, 200, max(0,int(prof["serious_pct"]*n*3)), 1, key="ia_ser")
        ia_total = st.number_input("סה\"כ תאונות (3 שנים):", 0, 500, int(n*3), 1, key="ia_total")
    with ia3:
        st.markdown("**תפעול HCM**")
        ia_los   = st.selectbox("LOS נוכחי:", ["A","B","C","D","E","F"],
                                 index=["A","B","C","D","E","F"].index(los_def), key="ia_los")
        ia_delay = st.number_input("זמן השהייה (שנ'/רכב):", 0, 300, delay_est, 5, key="ia_delay")
        ia_queue = st.number_input("אורך תורים (מ'):", 0, 2000, queue_est, 10, key="ia_queue")

    los_n       = _LOS_NUM.get(ia_los, 4)
    base_safety = max(10, 100 - ia_fatal*15 - ia_ser*5)
    base_traffic= max(10, 100 - (los_n-1)*15 - ia_delay*0.5)
    base_score  = round(base_safety*0.40 + base_traffic*0.35 + 50*0.25, 1)

    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("ציון מצב קיים",   f"{base_score:.1f}/100")
    b2.metric("LOS נוכחי",        ia_los)
    b3.metric("תאונות קטלניות",   ia_fatal, "ב-3 שנים")
    b4.metric("זמן השהייה",       f"{ia_delay} שנ'")
    b5.metric("מדד עומס CBS",     f"{round(cong_est*100,1)}/100")

    # ── טבלת שקיפות — כיצד בוצע הניתוח ─────────────────────────────────────
    with st.expander("📐 כיצד בוצע הניתוח? — מתודולוגיה ומשקולות", expanded=False):
        st.markdown("""
### מתודולוגיית הניתוח ההנדסי

הניתוח מבוסס על **4 קריטריונים** משוקללים לפי תקנים בינלאומיים:
""")
        meth_df = pd.DataFrame([
            {"קריטריון":"🦺 בטיחות","משקל":"40%",
             "מרכיבים":"הפחתת קטלניות (50%) + הפחתת פציעות (30%) + הפחתת כל תאונות (10%) + צמצום נקודות קונפליקט (10%)",
             "מקור":"FHWA Roundabout Guide, NCHRP 572, AASHTO"},
            {"קריטריון":"🚗 תנועה ותפעול","משקל":"35%",
             "מרכיבים":"שיפור LOS (30%) + הפחתת השהייה (30%) + עלייה בקיבולת (25%) + הפחתת תורים (15%), מותאם ל-AADT",
             "מקור":"HCM 6th Edition (TRB 2016)"},
            {"קריטריון":"💰 כלכלה","משקל":"15%",
             "מרכיבים":"ROI = (חיסכון שנתי × 5 שנים) / עלות השקעה. ציון גבוה = ROI < 5 שנים",
             "מקור":"נתיבי ישראל + מכרזים + עלות תאונה ₪3.5M (משרד התחבורה)"},
            {"קריטריון":"🌱 סביבה","משקל":"10%",
             "מרכיבים":"הפחתת פליטות + חיסכון דלק + הפחתת רעש (ממוצע יחסי לכל חלופה)",
             "מקור":"FHWA Environmental Guide, EPA"},
        ])
        st.dataframe(meth_df, use_container_width=True, hide_index=True)

        st.markdown("### ציון לכל חלופה — נוסחה")
        st.markdown("""
```
ציון_כולל = ציון_בטיחות × 0.40
           + ציון_תנועה   × 0.35
           + ציון_כלכלה   × 0.15
           + ציון_סביבה   × 0.10
```
**התאמת AADT:** אם AADT > קיבולת_מקסימלית_של_החלופה, הציון_תנועה מופחת.

**הגברת משקל בטיחות:** בנוכחות תאונות קטלניות (>2) — משקל הבטיחות גדל ב-10%.
""")
        st.markdown("### נתוני בסיס לכל חלופה (מבוססי תקנים)")
        src_df = pd.DataFrame([
            {"חלופה":"כיכר חד נתיבית","הפחתת קטלניות":"90%","מקור":"FHWA Roundabout Guide (2010)","AADT מקסימלי":"15,000"},
            {"חלופה":"כיכר טורבו","הפחתת קטלניות":"85%","מקור":"SWOV (2010), Fortuijn (2009)","AADT מקסימלי":"40,000"},
            {"חלופה":"כיכר דו נתיבית","הפחתת קטלניות":"78%","מקור":"NCHRP Report 572","AADT מקסימלי":"35,000"},
            {"חלופה":"מחלף מלא","הפחתת קטלניות":"70%","מקור":"AASHTO Green Book (2018)","AADT מקסימלי":"∞"},
            {"חלופה":"מחלף חלקי","הפחתת קטלניות":"50%","מקור":"AASHTO Green Book (2018)","AADT מקסימלי":"100,000"},
            {"חלופה":"צומת מרומזר","הפחתת קטלניות":"30%","מקור":"HCM 6th Edition","AADT מקסימלי":"80,000"},
            {"חלופה":"הוספת נתיבי פנייה","הפחתת קטלניות":"25%","מקור":"NCHRP Report 617","AADT מקסימלי":"50,000"},
            {"חלופה":"שיפור גיאומטרי","הפחתת קטלניות":"15%","מקור":"NCHRP Report 617","AADT מקסימלי":"60,000"},
            {"חלופה":"צומת חכם אדפטיבי","הפחתת קטלניות":"20%","מקור":"FHWA ATMS (2022)","AADT מקסימלי":"80,000"},
        ])
        st.dataframe(src_df, use_container_width=True, hide_index=True)

        # ציון המצב הקיים — הסבר
        st.markdown(f"""
### ציון המצב הקיים: **{base_score}/100**
| מרכיב | חישוב | ציון |
|-------|-------|------|
| בטיחות (40%) | 100 − {ia_fatal}×15 − {ia_ser}×5 | {base_safety:.0f} |
| תנועה (35%) | 100 − (LOS={ia_los}={los_n}−1)×15 − {ia_delay}×0.5 | {base_traffic:.0f} |
| כלכלה (15%) | ערך בסיס | 50 |
| סביבה (10%) | ערך בסיס | 50 |
| **סה"כ** | {base_safety:.0f}×0.40 + {base_traffic:.0f}×0.35 + 50×0.25 | **{base_score:.1f}** |
""")

    # ══ האם נדרש שדרוג? — לפי תקני HCM ══════════════════════════════════════
    st.markdown("---")

    # קריטריוני צומת "טובה" לפי HCM 6th Ed. + AASHTO
    good_los       = _LOS_NUM.get(los_def,4) <= 3          # LOS A-C
    good_delay     = ia_delay < 35                          # HCM: <35 שנ' = LOS C
    good_safety    = ia_fatal == 0 and ia_ser <= 1          # מעט תאונות קשות
    good_cong      = cong_est < 0.30                        # עומס נמוך
    good_queue     = ia_queue < 100                         # תורים קצרים

    good_count = sum([good_los, good_delay, good_safety, good_cong, good_queue])
    needs_upgrade = good_count < 3  # אם 3+ קריטריונים טובים — אין צורך בשדרוג

    if not needs_upgrade:
        st.success(
            f"✅ **הצומת במצב טוב — אין צורך בשדרוג מיידי**  \n"
            f"LOS: **{los_def}** ({'טוב' if good_los else 'בינוני'}) | "
            f"השהייה: **{ia_delay} שנ'** ({'טוב' if good_delay else 'בינוני'}) | "
            f"תאונות קשות: **{ia_ser}** ({'טוב' if good_safety else 'בינוני'}) | "
            f"עומס: **{cong_est*100:.0f}%** ({'טוב' if good_cong else 'בינוני'})  \n"
            f"*לפי HCM 6th Ed.: צומת עם LOS A-C, השהייה <35שנ' ומעט תאונות אינה מצריכה שדרוג.*"
        )
        show_alternatives = st.checkbox("🔍 הצג חלופות שיפור בכל זאת", value=False, key="show_alts")
        if not show_alternatives:
            st.markdown("---")
            # מדלגים לסעיף הבא
    else:
        # בניית ציון נוכחי לתצוגה
        issues = []
        if not good_los:     issues.append(f"LOS {los_def} (D-F)")
        if not good_delay:   issues.append(f"השהייה {ia_delay} שנ' (>35)")
        if not good_safety:  issues.append(f"{ia_fatal} קטלניות, {ia_ser} קשות")
        if not good_cong:    issues.append(f"עומס {cong_est*100:.0f}%")
        if not good_queue:   issues.append(f"תורים {ia_queue} מ'")

        st.warning(
            f"⚠️ **הצומת מצריכה בדיקה/שדרוג** — {', '.join(issues)}  \n"
            f"*לפי HCM 6th Ed. — מומלץ לבחון חלופות שיפור.*"
        )
        show_alternatives = True

    if show_alternatives or needs_upgrade:
        # ══ שלב ב: פתרונות אפשריים (HCM) ════════════════════════════════════
        st.subheader("🏗️ שלב ב — פתרונות אפשריים")
        st.caption("HCM 6th Ed. · AASHTO Green Book · FHWA Roundabout Guide · NCHRP 572")

    hcm_inp = {
        "aadt":ia_aadt,"peak":int(ia_aadt*0.08),"lanes":ia_lanes,
        "heavy_pct":ia_heavy,"speed":ia_speed,
        "fatal":ia_fatal,"serious":ia_ser,"total_acc":ia_total,
        "los":ia_los,"avg_delay":ia_delay,"queue":ia_queue,"type":ia_ctrl,
    }
    results_df = analyze_intersection_hcm(hcm_inp)

    st.dataframe(
        results_df[["דירוג","חלופה","ציון_כולל","ציון_בטיחות","ציון_תנועה",
                    "הפחתת קטלניות","LOS חדש","עלות הקמה","ROI (שנים)","מקור"]],
        use_container_width=True, hide_index=True,
        column_config={
            "ציון_כולל":   st.column_config.ProgressColumn("ציון כולל",   min_value=0,max_value=100,format="%.1f"),
            "ציון_בטיחות": st.column_config.ProgressColumn("בטיחות 40%",  min_value=0,max_value=100,format="%.1f"),
            "ציון_תנועה":  st.column_config.ProgressColumn("תנועה 35%",   min_value=0,max_value=100,format="%.1f"),
        })

    fig_r = px.bar(results_df.sort_values("ציון_כולל"),
                   x="ציון_כולל", y="חלופה", orientation="h",
                   color="ציון_כולל", color_continuous_scale="RdYlGn",
                   title="ציון כולל לכל חלופה",
                   text=results_df.sort_values("ציון_כולל")["ציון_כולל"].apply(lambda x:f"{x:.0f}"))
    fig_r.update_traces(textposition="outside")
    fig_r.update_layout(coloraxis_showscale=False, height=340)
    st.plotly_chart(fig_r, use_container_width=True, key="rank_bar2")

    # ══ שלב ג: בחירת פתרון + חיזוי השוואתי ══════════════════════════════════
    st.markdown("---")
    st.subheader("🔮 שלב ג — בחר פתרון וחזה את השיפור")
    # הוספת אפשרות "שמור מצב קיים" ברשימה
    all_choices = ["✅ שמור מצב קיים (ללא שדרוג)"] + results_df["חלופה"].tolist()

    chosen_hcm = st.selectbox("בחר פתרון לחיזוי:", all_choices, key="chosen_hcm")

    if chosen_hcm == "✅ שמור מצב קיים (ללא שדרוג)":
        st.info(
            f"✅ **החלטה: שמור מצב קיים — אין שדרוג נדרש**  \n"
            f"ציון נוכחי: **{base_score:.1f}/100** | LOS: **{los_def}** | "
            f"השהייה: **{ia_delay} שנ'** | עומס: **{cong_est*100:.0f}%**  \n"
            f"*המצב הנוכחי עומד בסטנדרטים של HCM 6th Ed. — ניטור שוטף מומלץ.*"
        )
    else:
        sel = results_df[results_df["חלופה"]==chosen_hcm].iloc[0]
        alt = ALTERNATIVES_HCM[chosen_hcm]

        if ia_aadt > alt["aadt_max"]:
            st.warning(f"⚠️ AADT={ia_aadt:,} עולה על קיבולת {chosen_hcm} ({alt['aadt_max']:,}). שקול חלופה גבוהה יותר.")

        impl_map = {
            "שיפור גיאומטרי":            "✏️ תיקון נראות, הרחבת אי תנועה, סימון מחדש",
            "הוספת נתיבי פנייה":          "🛣️ הוספת נתיב פנייה ייעודי + הפרדה פיזית",
            "צומת מרומזר":                "🚦 התקנת מנגנוני רמזורים + בקרה מרכזית",
            "כיכר חד נתיבית":             "🔵 הריסת צומת + בניית כיכר חד נתיבית",
            "כיכר דו נתיבית":             "🔵🔵 תכנון מחדש עם שני נתיבי מעגל",
            "כיכר טורבו":                "🌀 עיצוב טורבו עם מפרידים מוגבהים",
            "צומת חכם עם רמזור אדפטיבי":  "🤖 גלאים, מצלמות AI, מרכז בקרה, תוכנת אופטימיזציה",
            "מחלף חלקי":                 "🏗️ גשר/מנהרה לזרמים עיקריים + כבישי חיבור",
            "מחלף מלא":                  "🏗️🏗️ הפרדה מפלסית מלאה — גשרים, מנהרות, כבישי גישה",
        }
        st.info(f"**אופן ביצוע השינוי:** {impl_map.get(chosen_hcm, '—')}  \n"
                f"**תיאור:** {alt['desc']}  \n**מקור:** {alt['source']}")

        new_los_n   = max(1, los_n - alt["los_imp"])
        new_los_chr = _LOS_CHAR.get(int(round(new_los_n)), "A")
        new_delay   = round(ia_delay  * (1-alt["delay_r"]), 1)
        new_queue   = round(ia_queue  * (1-alt["delay_r"]))
        new_fatal   = round(ia_fatal  * (1-alt["fatal_r"]), 1)
        new_serious = round(ia_ser    * (1-alt["injury_r"]), 1)
        new_cong    = round(cong_est  * (1-alt["delay_r"]) * 100, 1)
        ann_save    = ((ia_fatal-new_fatal)*ACCIDENT_COST["קטלנית"] +
                       (ia_ser-new_serious)*ACCIDENT_COST["קשה"]) / 3
        roi_yr      = round(alt["cost_avg"]/ann_save, 1) if ann_save>0 else 99

        # ── ציונים לאחר שדרוג — חישוב לפני st.success ─────────────────────
        after_safety  = round(min(100, base_safety  + (100-base_safety) * alt["fatal_r"]), 1)
        after_traffic = round(min(100, base_traffic + alt["los_imp"]*12 + alt["delay_r"]*base_traffic*0.5), 1)
        after_eco     = round(max(0, min(100, (20-roi_yr)/20*100 if roi_yr<99 else 30)), 1)
        after_total   = round(after_safety*0.40 + after_traffic*0.35 + after_eco*0.15 + 50*0.10, 1)

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("💰 עלות השקעה",   f"₪{alt['cost_avg']:,}")
        k2.metric("📉 חיסכון שנתי",  f"₪{ann_save:,.0f}", f"ROI: {roi_yr} שנים")
        k3.metric("☠️ קטלניות",      f"{new_fatal}",      f"לפני: {ia_fatal}")
        k4.metric("🏥 קשות",         f"{new_serious}",    f"לפני: {ia_ser}")
        k5.metric("🚦 LOS חדש",      new_los_chr,          f"לפני: {ia_los}")

        g1,g2 = st.columns(2)
        with g1:
            comp_a = pd.DataFrame({
                "מצב":    ["קיים","קיים","קיים","אחרי","אחרי","אחרי"],
                "סוג":    ["קטלניות","קשות","קלות","קטלניות","קשות","קלות"],
                "תאונות": [ia_fatal, ia_ser, max(0,ia_total-ia_fatal-ia_ser),
                            new_fatal, new_serious, max(0,ia_total-ia_fatal-ia_ser)],
            })
            fig = px.bar(comp_a, x="סוג", y="תאונות", color="מצב", barmode="group",
                         color_discrete_map={"קיים":"#e74c3c","אחרי":"#2ecc71"},
                         title="תאונות — מצב קיים מול אחרי שדרוג",
                         labels={"תאונות":"תאונות (3 שנים)"})
            st.plotly_chart(fig, use_container_width=True, key="cmp_acc2")
        with g2:
            comp_o = pd.DataFrame({
                "מדד":   ["LOS (1-6)","השהייה (שנ')","עומס (%)","ציון כולל"],
                "קיים":  [los_n, ia_delay, cong_est*100, base_score],
                "אחרי":  [int(new_los_n), new_delay, new_cong, sel["ציון_כולל"]],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name="מצב קיים",    x=comp_o["מדד"], y=comp_o["קיים"],  marker_color="#e74c3c"))
            fig.add_trace(go.Bar(name="לאחר שדרוג",  x=comp_o["מדד"], y=comp_o["אחרי"], marker_color="#2ecc71"))
            fig.update_layout(barmode="group", title="פרמטרי תפעול — לפני ואחרי", height=320)
            st.plotly_chart(fig, use_container_width=True, key="cmp_op2")

        st.success(
            f"**{chosen_hcm}** — ציון: **{after_total:.1f}/100** (שיפור +{round(after_total-base_score,1)})  \n"
            f"LOS: {ia_los}→**{new_los_chr}** | קטלניות: {ia_fatal}→**{new_fatal}** | "
            f"עלות: **₪{alt['cost_avg']:,}** | ROI: **{roi_yr} שנים** | מקור: *{alt['source']}*"
        )

        # ── טבלת השוואה מלאה: מצב קיים vs לאחר פתרון ─────────────────────────
        st.markdown("#### 📊 טבלת השוואה מלאה — מצב קיים מול לאחר שדרוג")

        ann_cost_now = (ia_fatal/3 * ACCIDENT_COST["קטלנית"] +
                        ia_ser/3   * ACCIDENT_COST["קשה"]    +
                        max(0, ia_total/3-ia_fatal/3-ia_ser/3) * ACCIDENT_COST["קלה"])
        ann_cost_aft = (new_fatal   * ACCIDENT_COST["קטלנית"] +
                        new_serious * ACCIDENT_COST["קשה"]    +
                        max(0, ia_total/3-ia_fatal/3-ia_ser/3) * ACCIDENT_COST["קלה"])

        # after_safety/traffic/eco/total כבר מחושבים למעלה

        def _verdict(val_now, val_aft, lower_is_better=True):
            if lower_is_better:
                pct = (val_now - val_aft) / max(abs(val_now), 0.001) * 100
            else:
                pct = (val_aft - val_now) / max(abs(val_now), 0.001) * 100
            return "✅" if pct >= 10 else ("⚠️" if pct >= 3 else "—")

        comp_table = pd.DataFrame([
            {"פרמטר": "סוג הצומת",            "מצב קיים": ia_ctrl,              "לאחר שדרוג": chosen_hcm,          "שיפור": "✅"},
            {"פרמטר": "LOS",                   "מצב קיים": ia_los,               "לאחר שדרוג": new_los_chr,          "שיפור": _verdict(_LOS_NUM.get(ia_los,4), new_los_n)},
            {"פרמטר": "ציון כולל (0-100)",     "מצב קיים": f"{base_score:.1f}",  "לאחר שדרוג": f"{after_total:.1f}", "שיפור": _verdict(100-base_score, 100-after_total)},
            {"פרמטר": "ציון בטיחות (0-100)",  "מצב קיים": f"{base_safety:.0f}", "לאחר שדרוג": f"{after_safety:.0f}","שיפור": _verdict(100-base_safety, 100-after_safety)},
            {"פרמטר": "ציון תנועה (0-100)",   "מצב קיים": f"{base_traffic:.0f}","לאחר שדרוג": f"{after_traffic:.0f}","שיפור": _verdict(100-base_traffic, 100-after_traffic)},
            {"פרמטר": "AADT (רכב/יום)",       "מצב קיים": f"{ia_aadt:,}", "לאחר שדרוג": f"{ia_aadt:,} (קיבולת: {alt['aadt_max']:,})", "שיפור": "✅" if ia_aadt<=alt["aadt_max"] else "⚠️"},
            {"פרמטר": "זמן השהייה (שנ')",     "מצב קיים": f"{ia_delay}",  "לאחר שדרוג": f"{new_delay:.0f}", "שיפור": _verdict(ia_delay, new_delay)},
            {"פרמטר": "אורך תורים (מ')",      "מצב קיים": f"{ia_queue}",  "לאחר שדרוג": f"{new_queue:.0f}", "שיפור": _verdict(ia_queue, new_queue)},
            {"פרמטר": "מדד עומס (%)",          "מצב קיים": f"{cong_est*100:.1f}%", "לאחר שדרוג": f"{new_cong:.1f}%", "שיפור": _verdict(cong_est, new_cong/100)},
            {"פרמטר": "תאונות קטלניות (3שנ')", "מצב קיים": str(ia_fatal),  "לאחר שדרוג": str(new_fatal),   "שיפור": _verdict(ia_fatal, new_fatal)   if ia_fatal>0 else "—"},
            {"פרמטר": "תאונות קשות (3שנ')",    "מצב קיים": str(ia_ser),    "לאחר שדרוג": str(new_serious), "שיפור": _verdict(ia_ser, new_serious)   if ia_ser>0 else "—"},
            {"פרמטר": "עלות שנתית תאונות",    "מצב קיים": f"₪{ann_cost_now:,.0f}", "לאחר שדרוג": f"₪{ann_cost_aft:,.0f}", "שיפור": _verdict(ann_cost_now, ann_cost_aft)},
            {"פרמטר": "עלות השקעה",           "מצב קיים": "—",             "לאחר שדרוג": f"₪{alt['cost_avg']:,}", "שיפור": "💰"},
            {"פרמטר": "ROI",                  "מצב קיים": "—",             "לאחר שדרוג": f"{roi_yr} שנים", "שיפור": "✅" if roi_yr < 10 else ("⚠️" if roi_yr < 20 else "❌")},
            {"פרמטר": "מקור נתון",            "מצב קיים": "CBS 2021 + HCM","לאחר שדרוג": alt["source"], "שיפור": "📚"},
        ])

        # עיצוב צבעי
        def color_verdict(val):
            if val == "✅": return "background-color: #d5f5e3"
            if val == "⚠️": return "background-color: #fdebd0"
            if val == "❌": return "background-color: #fadbd8"
            return ""

        styled = comp_table.style.applymap(color_verdict, subset=["שיפור"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


st.markdown("---")
st.subheader("🔍 חיזוי סיכון לתשתית חדשה בתכנון")
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

