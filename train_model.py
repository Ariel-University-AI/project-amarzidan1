"""
train_model.py — אימון מודל חיזוי התערבות מומלצת לצומת
========================================================
X: פרופיל תאונות + מדד עומס לאתר
y: התערבות מומלצת (7 סוגים)
מודל: Random Forest Classifier
"""

import os, sys, json, joblib
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from collections import Counter

# ── CBS decode ────────────────────────────────────────────────────────────────
_SEV    = {1:"קטלנית",2:"קשה",3:"קלה"}
_ROAD   = {1:"עירוני - בצומת",2:"עירוני - לא בצומת",
           3:"בין-עירוני - בצומת",4:"בין-עירוני - לא בצומת",5:"חניון / כיכר",9:"אחר"}
_WTHR   = {1:"בהיר",2:"גשם קל",3:"גשם",4:"ערפל",5:"חול",7:"שלג",8:"סופה",9:"אחר"}
_DNTM   = {1:"יום",5:"לילה"}
_DIST   = {1:"ירושלים",2:"צפון",3:"חיפה",4:"מרכז",5:"תל אביב",6:"דרום",7:'יו"ש'}
_SPD    = {1:"30",2:"40",3:"50",4:"60",5:"70",6:"80",7:"90",8:"100",9:"110"}
_ACCTYP = {1:"חזיתית",2:"אחורית",3:"צידית",4:"הולך רגל",
           5:"התהפכות",6:"פגיעה בעמוד",7:"נפילה מרכב",8:"אחר"}
_PEAK   = {7,8,9,16,17,18,19}

# ── Interventions (same as app) ───────────────────────────────────────────────
INTERVENTIONS = {
    "כיכר תנועה":              {"fatal_r":0.82,"serious_r":0.55,"cong_r":0.30,
                                 "cost_avg":500_000,
                                 "tags":{"חזיתית","התהפכות","מהירות_גבוהה"}},
    "רמזור חכם":               {"fatal_r":0.45,"serious_r":0.32,"cong_r":0.25,
                                 "cost_avg":7_000_000,
                                 "tags":{"הולך רגל","עירוני","עומס"}},
    "מעבר חצייה מוגן + תאורה": {"fatal_r":0.55,"serious_r":0.38,"cong_r":0.05,
                                 "cost_avg":120_000,
                                 "tags":{"הולך רגל","עירוני"}},
    "תאורת לד מוגברת":         {"fatal_r":0.32,"serious_r":0.22,"cong_r":0.05,
                                 "cost_avg":9_500_000,
                                 "tags":{"לילה","ערפל"}},
    "מצלמת אכיפה + מד-מהירות": {"fatal_r":0.25,"serious_r":0.18,"cong_r":0.10,
                                 "cost_avg":120_000,
                                 "tags":{"מהירות_גבוהה","בין-עירוני"}},
    "פסי האטה / מוקפצים":      {"fatal_r":0.30,"serious_r":0.25,"cong_r":0.05,
                                 "cost_avg":70_000,
                                 "tags":{"אחורית","מהירות_בינונית"}},
    "הוספת נתיב נסיעה":        {"fatal_r":0.20,"serious_r":0.35,"cong_r":0.45,
                                 "cost_avg":8_000_000,
                                 "tags":{"אחורית","עומס","בין-עירוני"}},
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE,"data","accidents_israel_2021_raw.csv")
MDL_DIR  = os.path.join(BASE,"models")
MDL_PATH = os.path.join(MDL_DIR,"model.pkl")
ENC_PATH = os.path.join(MDL_DIR,"encoders.pkl")
MET_PATH = os.path.join(MDL_DIR,"metrics.json")
SIT_PATH = os.path.join(MDL_DIR,"sites_cache.pkl")

FEAT_COLS = [
    "תאונות","fatal_pct","serious_pct","night_pct","rain_pct",
    "pedestrian_pct","frontal_pct","rear_pct","rollover_pct",
    "high_speed","peak_pct",  # peak_pct = מדד עומס חדש
]
TARGET = "label_intervention"


def load_and_decode():
    print("[*] Loading data...")
    df = pd.read_csv(DATA,low_memory=False)
    df.columns = df.columns.str.strip()
    for c in ["HUMRAT_TEUNA","SUG_DEREH","SUG_TEUNA","MEZEG_AVIR","PNE_KVISH",
              "YOM_LAYLA","MAHOZ","MEHIRUT_MUTERET","SEMEL_YISHUV","KVISH1",
              "HODESH_TEUNA","SHAA","X","Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c],errors="coerce")
    df["חומרת_תאונה"] = df["HUMRAT_TEUNA"].map(_SEV)
    df["סוג_דרך"]     = df["SUG_DEREH"].map(_ROAD)
    df["סוג_תאונה"]   = df["SUG_TEUNA"].map(_ACCTYP)
    df["מזג_אוויר"]   = df["MEZEG_AVIR"].map(_WTHR)
    df["חלק_יממה"]    = df["YOM_LAYLA"].map(_DNTM)
    df["מהירות_מותרת"]= df["MEHIRUT_MUTERET"].map(_SPD)
    df["שעה_מספר"]    = df["SHAA"].fillna(0).astype(int)
    df["בשעת_עומס"]   = df["שעה_מספר"].isin(_PEAK)
    df["כביש"]        = df["KVISH1"].where(df["KVISH1"].notna()&(df["KVISH1"]>0))
    df["מיקום"]       = df.apply(
        lambda r: f"כביש {int(r['כביש'])}" if pd.notna(r["כביש"]) else "לא ידוע",axis=1)
    df["אתר"]         = df["מיקום"]+" – "+df["סוג_דרך"].fillna("לא ידוע")
    print(f"    -> {len(df):,} rows loaded")
    return df


def build_site_profile(df):
    print("[*] Building per-site profiles...")
    records=[]
    for site,grp in df.groupby("אתר"):
        n   = max(len(grp),1)
        spd = grp["מהירות_מותרת"].mode()
        spv = spd.iloc[0] if len(spd) else "50"
        rd  = grp["סוג_דרך"].mode()
        rdv = rd.iloc[0] if len(rd) else ""

        rec = {
            "אתר":           site,
            "תאונות":        len(grp),
            "fatal_pct":     (grp["חומרת_תאונה"]=="קטלנית").sum()/n,
            "serious_pct":   (grp["חומרת_תאונה"]=="קשה").sum()/n,
            "night_pct":     (grp["חלק_יממה"]=="לילה").sum()/n,
            "rain_pct":      grp["מזג_אוויר"].isin(["גשם","גשם קל"]).sum()/n,
            "pedestrian_pct":(grp["סוג_תאונה"]=="הולך רגל").sum()/n,
            "frontal_pct":   (grp["סוג_תאונה"]=="חזיתית").sum()/n,
            "rear_pct":      (grp["סוג_תאונה"]=="אחורית").sum()/n,
            "rollover_pct":  (grp["סוג_תאונה"]=="התהפכות").sum()/n,
            "peak_pct":      grp["בשעת_עומס"].sum()/n,
            "high_speed":    1 if str(spv) in {"70","80","90","100","110"} else 0,
            "road_type":     rdv,
        }

        # Generate label
        scores={}
        for name,info in INTERVENTIONS.items():
            s=(rec["fatal_pct"]*info["fatal_r"]+rec["serious_pct"]*info["serious_r"])*rec["תאונות"]*10
            if rec["pedestrian_pct"]>0.12 and "הולך רגל"     in info["tags"]: s+=25
            if rec["frontal_pct"]   >0.18 and "חזיתית"       in info["tags"]: s+=25
            if rec["night_pct"]     >0.35 and "לילה"         in info["tags"]: s+=20
            if rec["high_speed"]          and "מהירות_גבוהה" in info["tags"]: s+=20
            if rec["rear_pct"]      >0.20 and "אחורית"       in info["tags"]: s+=15
            if "עירוני" in rdv            and "עירוני"       in info["tags"]: s+=10
            if "בין-עירוני" in rdv        and "בין-עירוני"   in info["tags"]: s+=10
            if name=="הוספת נתיב נסיעה"  and rec["high_speed"]:               s+=15
            if name=="הוספת נתיב נסיעה"  and rec["rear_pct"]>0.25:            s+=20
            if name=="הוספת נתיב נסיעה"  and rec["peak_pct"]>0.40:            s+=15
            scores[name]=max(0.0,s)
        rec[TARGET]=max(scores,key=scores.get)
        records.append(rec)

    sites_df=pd.DataFrame(records)
    sites_df=sites_df[sites_df["תאונות"]>=3].reset_index(drop=True)
    print(f"    -> {len(sites_df):,} sites (≥3 accidents)")
    return sites_df


def train():
    df       = load_and_decode()
    sites_df = build_site_profile(df)

    X = sites_df[FEAT_COLS].values
    y = sites_df[TARGET].values

    min_count = min(Counter(y).values())
    strat = y if min_count>=2 else None
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=strat)
    print(f"\n[*] Split: Train={len(X_tr):,} | Test={len(X_te):,}")

    print("\n[*] Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200,max_depth=10,
                                   min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(X_tr,y_tr)

    y_pred=model.predict(X_te)
    acc  =accuracy_score(y_te,y_pred)
    f1   =f1_score(y_te,y_pred,average="weighted",zero_division=0)
    prec =precision_score(y_te,y_pred,average="weighted",zero_division=0)
    rec  =recall_score(y_te,y_pred,average="weighted",zero_division=0)

    labels_order=sorted(INTERVENTIONS.keys())
    print(f"\n{'='*60}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\n"+classification_report(y_te,y_pred,zero_division=0))

    print("\n[Feature Importance]")
    for feat,imp in sorted(zip(FEAT_COLS,model.feature_importances_),key=lambda x:x[1],reverse=True):
        print(f"   {feat:22s}  {imp:.4f}  {'█'*int(imp*40)}")

    print("\n[Label Distribution]")
    vc=pd.Series(y).value_counts()
    for lbl,cnt in vc.items():
        print(f"   {lbl}: {cnt} ({cnt/len(y)*100:.1f}%)")

    os.makedirs(MDL_DIR,exist_ok=True)
    joblib.dump(model,MDL_PATH);  print(f"\n[SAVED] {MDL_PATH}")
    le=LabelEncoder(); le.fit(y)
    joblib.dump(le,ENC_PATH);     print(f"[SAVED] {ENC_PATH}")
    joblib.dump(sites_df,SIT_PATH);print(f"[SAVED] {SIT_PATH}")

    cm=confusion_matrix(y_te,y_pred,labels=labels_order)
    metrics={
        "accuracy":           round(acc,4),
        "f1_score":           round(f1,4),
        "precision":          round(prec,4),
        "recall":             round(rec,4),
        "confusion_matrix":   cm.tolist(),
        "labels":             labels_order,
        "feature_importance": dict(zip(FEAT_COLS,model.feature_importances_.tolist())),
        "feature_columns":    FEAT_COLS,
        "target_column":      TARGET,
        "train_size":         int(len(X_tr)),
        "test_size":          int(len(X_te)),
        "n_estimators":       200,
        "test_ratio":         0.2,
        "random_state":       42,
        "label_counts":       vc.to_dict(),
    }
    with open(MET_PATH,"w",encoding="utf-8") as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    print(f"[SAVED] {MET_PATH}")
    print("\n[DONE] Training completed!")
    return model,metrics


if __name__=="__main__":
    train()
