"""
train_model.py — אימון מודל לחיזוי ההתערבות המומלצת לצומת
═══════════════════════════════════════════════════════════════
שלב 1 — החלטה:
  y  (מה לחזות):  ההתערבות המומלצת לצומת — כיכר / רמזור / מצלמה / וכו'
  X  (פיצ'רים):   פרופיל תאונות לפי אתר:
                   % תאונות קטלניות, % קשות, % לילה, % גשם,
                   % הולכי רגל, % חזיתיות, % אחוריות, % התהפכות,
                   מהירות גבוהה (בוליאני), מספר תאונות כולל
  סוג מודל:       Classification (Random Forest)

Label Generation:
  אין עמודת "התערבות" בנתונים — Label נוצר מהלוגיקה העסקית
  הקיימת ב-_rank_interventions (מבוסס ספרות בינלאומית)

הרצה:
    python train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Decode tables (CBS PUF) ───────────────────────────────────────────────────
_SEV    = {1: "קטלנית", 2: "קשה", 3: "קלה"}
_ROAD   = {1: "עירוני - בצומת", 2: "עירוני - לא בצומת",
           3: "בין-עירוני - בצומת", 4: "בין-עירוני - לא בצומת",
           5: "חניון / כיכר", 9: "אחר"}
_WTHR   = {1: "בהיר", 2: "גשם קל", 3: "גשם", 4: "ערפל",
           5: "חול", 7: "שלג", 8: "סופה", 9: "אחר"}
_SURF   = {1: "יבש", 2: "רטוב", 3: "קפוא", 4: "שלג", 9: "אחר"}
_DNTM   = {1: "יום", 5: "לילה"}
_DIST   = {1: "ירושלים", 2: "צפון", 3: "חיפה",
           4: "מרכז", 5: "תל אביב", 6: "דרום", 7: 'יו"ש'}
_SPD    = {1: "30", 2: "40", 3: "50", 4: "60", 5: "70",
           6: "80", 7: "90", 8: "100", 9: "110"}
_ACCTYP = {1: "חזיתית", 2: "אחורית", 3: "צידית", 4: "הולך רגל",
           5: "התהפכות", 6: "פגיעה בעמוד", 7: "נפילה מרכב", 8: "אחר"}
_MONTHS = {1: "ינואר", 2: "פברואר", 3: "מרץ", 4: "אפריל", 5: "מאי",
           6: "יוני", 7: "יולי", 8: "אוגוסט", 9: "ספטמבר",
           10: "אוקטובר", 11: "נובמבר", 12: "דצמבר"}

# ── Intervention logic (same as eda_app.py) ───────────────────────────────────
INTERVENTIONS = {
    "כיכר תנועה":              {"fatal": 0.82, "serious": 0.55,
                                 "tags": {"חזיתית", "התהפכות", "מהירות_גבוהה"}},
    "רמזור חכם":               {"fatal": 0.45, "serious": 0.32,
                                 "tags": {"הולך רגל", "עירוני", "עומס"}},
    "מעבר חצייה מוגן + תאורה": {"fatal": 0.55, "serious": 0.38,
                                 "tags": {"הולך רגל", "עירוני"}},
    "תאורת לד מוגברת":         {"fatal": 0.32, "serious": 0.22,
                                 "tags": {"לילה", "ערפל"}},
    "מצלמת אכיפה + מד-מהירות": {"fatal": 0.25, "serious": 0.18,
                                 "tags": {"מהירות_גבוהה", "בין-עירוני"}},
    "פסי האטה / מוקפצים":      {"fatal": 0.30, "serious": 0.25,
                                 "tags": {"אחורית", "מהירות_בינונית"}},
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "data", "accidents_israel_2021_raw.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.pkl")
ENC_PATH     = os.path.join(MODEL_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SITES_PATH   = os.path.join(MODEL_DIR, "sites_cache.pkl")

# ── Feature columns (per-site profile) ───────────────────────────────────────
FEAT_COLS = [
    "תאונות",          # מספר תאונות כולל
    "fatal_pct",       # % קטלניות
    "serious_pct",     # % קשות
    "night_pct",       # % לילה
    "rain_pct",        # % גשם
    "pedestrian_pct",  # % הולכי רגל
    "frontal_pct",     # % חזיתיות
    "rear_pct",        # % אחוריות
    "rollover_pct",    # % התהפכות
    "high_speed",      # מהירות גבוהה (0/1)
]
TARGET = "label_intervention"


def load_and_decode() -> pd.DataFrame:
    """Load raw CBS data and decode all columns."""
    print("[*] Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    num_cols = [
        "HUMRAT_TEUNA", "SUG_DEREH", "SUG_TEUNA", "MEZEG_AVIR", "PNE_KVISH",
        "YOM_LAYLA", "MAHOZ", "MEHIRUT_MUTERET", "SEMEL_YISHUV", "KVISH1",
        "HODESH_TEUNA", "SHAA", "X", "Y",
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
    df["מחוז"]         = df["MAHOZ"].map(_DIST)
    df["מהירות_מותרת"] = df["MEHIRUT_MUTERET"].map(_SPD)

    # Location
    df["כביש"] = df["KVISH1"].where(df["KVISH1"].notna() & (df["KVISH1"] > 0))
    df["מיקום"] = df.apply(
        lambda r: f"כביש {int(r['כביש'])}" if pd.notna(r["כביש"]) else "לא ידוע",
        axis=1,
    )
    df["אתר"] = df["מיקום"] + " – " + df["סוג_דרך"].fillna("לא ידוע")

    print(f"    -> {len(df):,} rows loaded")
    return df


def build_site_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-site accident profile features (X) and labels (y)."""
    print("[*] Building per-site profiles...")

    records = []
    for site, grp in df.groupby("אתר"):
        n = max(len(grp), 1)
        spd_mode = grp["מהירות_מותרת"].mode()
        spd_val  = spd_mode.iloc[0] if len(spd_mode) else "50"
        road_mode = grp["סוג_דרך"].mode()
        road_type = road_mode.iloc[0] if len(road_mode) else ""

        prof = {
            "אתר":           site,
            "תאונות":        len(grp),
            "fatal_pct":     (grp["חומרת_תאונה"] == "קטלנית").sum() / n,
            "serious_pct":   (grp["חומרת_תאונה"] == "קשה").sum() / n,
            "night_pct":     (grp["חלק_יממה"] == "לילה").sum() / n,
            "rain_pct":      grp["מזג_אוויר"].isin(["גשם", "גשם קל"]).sum() / n,
            "pedestrian_pct":(grp["סוג_תאונה"] == "הולך רגל").sum() / n,
            "frontal_pct":   (grp["סוג_תאונה"] == "חזיתית").sum() / n,
            "rear_pct":      (grp["סוג_תאונה"] == "אחורית").sum() / n,
            "rollover_pct":  (grp["סוג_תאונה"] == "התהפכות").sum() / n,
            "high_speed":    1 if str(spd_val) in {"70","80","90","100","110"} else 0,
            "road_type":     road_type,
        }

        # ── Generate label using intervention scoring logic ──
        scores = {}
        for name, info in INTERVENTIONS.items():
            s = (
                prof["fatal_pct"]   * info["fatal"]   +
                prof["serious_pct"] * info["serious"]
            ) * prof["תאונות"] * 10

            if prof["pedestrian_pct"] > 0.12 and "הולך רגל"     in info["tags"]: s += 25
            if prof["frontal_pct"]    > 0.18 and "חזיתית"        in info["tags"]: s += 25
            if prof["night_pct"]      > 0.35 and "לילה"          in info["tags"]: s += 20
            if prof["high_speed"]            and "מהירות_גבוהה"  in info["tags"]: s += 20
            if prof["rear_pct"]       > 0.20 and "אחורית"        in info["tags"]: s += 15
            if "עירוני" in road_type         and "עירוני"        in info["tags"]: s += 10
            if "בין-עירוני" in road_type    and "בין-עירוני"    in info["tags"]: s += 10
            scores[name] = max(0.0, s)

        prof[TARGET] = max(scores, key=scores.get)
        records.append(prof)

    sites_df = pd.DataFrame(records)
    # Filter sites with at least 3 accidents for reliability
    sites_df = sites_df[sites_df["תאונות"] >= 3].reset_index(drop=True)
    print(f"    -> {len(sites_df):,} sites with 3+ accidents")
    return sites_df


def train():
    """Train model, print metrics, save everything."""
    df       = load_and_decode()
    sites_df = build_site_profile(df)

    # ── Features & target ──
    X = sites_df[FEAT_COLS].values
    y = sites_df[TARGET].values

    # ── Train / Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[*] Split: Train={len(X_train):,} | Test={len(X_test):,}")

    # ── Train ──
    print("\n[*] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    labels_order = sorted(INTERVENTIONS.keys())

    print("\n" + "=" * 60)
    print("[METRICS] Test Set Performance")
    print("=" * 60)
    print(f"   Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[Confusion Matrix]")
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    print(cm_df.to_string())

    print("\n[Feature Importance]")
    for feat, imp in sorted(zip(FEAT_COLS, model.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"   {feat:20s}  {imp:.4f}  {bar}")

    # ── Label distribution ──
    print("\n[Label Distribution]")
    vc = pd.Series(y).value_counts()
    for label, count in vc.items():
        print(f"   {label}: {count} ({count/len(y)*100:.1f}%)")

    # ── Save ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\n[SAVED] Model -> {MODEL_PATH}")

    # Save label encoder for classes
    le_target = LabelEncoder()
    le_target.fit(y)
    joblib.dump(le_target, ENC_PATH)
    print(f"[SAVED] Encoder -> {ENC_PATH}")

    # Save sites cache for fast app loading
    joblib.dump(sites_df, SITES_PATH)
    print(f"[SAVED] Sites cache -> {SITES_PATH}")

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy":               round(acc, 4),
        "f1_score":               round(f1, 4),
        "precision":              round(prec, 4),
        "recall":                 round(rec, 4),
        "classification_report":  report_dict,
        "confusion_matrix":       cm.tolist(),
        "labels":                 labels_order,
        "feature_importance":     dict(zip(FEAT_COLS,
                                           model.feature_importances_.tolist())),
        "feature_columns":        FEAT_COLS,
        "target_column":          TARGET,
        "train_size":             int(len(X_train)),
        "test_size":              int(len(X_test)),
        "n_estimators":           200,
        "test_ratio":             0.2,
        "random_state":           42,
        "label_counts":           vc.to_dict(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Metrics -> {METRICS_PATH}")

    print("\n[DONE] Training completed successfully!")
    return model, metrics


if __name__ == "__main__":
    train()
