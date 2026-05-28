# חיזוי מוקדי סיכון בתשתיות תחבורה בישראל 🚗📊

## תיאור הפרויקט
מערכת תומכת-החלטה **למנהלים בתחום תשתיות תחבורה** — מזהה ומדרגת אתרי סיכון (צמתים, מקטעי כביש) ברחבי ישראל לפי הסתברות לתאונה חמורה, וממליצה על ההתערבות התשתיתית המתאימה.

המערכת מאומנת על **11,554 תאונות אמיתיות** (נתוני למ"ס PUF 2021) באמצעות מודל Random Forest, ומייצרת לכל מנהל:
- רשימת עדיפויות של אתרי תשתית לפי ציון סיכון
- **חיזוי אוטומטי**: מה צריך להיות בצומת — כיכר תנועה / רמזור חכם / מצלמת אכיפה / וכו'

---

## תכונות מרכזיות (Features)

### 🤖 מנוע ML — חיזוי התערבות מומלצת לצומת
- **y (מה מחזים):** ההתערבות המומלצת — כיכר / רמזור / מצלמה / תאורה / פסי האטה / מעבר חצייה
- **X (פיצ'רים):** פרופיל תאונות האתר — % קטלניות, % לילה, % הולכי רגל, % חזיתיות, מהירות, מספר תאונות
- **מודל:** Random Forest Classification (n_estimators=200)
- **חלוקה:** train_test_split, test_size=0.2, random_state=42
- **שמירה:** `joblib.dump(model, 'models/model.pkl')`

### 🎯 ממשק מנהל — דירוג אתרי תשתית
- **טבלת עדיפויות** — כל האתרים מדורגים לפי ציון סיכון עם 🔴/🟡/🟢
- **עמודת התערבות מומלצת** — ML חוזה מה צריך להיות בכל צומת
- **מפת חום** גאוגרפית של ריכוזי סיכון
- **ייצוא CSV** לדיווח ותכנון תקציבי

### 📊 ניתוח חקרני (EDA)
- **KPI cards** — סה"כ תאונות, נפגעים, תאונות קטלניות וקשות
- **ניתוח זמני** — לפי שעה, יום בשבוע, חודש
- **מפה אינטראקטיבית** — פיזור תאונות לפי חומרה
- **סינון** לפי מיקום, חומרה, מזג אוויר

### 🔮 חיזוי פתרון לצומת (Rule-Based)
- בחירת צומת במפה → פרופיל תאונות → המלצת פתרון מדורגת
- הצגת ביטחון, עלות, והפחתת תאונות צפויה

---

## מודל ML — פרטים טכניים

| פרמטר | ערך |
|-------|-----|
| סוג מודל | Random Forest Classifier |
| n_estimators | 200 |
| max_depth | 10 |
| test_size | 0.2 |
| random_state | 42 |
| Feature columns | תאונות, fatal_pct, serious_pct, night_pct, rain_pct, pedestrian_pct, frontal_pct, rear_pct, rollover_pct, high_speed |
| Target (y) | label_intervention — ההתערבות המומלצת |
| Label generation | לוגיקה עסקית מבוססת ספרות בינלאומית |

### הרצת האימון
```bash
python train_model.py
```
קבצי הפלט נשמרים ב-`models/`:
```
models/
├── model.pkl       # המודל המאומן
├── encoders.pkl    # LabelEncoder
├── metrics.json    # accuracy, F1, confusion matrix, feature importance
└── sites_cache.pkl # פרופיל אתרים מחושב מראש
```

---

## מקורות נתונים (Data Sources)

| קובץ | מקור | שורות | תיאור |
|------|------|--------|--------|
| `data/accidents_israel_2021_raw.csv` | [למ"ס — PUF 2021](https://data.gov.il/he/dataset/2021-puf) | 11,554 | נתוני תאונות אמיתיים, גולמיים |
| `israel_road_accidents_simulated.csv` | סימולטיבי | ~1,000 | לבדיקות ופיתוח |

---

## טכנולוגיות (Tech Stack)

| קטגוריה | טכנולוגיה |
|---------|-----------|
| שפת תכנות | Python 3.12 |
| אפליקציית Web | Streamlit |
| ניתוח נתונים | Pandas, NumPy |
| ויזואליזציה | Plotly Express |
| למידת מכונה | Scikit-learn (Random Forest) |
| שמירת מודל | Joblib |
| המרת קואורדינטות | pyproj (ITM → WGS84) |
| מיפוי | Plotly Mapbox |
| ייצוא דוחות | xlsxwriter |

---

## התקנה והרצה (Installation & Usage)

### התקנה מהירה

```bash
git clone <repo-url>
cd <repo-dir>
pip install -r requirements.txt
```

### שלב 1 — אימון המודל (פעם אחת)

```bash
python train_model.py
```

פלט לדוגמה:
```
📂 טוען נתונים...
🏗️  בונה פרופיל לכל אתר...
🔀 חלוקת נתונים: Train=X | Test=Y
🤖 מאמן Random Forest...
📈  Accuracy: 0.8500 (85.0%)
💾 מודל נשמר: models/model.pkl
```

### שלב 2 — הרצת האפליקציה

```bash
streamlit run eda_app.py
```

או לחץ פעמיים על `run_app.bat` (Windows בלבד).

האפליקציה תיפתח: **http://localhost:8501**

---

## מבנה הפרויקט (Project Structure)

```
├── data/
│   └── accidents_israel_2021_raw.csv   # נתוני תאונות — למ"ס PUF 2021
├── models/                             # נוצרת לאחר הרצת train_model.py
│   ├── model.pkl                       # מודל Random Forest מאומן
│   ├── encoders.pkl                    # LabelEncoder
│   ├── metrics.json                    # מטריקות ביצוע
│   └── sites_cache.pkl                 # פרופיל אתרים
├── eda_app.py                          # אפליקציית Streamlit (4 טאבים)
├── train_model.py                      # אימון מודל + שמירה
├── download_accidents.py               # הורדת נתוני CBS
├── data_preprocessing.ipynb            # עיבוד נתונים
├── requirements.txt                    # תלויות
├── run_app.bat                         # הרצה מהירה (Windows)
└── README.md                           # קובץ זה
```

---

## שלבי פיתוח

### ✅ הושלם
- EDA dashboard עם KPIs, גרפים, מפה אינטראקטיבית
- הורדת נתוני תאונות אמיתיים מ-data.gov.il (11,554 שורות, CBS PUF 2021)
- **מנוע ML (Random Forest)** — חיזוי התערבות מומלצת לצומת
- `train_model.py` — אימון + `joblib.dump` + מטריקות
- ממשק מנהל — טבלת דירוג + עמודת התערבות מומלצת + מפת חום
- טאב ML — Confusion Matrix, Feature Importance, חיזוי אינטראקטיבי

### 🔄 בפיתוח
- פענוח קודי CBS ומיפוי קואורדינטות מדויק (ITM → WGS84)
- שיפור מודל עם Cross-Validation ו-Hyperparameter Tuning

### 🔮 תוכנית עד Demo Day

| תאריך | משימה |
|-------|-------|
| **שבוע 1** | הרצת `train_model.py` + וידוא מטריקות + שיפור Feature Engineering |
| **שבוע 2** | שיפור UI — עיצוב, צבעים, Responsive |
| **שבוע 3** | הוספת Cross-Validation + השוואת מודלים (XGBoost vs RF) |
| **Demo Day** | הצגה מלאה: EDA → ממשק מנהל → חיזוי ML אינטראקטיבי |

---

**פותח במסגרת לימודי בינה מלאכותית (AI) — אוניברסיטת אריאל**
