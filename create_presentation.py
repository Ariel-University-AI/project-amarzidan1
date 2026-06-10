"""יצירת מצגת PowerPoint לפרויקט"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

NAVY   = RGBColor(0x1a,0x23,0x7e)
BLUE   = RGBColor(0x0d,0x47,0xa1)
GOLD   = RGBColor(0xf5,0xa6,0x23)
WHITE  = RGBColor(0xFF,0xFF,0xFF)
GRAY   = RGBColor(0xF5,0xF5,0xF5)
LGRAY  = RGBColor(0xEE,0xEE,0xEE)
RED    = RGBColor(0xe7,0x4c,0x3c)
GREEN  = RGBColor(0x2e,0xcc,0x71)
DGREEN = RGBColor(0x1b,0x5e,0x20)
ORANGE = RGBColor(0xe6,0x7e,0x22)
DARK   = RGBColor(0x2d,0x34,0x36)
PURPLE = RGBColor(0x6a,0x1b,0x9a)

def rect(slide,l,t,w,h,fill=None,line=None,lw=1.5):
    s=slide.shapes.add_shape(1,Inches(l),Inches(t),Inches(w),Inches(h))
    if fill: s.fill.solid(); s.fill.fore_color.rgb=fill
    else: s.fill.background()
    if line: s.line.color.rgb=line; s.line.width=Pt(lw)
    else: s.line.fill.background()
    return s

def txt(slide,text,l,t,w,h,size=14,bold=False,color=DARK,align=PP_ALIGN.RIGHT,wrap=True):
    tb=slide.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=tb.text_frame; tf.word_wrap=wrap
    p=tf.paragraphs[0]; p.alignment=align
    r=p.add_run(); r.text=text
    r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=color
    return tb

def bg(slide,color=GRAY):
    s=slide.shapes.add_shape(1,0,0,prs.slide_width,prs.slide_height)
    s.fill.solid(); s.fill.fore_color.rgb=color; s.line.fill.background()

def header(slide,title,sub=None):
    rect(slide,0,0,13.33,1.15,fill=NAVY)
    rect(slide,12.83,0,0.5,7.5,fill=GOLD)
    txt(slide,title,0.4,0.1,12,0.9,size=27,bold=True,color=WHITE,align=PP_ALIGN.RIGHT)
    if sub:
        txt(slide,sub,0.4,0.82,12,0.3,size=13,color=RGBColor(0xBB,0xDE,0xFB),align=PP_ALIGN.RIGHT)

BLANK=prs.slide_layouts[6]

# ════════════════════════════════════════════════════════════════════════
# 1 — שער
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
rect(sl,0,0,13.33,7.5,fill=NAVY)
rect(sl,12.83,0,0.5,7.5,fill=GOLD)
txt(sl,"מערכת ניתוח וחיזוי צמתים",0.5,1.1,12,1.3,size=38,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
txt(sl,"תמיכת החלטות למשרד התחבורה — ישראל 2021",0.5,2.6,12,0.7,size=20,color=RGBColor(0xBB,0xDE,0xFB),align=PP_ALIGN.CENTER)
rect(sl,3,3.55,7.3,0.06,fill=GOLD)
stats=[("11,554","תאונות אמיתיות"),("2,992","צמתים מנותחים"),("79.1%","Accuracy ML"),("9","חלופות הנדסיות")]
for i,(n,l) in enumerate(stats):
    x=0.9+i*3.0
    rect(sl,x,3.75,2.6,1.5,fill=RGBColor(0x1e,0x3a,0x8a))
    txt(sl,n,x,3.82,2.6,0.75,size=26,bold=True,color=GOLD,align=PP_ALIGN.CENTER)
    txt(sl,l,x,4.55,2.6,0.55,size=12,color=WHITE,align=PP_ALIGN.CENTER)
txt(sl,"גאודזיה מתמטית 444210  ·  אוניברסיטת אריאל  ·  ענף מדידות  ·  Demo Day 11/06/2026",0.5,6.6,12,0.55,size=13,color=RGBColor(0xBB,0xDE,0xFB),align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 2 — הבעיה
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"הבעיה")
txt(sl,'ישראל 2021: 11,554 תאונות בצמתים — 331 קטלניות, עלות שנתית ₪957M',0.4,1.35,12.4,0.6,size=19,bold=True,color=NAVY)
txt(sl,"השאלה: כיצד מחליטים אילו צמתים לשדרג? על סמך מה? מה הפתרון המתאים?",0.4,1.9,12.4,0.55,size=15,color=DARK)

probs=[("☠️ 331 קטלניות",RED,"תאונות קטלניות בצמתים בלבד"),
       ("💰 ₪957M",BLUE,"עלות שנתית לחברה"),
       ("❓ אין מענה",ORANGE,"אין מערכת להמלצות הנדסיות")]
for i,(t,c,s) in enumerate(probs):
    x=0.4+i*4.2
    rect(sl,x,2.65,3.9,1.3,fill=c)
    txt(sl,t,x+0.1,2.72,3.7,0.72,size=22,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    txt(sl,s,x+0.1,3.42,3.7,0.47,size=13,color=WHITE,align=PP_ALIGN.CENTER)

rect(sl,0.4,4.15,12.5,0.06,fill=GOLD)
txt(sl,"הפתרון שלנו:",0.4,4.3,3,0.55,size=15,bold=True,color=NAVY)
txt(sl,"מערכת שמנתחת כל צומת לפי נתוני CBS אמיתיים, מחשבת ציון סיכון, ומציגה 9 חלופות שדרוג עם חיזוי השפעה — הכל מבוסס תקנים הנדסיים בינלאומיים",
    3.4,4.3,9.5,0.55,size=14,color=DARK)

q_items=["לחיצה על צומת במפה → ניתוח מלא מיידי",
         "זיהוי אוטומטי של תשתית קיימת (כיכר/רמזור)",
         "ניתוח: האם הצומת טובה? האם נדרש שדרוג?",
         "9 חלופות מדורגות + השוואה + חיזוי לפני/אחרי"]
for i,q in enumerate(q_items):
    x=0.4 if i<2 else 6.8
    y=5.05+(i%2)*0.7
    rect(sl,x,y,6.1,0.6,fill=LGRAY,line=BLUE)
    txt(sl,f"✓ {q}",x+0.15,y+0.1,5.8,0.45,size=13,color=DARK)

# ════════════════════════════════════════════════════════════════════════
# 3 — הנתונים שמבססים על
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"על מה אנחנו מתבססים? — הנתונים")
txt(sl,"הנתון המרכזי: נתוני תאונות CBS PUF 2021 — 11,554 תאונות אמיתיות ב-45 עמודות",0.4,1.3,12.4,0.5,size=16,bold=True,color=NAVY)

cols=[("עמודות מרכזיות מ-CBS:",BLUE,[
    "HUMRAT_TEUNA — חומרת התאונה (קטלנית/קשה/קלה)",
    "SUG_TEUNA — סוג תאונה (חזיתית/עורפית/הולך רגל...)",
    "ZOMET_IRONI — קוד צומת ייחודי לזיהוי מדויק",
    "X,Y — קואורדינטות ITM → המרה ל-WGS84",
    "SHAA — שעת התאונה → זיהוי שעות עומס",
    "YOM_LAYLA — יום/לילה",
    "MEZEG_AVIR — מזג אוויר",
]),
("מה אנחנו מחשבים מ-CBS:",DGREEN,[
    "rear_pct = % תאונות עורפיות → מדד פקק",
    "peak_pct = % תאונות בשעות עומס 07-09, 16-19",
    "מדד עומס = rear_pct×0.6 + peak_pct×0.4",
    "fatal_pct, serious_pct → חומרת הצומת",
    "AADT מוערך (NCHRP 17-45): תאונות÷0.35",
    "LOS מוערך לפי HCM Table 19-8",
    "זמן השהייה לפי HCM Exhibit 19-1",
])]
for i,(title,col,items) in enumerate(cols):
    x=0.4+i*6.5
    rect(sl,x,1.95,6.1,0.55,fill=col)
    txt(sl,title,x+0.1,2.0,5.9,0.45,size=14,bold=True,color=WHITE)
    for j,item in enumerate(items):
        c=LGRAY if j%2==0 else WHITE
        rect(sl,x,2.52+j*0.55,6.1,0.52,fill=c)
        txt(sl,f"• {item}",x+0.15,2.57+j*0.55,5.85,0.45,size=12,color=DARK)

rect(sl,0.4,6.4,12.5,0.06,fill=GOLD)
txt(sl,"מקורות עלויות: נתיבי ישראל · מכרז משטרה 2025 · עיריית ירושלים 2023 · משרד התחבורה (₪3.5M לתאונה קטלנית)",
    0.4,6.52,12.5,0.55,size=12,bold=True,color=NAVY,align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 4 — איך יודעים צומת טובה/רעה
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"איך יודעים אם צומת טובה או בעייתית?","לפי תקני HCM 6th Edition + AASHTO")

txt(sl,"המערכת בודקת 5 קריטריונים הנדסיים לכל צומת:",0.4,1.3,12.4,0.5,size=16,bold=True,color=NAVY)

crits=[
    ("LOS A–C","LOS D–F","רמת שירות\n(Level of Service)","רמת שירות טובה מ-C = תנועה זורמת · D ומעלה = עומסים","HCM 6th Ed."),
    ("< 35 שנ'","> 40 שנ'","זמן השהייה","זמן ממוצע שרכב ממתין · מחושב מ-LOS + rear_pct","HCM Exhibit 19-1"),
    ("תורים < 100מ'","תורים > 200מ'","אורך תורים","תור ארוך = עומס · מחושב מהשהייה + rear_pct","HCM Queuing"),
    ("0 קטלניות\n≤1 קשות","≥1 קטלנית\nאו ≥2 קשות","תאונות חמורות","מספר תאונות קטלניות וקשות ב-3 שנים אחרונות","CBS PUF 2021"),
    ("עומס < 30%","עומס > 50%","מדד עומס תנועה","rear_pct×0.6 + peak_pct×0.4 · נגזר מנתוני CBS","CBS + HCM"),
]
headers=["קריטריון","מצב טוב ✅","מצב בעייתי ❌","הסבר","מקור"]
widths=[2.2,1.6,1.8,5.0,1.45]
# header row
xs=[0.35]; [xs.append(xs[-1]+w) for w in widths[:-1]]
rect(sl,0.35,1.9,12.6,0.48,fill=NAVY)
for j,(h,w) in enumerate(zip(headers,widths)):
    txt(sl,h,xs[j]+0.05,1.95,w-0.1,0.38,size=12,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
for i,(name,good,bad,desc,src) in enumerate(crits):
    y=2.42+i*0.75
    c=LGRAY if i%2==0 else WHITE
    rect(sl,0.35,y,12.6,0.72,fill=c)
    vals=[name,good,bad,desc,src]
    colors=[DARK,DGREEN,RED,DARK,BLUE]
    for j,(v,col,w) in enumerate(zip(vals,colors,widths)):
        txt(sl,v,xs[j]+0.05,y+0.08,w-0.1,0.6,size=11,color=col,align=PP_ALIGN.CENTER if j!=3 else PP_ALIGN.RIGHT)

rect(sl,0.35,6.25,12.6,0.55,fill=RGBColor(0xe3,0xf2,0xfd),line=BLUE)
txt(sl,"תוצאה: אם 3+ קריטריונים טובים → ✅ הצומת במצב טוב, אין צורך בשדרוג מיידי · אחרת → ⚠️ נדרשת בדיקה",
    0.5,6.33,12.3,0.4,size=13,bold=True,color=NAVY,align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 5 — מדוע משווים חלופות + ציון
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"למה משווים 9 חלופות? ואיך מחשבים ציון?","מנוע ניתוח הנדסי — HCM · AASHTO · FHWA · NCHRP")

txt(sl,"לא כל פתרון מתאים לכל צומת — תלוי ב: נפח תנועה (AADT) · חומרת תאונות · סוג בעיה (פקק/חזיתיות/לילה...)",
    0.4,1.3,12.4,0.5,size=15,bold=True,color=NAVY)

# 4 קריטריונים
weights=[("🦺 בטיחות","40%",RED,"הפחתת קטלניות (50%)\nהפחתת פציעות (30%)\nהפחתת קונפליקטים (20%)","FHWA, NCHRP 572"),
         ("🚗 תנועה","35%",BLUE,"שיפור LOS (30%)\nהפחתת השהייה (30%)\nעלייה בקיבולת (25%)\nהפחתת תורים (15%)","HCM 6th Edition"),
         ("💰 כלכלה","15%",ORANGE,"ROI = (חיסכון שנתי × 5)\n÷ עלות השקעה\nציון גבוה = ROI<5 שנים","נתיבי ישראל + CBS"),
         ("🌱 סביבה","10%",DGREEN,"הפחתת פליטות\nחיסכון דלק\nהפחתת רעש","FHWA Environmental")]
for i,(lbl,pct,col,items,src) in enumerate(weights):
    x=0.35+i*3.2
    rect(sl,x,1.95,3.05,0.65,fill=col)
    txt(sl,f"{lbl}  {pct}",x+0.05,2.0,2.95,0.55,size=16,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    rect(sl,x,2.62,3.05,2.5,fill=LGRAY,line=col)
    txt(sl,items,x+0.12,2.72,2.85,1.85,size=12,color=DARK)
    txt(sl,f"מקור: {src}",x+0.12,4.65,2.85,0.4,size=10,color=col)

rect(sl,0.35,5.15,12.6,0.06,fill=GOLD)
txt(sl,"נוסחת הציון:",0.35,5.3,3,0.5,size=14,bold=True,color=NAVY)
txt(sl,"ציון_כולל = בטיחות×40% + תנועה×35% + כלכלה×15% + סביבה×10%",
    3.3,5.3,9.4,0.5,size=15,bold=True,color=NAVY)

# מה ש-AADT עושה
txt(sl,"⚡ מה ש-AADT קובע:",0.35,5.95,3.5,0.45,size=13,bold=True,color=ORANGE)
aadt_items=["AADT<15K → כיכר עדיפה","AADT 15K–35K → כיכר דו-נתיבית/רמזור","AADT 35K–60K → רמזור חכם","AADT>60K → מחלף"]
for i,a in enumerate(aadt_items):
    x=0.35 if i<2 else 6.7
    y=5.95+(i%2)*0.55
    txt(sl,f"• {a}",x,y,6,0.5,size=13,color=DARK)

# ════════════════════════════════════════════════════════════════════════
# 6 — שלוש רמות
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"שלוש רמות ניתוח — M1 + M2 + M3")

levels=[
    ("📊 תיאורי — M1\n\"מה קורה?\"",BLUE,
     "• KPIs: תאונות + עומסי תנועה\n• מפה אינטראקטיבית — 2,992 צמתים\n• ציון סיכון משולב = תאונות(65%)+עומס(35%)\n• גרפי EDA: שעות, מחוז, סוגים\n• TOP10 צמתים עמוסים"),
    ("🔮 חיזוי — M2\n\"מה יקרה?\"",PURPLE,
     "• Random Forest ML (79.1% Accuracy)\n• חיזוי AADT, LOS, השהייה מ-CBS\n• חיזוי: לפני/אחרי שדרוג\n• טבלת השוואה: 15 פרמטרים\n• בדיקת תכנון צמתים חדשים"),
    ("🎯 מנחה — M3\n\"מה לעשות?\"",DGREEN,
     "• 9 חלופות הנדסיות מדורגות\n• ציון 0-100 לכל חלופה\n• בדיקת 'האם נדרש שדרוג?'\n• אפשרות 'שמור מצב קיים'\n• ROI + עלות + מקור תקן"),
]
for i,(title,col,body) in enumerate(levels):
    x=0.35+i*4.3
    rect(sl,x,1.4,4.1,1.1,fill=col)
    txt(sl,title,x+0.1,1.45,3.9,1.0,size=17,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    rect(sl,x,2.52,4.1,4.1,fill=LGRAY,line=col)
    txt(sl,body,x+0.15,2.62,3.85,3.9,size=14,color=DARK)

# חץ
txt(sl,"← ← ←",0.35,6.7,12.6,0.55,size=24,bold=True,color=GOLD,align=PP_ALIGN.CENTER)
txt(sl,"מתיאורי לחיזוי למנחה — זרימה רציפה, כל שלב מבוסס על הקודם",0.35,6.4,12.6,0.4,size=13,color=NAVY,align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 7 — המודל ML
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"מודל Machine Learning — Random Forest","train_test_split 80/20 · random_state=42")

rect(sl,0.35,1.35,6.2,2.9,fill=RGBColor(0xe3,0xf2,0xfd),line=BLUE)
txt(sl,"X — 11 פיצ'רים (קלט):",0.5,1.45,5.9,0.5,size=14,bold=True,color=NAVY)
feats="תאונות · fatal_pct · serious_pct · night_pct · rain_pct\npedestrian_pct · frontal_pct · rear_pct · rollover_pct\nhigh_speed · peak_pct ✨ (חדש: מדד עומס)"
txt(sl,feats,0.5,1.95,5.9,1.7,size=13,color=DARK)

rect(sl,6.8,1.35,6.1,2.9,fill=RGBColor(0xe8,0xf5,0xe9),line=DGREEN)
txt(sl,"y — התערבות מומלצת (7 סוגים):",6.95,1.45,5.8,0.5,size=14,bold=True,color=DGREEN)
targets="כיכר תנועה\nרמזור חכם\nמעבר חצייה + תאורה\nתאורת LED מוגברת\nמצלמת אכיפה\nפסי האטה\nהוספת נתיב נסיעה"
txt(sl,targets,6.95,1.95,5.8,1.7,size=13,color=DARK)

metrics=[("79.1%","Accuracy",BLUE),("74.5%","F1-Score",DGREEN),
         ("76.2%","Precision",ORANGE),("79.1%","Recall",RED)]
for i,(v,l,c) in enumerate(metrics):
    x=0.35+i*3.15
    rect(sl,x,4.45,2.95,1.3,fill=c)
    txt(sl,v,x+0.05,4.5,2.85,0.72,size=26,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    txt(sl,l,x+0.05,5.2,2.85,0.48,size=13,color=WHITE,align=PP_ALIGN.CENTER)

rect(sl,0.35,5.95,12.6,0.06,fill=GOLD)
txt(sl,"Feature הכי חשוב: rear_pct (41.6%) — תאונות עורפיות = מדד ישיר לצפיפות תנועה ועצירות תכופות",
    0.35,6.08,12.6,0.55,size=14,bold=True,color=NAVY,align=PP_ALIGN.CENTER)
txt(sl,"211 אתרים לאימון (≥3 תאונות) · Labels נוצרו מלוגיקה עסקית מבוססת ספרות בינלאומית",
    0.35,6.65,12.6,0.45,size=12,color=DARK,align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 8 — החיזוי: איך עובד
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"איך עובד החיזוי — מצב קיים מול לאחר שדרוג?")

txt(sl,"שאלה: אחרי שבוחרים פתרון — כיצד מחשבים מה יקרה? איך יודעים שהחיזוי אמין?",
    0.4,1.3,12.4,0.5,size=15,bold=True,color=NAVY)

# שלבים
steps=[
    ("1️⃣ ציון מצב קיים",BLUE,
     "מחושב מ-CBS:\nbטיחות = 100 − fatal×15 − serious×5\nתנועה = 100 − (LOS-1)×15 − השהייה×0.5\nציון = בטיחות×40% + תנועה×35% + 50×25%"),
    ("2️⃣ שיעורי הפחתה",ORANGE,
     "מבוססים על מחקרים:\nFHWA: כיכר → 90% קטלניות\nNCHRP 572: כיכר → 47% כל תאונה\nHCM: שיפור LOS לפי חלופה\nחישוב: ציון_אחרי = ציון_נוכחי + שיפור_יחסי"),
    ("3️⃣ השוואה 15 פרמטרים",DGREEN,
     "טבלה: מצב קיים vs לאחר שדרוג\nLOS · ציון בטיחות · ציון תנועה\nהשהייה · תורים · מדד עומס\nקטלניות · קשות · עלות שנתית\nROI · עלות השקעה"),
]
for i,(title,col,body) in enumerate(steps):
    x=0.35+i*4.3
    rect(sl,x,1.95,4.1,0.75,fill=col)
    txt(sl,title,x+0.1,2.0,3.9,0.65,size=15,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    rect(sl,x,2.72,4.1,2.8,fill=LGRAY,line=col)
    txt(sl,body,x+0.15,2.82,3.85,2.6,size=12,color=DARK)

# שורת validaton
rect(sl,0.35,5.7,12.6,0.06,fill=GOLD)
txt(sl,"⚡ אמינות החיזוי:",0.35,5.85,3.0,0.5,size=14,bold=True,color=ORANGE)
val_items=["שיעורי הפחתה מ-FHWA/NCHRP — מבוסס מאות כיכרות בארה\"ב",
           "ציון לפני ≤ ציון אחרי תמיד (נוסחה עקבית)",
           "אם AADT > קיבולת חלופה → אזהרה אוטומטית"]
for i,v in enumerate(val_items):
    x=3.5 if i<2 else 0.35
    y=5.85+(0 if i<1 else 0.52*i)
    txt(sl,f"✓ {v}",x,y,9.4 if i<2 else 12.6,0.48,size=13,color=DARK)

# ════════════════════════════════════════════════════════════════════════
# 9 — תוצאות
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"תוצאות ותובנות מרכזיות")

findings=[
    ("🔴 15 צמתים בסיכון גבוה",RED,"ציון ≥60/100 — מצריכים שדרוג מיידי\nרובם בצפון תל אביב וחיפה"),
    ("🔵 כיכר — הפתרון הטוב ביותר",BLUE,"90% הפחתת קטלניות (FHWA)\nROI מהיר · עלות נמוכה ₪500K"),
    ("🚗 3,035 תאונות עורפיות",ORANGE,"60.5% מהתאונות — פקק כרוני\nFeature מס' 1 במודל: rear_pct"),
    ("⏰ שעות 07–09, 16–19",PURPLE,"שעות שיא: 66 תאונות נוספות\nמדד עומס = peak_pct×40%"),
    ("💰 ₪957M עלות שנתית",RED,"ניתן לחסוך עשרות מיליונים\nROI ממוצע: 3–10 שנים"),
    ("✅ מערכת תומכת החלטות",DGREEN,"אין החלטה אוטומטית\nמספקת בסיס הנדסי לתעדוף"),
]
for i,(t,c,s) in enumerate(findings):
    row,col=divmod(i,2)
    x=0.35+col*6.5; y=1.4+row*1.65
    rect(sl,x,y,6.1,1.45,fill=LGRAY,line=c)
    txt(sl,t,x+0.15,y+0.1,5.8,0.6,size=14,bold=True,color=c)
    txt(sl,s,x+0.15,y+0.72,5.8,0.65,size=13,color=DARK)

rect(sl,0.35,6.4,12.6,0.06,fill=GOLD)
txt(sl,"המערכת לא מחליפה סקר שטח — מספקת סדרי עדיפויות מבוססי נתונים להחלטה ראשונית",
    0.35,6.53,12.6,0.55,size=13,bold=True,color=NAVY,align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════
# 10 — מגבלות + המשך
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
bg(sl,WHITE); header(sl,"מגבלות — מה עוד לא יודעים ואיך לשפר?")

txt(sl,"השאלה: איך יודעים שהחיזוי נכון? — זו המגבלה המרכזית:",
    0.4,1.3,12.4,0.5,size=16,bold=True,color=RED)

limits=[
    ("❓ AADT לא ידוע","AADT מוערך מנוסחת NCHRP, לא ממדידה ישירה\n→ שגיאה אפשרית בחיזוי קיבולת ו-LOS","נתוני ספירת תנועה מ-data.gov.il"),
    ("❓ LOS/השהייה לא נמדדו","מוערכים מ-CBS — לא מסקר שטח אמיתי\n→ האם הצומת באמת LOS D? לא בטוח","HERE/TomTom API לנתונים בזמן אמת"),
    ("❓ נתון שנה אחת","CBS 2021 בלבד — לא ניתן לזהות מגמות\nהאם הצומת מחמירה או משתפרת?","נתונים 2019–2023 לניתוח מגמות"),
    ("❓ OSM לא שלם","לא כל הצמתים ממופים ב-OpenStreetMap\n→ זיהוי תשתית קיימת אינו תמיד מדויק","ולידציה ידנית + Google Maps API"),
]
for i,(t,s,fix) in enumerate(limits):
    row,col=divmod(i,2)
    x=0.35+col*6.5; y=2.0+row*1.55
    rect(sl,x,y,6.1,1.4,fill=RGBColor(0xff,0xf3,0xe0),line=ORANGE)
    txt(sl,t,x+0.15,y+0.08,5.8,0.5,size=13,bold=True,color=ORANGE)
    txt(sl,s,x+0.15,y+0.58,5.8,0.6,size=12,color=DARK)
    txt(sl,f"🔧 {fix}",x+0.15,y+1.1,5.8,0.25,size=11,color=BLUE)

rect(sl,0.35,5.3,12.6,0.06,fill=GOLD)
txt(sl,"כיצד ולידציה? — השוואת צמתים שכבר שודרגו: מציאת נתוני לפני/אחרי ממאגרי נתיבי ישראל",
    0.35,5.45,12.6,0.5,size=13,bold=True,color=NAVY,align=PP_ALIGN.CENTER)

futures=["ספירות תנועה מ-data.gov.il → AADT אמיתי",
         "HERE API → LOS בזמן אמת לכל צומת",
         "נתוני לפני/אחרי שדרוג → ולידציה אמיתית",
         "נתונים רב-שנתיים → זיהוי מגמות"]
for i,f in enumerate(futures):
    x=0.35 if i<2 else 6.8
    y=6.1+(i%2)*0.55
    txt(sl,f"→ {f}",x,y,6.2,0.5,size=13,color=DGREEN)

# ════════════════════════════════════════════════════════════════════════
# 11 — סיום
# ════════════════════════════════════════════════════════════════════════
sl=prs.slides.add_slide(BLANK)
rect(sl,0,0,13.33,7.5,fill=NAVY)
rect(sl,12.83,0,0.5,7.5,fill=GOLD)
txt(sl,"תודה!",0.5,1.5,12,1.5,size=56,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
rect(sl,2.5,3.3,8.3,0.07,fill=GOLD)
txt(sl,"מערכת ניתוח וחיזוי צמתים — ישראל 2021",0.5,3.55,12,0.7,size=20,color=RGBColor(0xBB,0xDE,0xFB),align=PP_ALIGN.CENTER)
txt(sl,"גאודזיה מתמטית 444210  ·  אוניברסיטת אריאל  ·  Demo Day 11/06/2026",0.5,4.35,12,0.55,size=14,color=RGBColor(0xBB,0xDE,0xFB),align=PP_ALIGN.CENTER)
txt(sl,"❓  שאלות?",0.5,5.3,12,0.8,size=28,bold=True,color=GOLD,align=PP_ALIGN.CENTER)

prs.save("presentation.pptx")
print("נשמר: presentation.pptx  (11 שקופיות)")
