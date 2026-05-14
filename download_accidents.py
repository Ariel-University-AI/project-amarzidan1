"""
Download CBS road accident data (2021 PUF) from data.gov.il and convert to CSV.
Sources:
  - Accidents: resource a5e15f5c-7583-45d0-9b31-3dd94bf0edd2  (11,554 records)
  - City names: resource 5c78e9fa-c2e2-4771-93ff-7f400a12f7ba  (1,306 settlements)
"""
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Decode tables  (based on CBS PUF documentation)
# ---------------------------------------------------------------------------
SEVERITY = {1: "קטלנית", 2: "קשה", 3: "קלה"}

ROAD_TYPE = {
    1: "עירוני - בצומת",
    2: "עירוני - לא בצומת",
    3: "בין-עירוני - בצומת",
    4: "בין-עירוני - לא בצומת",
    5: "חניון / כיכר",
    9: "אחר",
}

ACCIDENT_TYPE = {
    1: "התנגשות חזיתית",
    2: "התנגשות אחורית",
    3: "התנגשות צידית",
    4: "פגיעה בהולך רגל",
    5: "התהפכות",
    6: "פגיעה בעמוד / גדר",
    7: "נפילה מרכב",
    8: "אחר",
}

WEATHER = {
    1: "בהיר",
    2: "גשם קל",
    3: "גשם",
    4: "ערפל",
    5: "חול / אבק",
    7: "שלג",
    8: "סופה",
    9: "אחר",
}

ROAD_SURFACE = {
    1: "יבש",
    2: "רטוב",
    3: "עטוף / קפוא",
    4: "עם שלג",
    9: "אחר",
}

DAY_NIGHT = {1: "יום", 5: "לילה"}

DAY_OF_WEEK = {
    1: "ראשון",
    2: "שני",
    3: "שלישי",
    4: "רביעי",
    5: "חמישי",
    6: "שישי",
    7: "שבת",
}

DISTRICT = {
    1: "ירושלים",
    2: "צפון",
    3: "חיפה",
    4: "מרכז",
    5: "תל אביב",
    6: "דרום",
    7: "יהודה ושומרון",
}

SPEED_LIMIT = {
    1: "30",
    2: "40",
    3: "50",
    4: "60",
    5: "70",
    6: "80",
    7: "90",
    8: "100",
    9: "110",
}

# ---------------------------------------------------------------------------
# ITM (EPSG:2039)  →  WGS84 (lat/lon)
# ---------------------------------------------------------------------------
def itm_to_wgs84_series(x_col, y_col):
    """Vectorised conversion. Falls back to rough linear approx if pyproj absent."""
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)
        lon_arr, lat_arr = tr.transform(x_col.values, y_col.values)
        return pd.Series(lat_arr.round(6)), pd.Series(lon_arr.round(6))
    except Exception:
        # Rough linear approximation (~1 km accuracy, good enough for visualisation)
        lat = ((y_col - 626907) / 111320 + 31.5).round(5)
        lon = ((x_col - 219529) / (111320 * 0.857) + 35.21).round(5)
        return lat, lon


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
BASE = "https://data.gov.il/api/3/action/datastore_search"


def download_resource(resource_id, batch=1000):
    records, offset = [], 0
    total = None
    while True:
        r = requests.get(
            BASE,
            params={"resource_id": resource_id, "limit": batch, "offset": offset},
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()["result"]
        if total is None:
            total = result["total"]
            print(f"  total: {total:,}")
        batch_records = result["records"]
        records.extend(batch_records)
        offset += batch
        print(f"  fetched {min(offset, total):,} / {total:,}", end="\r")
        if offset >= total or not batch_records:
            break
    print()
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("1/3  Downloading city names lookup …")
    cities_raw = download_resource("5c78e9fa-c2e2-4771-93ff-7f400a12f7ba")
    cities = pd.DataFrame(cities_raw)[["סמל_ישוב", "שם_ישוב"]].copy()
    cities["סמל_ישוב"] = pd.to_numeric(cities["סמל_ישוב"], errors="coerce")
    city_map = dict(zip(cities["סמל_ישוב"], cities["שם_ישוב"]))

    print("2/3  Downloading accident records …")
    acc_raw = download_resource("a5e15f5c-7583-45d0-9b31-3dd94bf0edd2")
    df = pd.DataFrame(acc_raw)

    print("3/3  Decoding and converting coordinates …")

    # Numeric coercion for key columns
    for col in ["HUMRAT_TEUNA", "SUG_DEREH", "SUG_TEUNA", "MEZEG_AVIR",
                "PNE_KVISH", "YOM_LAYLA", "YOM_BASHAVUA", "MAHOZ",
                "MEHIRUT_MUTERET", "SEMEL_YISHUV", "KVISH1",
                "SHNAT_TEUNA", "HODESH_TEUNA", "SHAA", "X", "Y"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # Derived time columns
    # SHAA is a 15-minute slot index (0–95); slot → hour
    df["שעה"] = (df["SHAA"] // 4).clip(0, 23).astype("Int64")

    # Coordinates
    mask = df["X"].notna() & df["Y"].notna()
    df["קו_רוחב"] = pd.NA
    df["קו_אורך"] = pd.NA
    lat, lon = itm_to_wgs84_series(df.loc[mask, "X"], df.loc[mask, "Y"])
    df.loc[mask, "קו_רוחב"] = lat.values
    df.loc[mask, "קו_אורך"] = lon.values

    # City name (urban) or road number label (inter-urban)
    df["שם_ישוב"] = df["SEMEL_YISHUV"].map(city_map)
    df["כביש"] = df["KVISH1"].where(df["KVISH1"].notna()).astype("Int64")

    # Location label  (used as "אתר" in hotspot model)
    df["מיקום"] = df.apply(
        lambda r: r["שם_ישוב"]
        if pd.notna(r["שם_ישוב"]) and r["שם_ישוב"]
        else (f"כביש {int(r['כביש'])}" if pd.notna(r["כביש"]) else "לא ידוע"),
        axis=1,
    )

    # Decoded columns
    df["חומרת_תאונה"]  = df["HUMRAT_TEUNA"].map(SEVERITY)
    df["סוג_דרך"]      = df["SUG_DEREH"].map(ROAD_TYPE)
    df["סוג_תאונה"]    = df["SUG_TEUNA"].map(ACCIDENT_TYPE)
    df["מזג_אוויר"]    = df["MEZEG_AVIR"].map(WEATHER)
    df["מצב_כביש"]     = df["PNE_KVISH"].map(ROAD_SURFACE)
    df["חלק_יממה"]     = df["YOM_LAYLA"].map(DAY_NIGHT)
    df["יום_בשבוע"]    = df["YOM_BASHAVUA"].map(DAY_OF_WEEK)
    df["מחוז"]         = df["MAHOZ"].map(DISTRICT)
    df["מהירות_מותרת"] = df["MEHIRUT_MUTERET"].map(SPEED_LIMIT)

    # Final output columns
    out = df[[
        "pk_teuna_fikt",
        "SHNAT_TEUNA", "HODESH_TEUNA", "שעה",
        "חלק_יממה", "יום_בשבוע",
        "חומרת_תאונה", "סוג_תאונה", "סוג_דרך",
        "מזג_אוויר", "מצב_כביש", "מהירות_מותרת",
        "מיקום", "שם_ישוב", "כביש", "מחוז",
        "קו_רוחב", "קו_אורך",
    ]].copy()

    out.columns = [
        "מזהה_תאונה",
        "שנה", "חודש", "שעה",
        "חלק_יממה", "יום_בשבוע",
        "חומרת_תאונה", "סוג_תאונה", "סוג_דרך",
        "מזג_אוויר", "מצב_כביש", "מהירות_מותרת",
        "מיקום", "שם_ישוב", "כביש", "מחוז",
        "קו_רוחב", "קו_אורך",
    ]

    out_path = "data/accidents_israel_2021_real.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out):,} records  →  {out_path}")
    print(out[["חומרת_תאונה", "סוג_דרך", "מזג_אוויר", "מיקום", "קו_רוחב", "קו_אורך"]].head(5).to_string())


if __name__ == "__main__":
    main()
