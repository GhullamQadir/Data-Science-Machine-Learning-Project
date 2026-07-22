import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib

RAW_DATA_DIR = os.path.join("data", "Raw_data")
xlsx_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith((".xlsx", ".xls"))]
if not xlsx_files:
    print("[ERROR] No Excel files in data/Raw_data/")
    sys.exit(1)

file_path = os.path.join(RAW_DATA_DIR, xlsx_files[0])
print(f"[INFO] Loading: {file_path}")
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names
bat_sheets   = [s for s in sheet_names if "_BAT"   in s.upper()]
bowl_sheets  = [s for s in sheet_names if "_BOWL"  in s.upper()]
field_sheets = [s for s in sheet_names if "_FIELD" in s.upper()]
print(f"[INFO] BAT={len(bat_sheets)}, BOWL={len(bowl_sheets)}, FIELD={len(field_sheets)}")

def find_header_row(raw_df):
    for i, row in raw_df.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if "player" in vals:
            return i
    return 0

def read_sheet(name):
    raw = pd.read_excel(xl, sheet_name=name, header=None)
    hrow = find_header_row(raw)
    df = pd.read_excel(xl, sheet_name=name, header=hrow)
    df.columns = [str(c).strip().replace(" ","_") for c in df.columns]
    if "Player" in df.columns:
        df = df[df["Player"].notna()]
        df = df[~df["Player"].astype(str).str.strip().str.lower().isin(["player","nan",""])]
    return df.reset_index(drop=True)

bat_frames = []
for s in bat_sheets:
    try:
        df = read_sheet(s)
        if "Player" not in df.columns: continue
        cm = {}
        for col in df.columns:
            cl = col.lower()
            if cl=="player": cm[col]="Player_Name"
            elif cl=="country": cm[col]="Country"
            elif cl in ("runs","run","r"): cm[col]="Runs"
            elif cl in ("avg","average","batting_avg"): cm[col]="Batting_Avg"
            elif cl in ("sr","strike_rate","bsr"): cm[col]="Strike_Rate"
            elif cl in ("mat","matches","m"): cm[col]="Matches"
        df = df.rename(columns=cm)
        ag = "U19" if "U19" in s else ("U23" if "U23" in s else "Senior")
        df["Age_Group"] = ag
        cols = [c for c in ["Player_Name","Country","Matches","Runs","Batting_Avg","Strike_Rate","Age_Group"] if c in df.columns]
        bat_frames.append(df[cols])
    except Exception as e:
        print(f"  [WARN] bat {s}: {e}")

bowl_frames = []
for s in bowl_sheets:
    try:
        df = read_sheet(s)
        if "Player" not in df.columns: continue
        cm = {}
        for col in df.columns:
            cl = col.lower()
            if cl=="player": cm[col]="Player_Name"
            elif cl=="country": cm[col]="Country"
            elif cl in ("wkts","wkt","wickets","w"): cm[col]="Wickets"
            elif cl in ("avg","average","bowling_avg"): cm[col]="Bowling_Avg"
            elif cl in ("econ","economy","economy_rate","er"): cm[col]="Economy_Rate"
        df = df.rename(columns=cm)
        cols = [c for c in ["Player_Name","Country","Wickets","Bowling_Avg","Economy_Rate"] if c in df.columns]
        bowl_frames.append(df[cols])
    except Exception as e:
        print(f"  [WARN] bowl {s}: {e}")

field_frames = []
for s in field_sheets:
    try:
        df = read_sheet(s)
        if "Player" not in df.columns and "Player_Name" not in df.columns:
            print(f"  [SKIP] {s} (team-level)")
            continue
        cm = {}
        for col in df.columns:
            cl = col.lower()
            if cl=="player": cm[col]="Player_Name"
            elif cl=="country": cm[col]="Country"
            elif cl in ("catches","catch","ct"): cm[col]="Catches"
            elif cl in ("stumpings","stumping","st"): cm[col]="Stumpings"
            elif cl in ("total_dismissals","dismissals","dis"): cm[col]="Total_Dismissals"
            elif cl in ("run_outs","runouts","ro"): cm[col]="Run_Outs"
        df = df.rename(columns=cm)
        cols = [c for c in ["Player_Name","Country","Catches","Stumpings","Total_Dismissals","Run_Outs"] if c in df.columns]
        if "Player_Name" in df.columns:
            field_frames.append(df[cols])
            print(f"  [OK] {s}: {len(df)} rows")
    except Exception as e:
        print(f"  [WARN] field {s}: {e}")

bat_all  = pd.concat(bat_frames,  ignore_index=True) if bat_frames  else pd.DataFrame()
bowl_all = pd.concat(bowl_frames, ignore_index=True) if bowl_frames else pd.DataFrame()
for col in ["Runs","Batting_Avg","Strike_Rate","Matches"]:
    if col in bat_all.columns: bat_all[col] = pd.to_numeric(bat_all[col], errors="coerce").fillna(0)
for col in ["Wickets","Bowling_Avg","Economy_Rate"]:
    if col in bowl_all.columns: bowl_all[col] = pd.to_numeric(bowl_all[col], errors="coerce").fillna(0)
for frame in [bat_all, bowl_all]:
    if not frame.empty and "Player_Name" in frame.columns:
        frame.drop(frame[frame["Player_Name"].astype(str).str.strip().isin(["Player","player","PLAYER"])].index, inplace=True)

bat_agg = bat_all.groupby("Player_Name", as_index=False).agg(
    Country=("Country", lambda x: x.mode().iloc[0] if len(x)>0 else "Unknown"),
    Total_Runs=("Runs","sum"),
    Batting_Avg=("Batting_Avg","mean"),
    Strike_Rate=("Strike_Rate","mean"),
    Total_Matches=("Matches","sum") if "Matches" in bat_all.columns else ("Player_Name","count"),
    Age_Group=("Age_Group", lambda x: "U19" if "U19" in x.values else ("U23" if "U23" in x.values else "Senior")),
) if not bat_all.empty else pd.DataFrame()

bowl_agg = bowl_all.groupby("Player_Name", as_index=False).agg(
    Total_Wickets=("Wickets","sum"),
    Bowling_Avg=("Bowling_Avg","mean"),
    Economy_Rate=("Economy_Rate","mean"),
) if not bowl_all.empty else pd.DataFrame()

field_agg = pd.DataFrame()
if field_frames:
    field_all = pd.concat(field_frames, ignore_index=True)
    for col in ["Catches","Stumpings","Total_Dismissals","Run_Outs"]:
        if col in field_all.columns:
            field_all[col] = pd.to_numeric(field_all[col], errors="coerce").fillna(0)
    if "Player_Name" in field_all.columns:
        field_all = field_all[~field_all["Player_Name"].astype(str).str.strip().isin(["Player","player","PLAYER"])]
    agg_d = {}
    if "Catches" in field_all.columns: agg_d["Total_Catches"]=("Catches","sum")
    if "Stumpings" in field_all.columns: agg_d["Total_Stumpings"]=("Stumpings","sum")
    if "Total_Dismissals" in field_all.columns: agg_d["Total_Dismissals"]=("Total_Dismissals","sum")
    if "Run_Outs" in field_all.columns: agg_d["Total_RunOuts"]=("Run_Outs","sum")
    if agg_d:
        field_agg = field_all.groupby("Player_Name", as_index=False).agg(**agg_d)

if not bat_agg.empty and not bowl_agg.empty:
    merged = bat_agg.merge(bowl_agg, on="Player_Name", how="outer")
elif not bat_agg.empty:
    merged = bat_agg
else:
    merged = bowl_agg
if not field_agg.empty and "Player_Name" in field_agg.columns:
    merged = merged.merge(field_agg, on="Player_Name", how="left")

for col, default in [("Total_Runs",0),("Batting_Avg",0),("Strike_Rate",100),("Total_Wickets",0),
                     ("Bowling_Avg",30),("Economy_Rate",8),("Total_Matches",5),("Total_Catches",0),
                     ("Total_Stumpings",0),("Total_Dismissals",0),("Total_RunOuts",0)]:
    merged[col] = merged.get(col, pd.Series([default]*len(merged))).fillna(default)
merged["Age_Group"] = merged.get("Age_Group", pd.Series(["Senior"]*len(merged))).fillna("Senior")
merged["Country"]   = merged.get("Country",   pd.Series(["Unknown"]*len(merged))).fillna("Unknown")

def assign_role(row):
    runs=row.get("Total_Runs",0); wkts=row.get("Total_Wickets",0); dismissals=row.get("Total_Dismissals",0)
    bat_score=runs; bowl_score=wkts*150
    if dismissals>=8 and bat_score<200 and bowl_score<200: return "Fielding"
    if bat_score>bowl_score*1.2: return "Batsman"
    elif bowl_score>bat_score*1.2: return "Bowler"
    else: return "Fielding"
merged["Role"] = merged.apply(assign_role, axis=1)

bat_avg_vals = pd.to_numeric(merged["Batting_Avg"], errors="coerce").fillna(0)
max_avg = bat_avg_vals.max() if bat_avg_vals.max()>0 else 1
merged["Form_Index_Last5"] = ((bat_avg_vals/max_avg)*8+2).round(1).clip(2,9.5)
merged = merged.drop_duplicates(subset="Player_Name").reset_index(drop=True)
print(f"[INFO] Unique players: {len(merged)}")
print(f"[INFO] Roles: {merged['Role'].value_counts().to_dict()}")

for col in ["Total_Runs","Batting_Avg","Strike_Rate","Total_Wickets","Bowling_Avg","Economy_Rate","Total_Catches","Total_Stumpings","Total_RunOuts"]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0 if col!="Bowling_Avg" else 30)
merged["Form_Index_last5"] = pd.to_numeric(merged["Form_Index_Last5"], errors="coerce").fillna(5.0)

merged["Batting_Impact"]  = merged["Total_Runs"]*0.4 + merged["Batting_Avg"]*0.3 + merged["Strike_Rate"]*0.3
merged["Bowling_Impact"]  = (merged["Total_Wickets"]*0.5 + (100/(merged["Bowling_Avg"]+1e-5))*0.25 + (merged["Economy_Rate"]+1e-5))*0.25
merged["Fielding_Impact"] = merged["Total_Catches"]*0.4 + merged["Total_Stumpings"]*0.3 + merged["Total_RunOuts"]*0.3

scaler = MinMaxScaler(feature_range=(0,100))
merged[["Batting_Impact","Bowling_Impact","Fielding_Impact"]] = scaler.fit_transform(merged[["Batting_Impact","Bowling_Impact","Fielding_Impact"]])
merged["Scout_Rating"] = merged["Batting_Impact"]*0.35 + merged["Bowling_Impact"]*0.35 + merged["Fielding_Impact"]*0.10 + merged["Form_Index_last5"]*2.0
merged["Scout_Rating"] = scaler.fit_transform(merged[["Scout_Rating"]])

features = ["Batting_Impact","Bowling_Impact","Fielding_Impact","Form_Index_last5","Scout_Rating"]
X = merged[features]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged["Cluster_Labels"] = kmeans.fit_predict(X)
mean_ratings = merged.groupby("Cluster_Labels")["Scout_Rating"].mean().sort_values(ascending=False)
tier_map = {cid: f"Tier_{r+1}" for r, cid in enumerate(mean_ratings.index)}
merged["Performance_Tier"] = merged["Cluster_Labels"].map(tier_map)
print(f"[INFO] Tiers: {merged['Performance_Tier'].value_counts().to_dict()}")

os.makedirs(os.path.join("data","processed"), exist_ok=True)
os.makedirs("models", exist_ok=True)
merged.to_csv(os.path.join("data","processed","processed_data.csv"), index=False)
joblib.dump(kmeans, os.path.join("models","talent_scout_model.pkl"))
print("[SUCCESS] Model saved: models/talent_scout_model.pkl")
print("[SUCCESS] CSV saved:   data/processed/processed_data.csv")

top = merged.nlargest(5,"Total_Dismissals")[["Player_Name","Country","Total_Catches","Total_Stumpings","Total_Dismissals","Role","Performance_Tier"]]
print()
print("--- Top 5 Fielders by Total Dismissals ---")
print(top.to_string(index=False))
