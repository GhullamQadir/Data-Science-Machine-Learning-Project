import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import re
from pathlib import Path

# Set UI Web Layout confugration limits
st.set_page_config(page_title="Talent Scouting Portal", page_icon="🏏", layout="wide")

# Layout configuration for header and Theme Toggle
col1, col2 = st.columns([0.85, 0.15])

with col1:
    st.markdown('<div class="main-header">Talent Scouting Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Data Science & Machine Learning Framework For Next-Gen Cricket Scouting </div>', unsafe_allow_html=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    dark_mode = st.toggle("🌙 Dark Theme", value=False)

if dark_mode:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        html, body { font-family: 'Outfit', sans-serif; }
        .stApp { background-color: #1A1A1A !important; }
        div[data-testid="stAppViewContainer"],
        div[data-testid="stAppViewBlockContainer"],
        div.block-container,
        div[data-testid="block-container"] { background-color: #1A1A1A !important; color: #FFFFFF !important; }
        header[data-testid="stHeader"],
        header[data-testid="stHeader"] > div { background-color: #111111 !important; border-bottom: 1px solid #2E2E2E !important; }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div { background-color: #141414 !important; border-right: 1px solid #2E2E2E !important; color: #FFFFFF !important; }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select { background-color: #2A2A2A !important; color: #FFFFFF !important; border: 1px solid #444 !important; }
        .main-header { font-size: 48px !important; font-weight: 800; color: #FFFFFF !important; margin-bottom: 5px; letter-spacing: -1.5px; line-height: 1.2; }
        .sub-header   { font-size: 18px !important; color: #D4AF37 !important; margin-bottom: 35px; font-weight: 500; letter-spacing: 0.5px; }
        .metric-box { background-color: #242424 !important; border: 1px solid #333 !important; border-left: 5px solid #D4AF37 !important; border-radius: 12px; text-align: center; padding: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.4); transition: all 0.3s ease; }
        .metric-box:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(212,175,55,0.25); }
        .metric-box h4 { color: #AAAAAA !important; font-size: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
        .metric-box h2 { color: #FFFFFF !important; font-weight: 800; font-size: 36px; margin-top: 0; letter-spacing: -1px; }
        h1, h2, h3, h4, h5, h6, p, label, li, td, th { color: #FFFFFF !important; }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] iframe { background-color: #1E1E1E !important; border-radius: 10px !important; }
        .stDataFrame [data-testid="glideDataEditor"],
        .stDataFrame canvas { background-color: #1E1E1E !important; }
        div[data-testid="stForm"] { background-color: #222222 !important; border: 1px solid #333333 !important; border-radius: 12px !important; padding: 16px !important; }
        input[type="number"], input[type="text"], textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input { background-color: #2A2A2A !important; color: #FFFFFF !important; border: 1px solid #444444 !important; border-radius: 8px !important; }
        div[data-testid="stNumberInput"], div[data-testid="stSlider"] { color: #FFFFFF !important; }
        div[data-testid="stSlider"] * { color: #FFFFFF !important; }
        .stSlider .st-ae { background-color: #D4AF37 !important; }
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] * { background-color: #222222 !important; color: #FFFFFF !important; border-color: #444 !important; }
        div[data-testid="stFileUploadDropzone"] { background-color: #1E1E1E !important; border: 2px dashed #D4AF37 !important; }
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"] * { background-color: #2A2A2A !important; color: #FFFFFF !important; }
        span[data-baseweb="tag"] { background-color: #D4AF37 !important; color: #1A1A1A !important; }
        div[data-testid="stAlert"], div[data-testid="stInfo"], div[data-testid="stWarning"], div[data-testid="stSuccess"], div[data-testid="stException"] { background-color: #222222 !important; border-left: 4px solid #D4AF37 !important; color: #FFFFFF !important; }
        div[data-testid="stAlert"] * { color: #FFFFFF !important; }
        .stButton > button { background-color: #D4AF37 !important; color: #1A1A1A !important; font-weight: 600; border-radius: 8px; border: 2px solid #D4AF37; padding: 10px 24px; transition: all 0.3s ease; }
        .stButton > button:hover { background-color: #1A1A1A !important; color: #D4AF37 !important; }
        label[data-testid="stToggleLabel"] * { color: #FFFFFF !important; }
        div[data-testid="stPlotlyChart"] { background-color: #1A1A1A !important; border-radius: 12px !important; }
        div[data-testid="stPlotlyChart"] > div { background-color: #1A1A1A !important; }
        hr { border-color: #333333 !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        html, body { font-family: 'Outfit', sans-serif; }
        .stApp, div[data-testid="stAppViewContainer"], div[data-testid="stAppViewBlockContainer"], div.block-container, div[data-testid="block-container"] { background-color: #FFFFFF !important; color: #000000 !important; }
        header[data-testid="stHeader"], header[data-testid="stHeader"] > div { background-color: #FFFFFF !important; border-bottom: 1px solid #EAEAEA !important; }
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background-color: #F8F8F8 !important; border-right: 1px solid #EAEAEA !important; }
        .main-header { font-size: 48px !important; font-weight: 800; color: #000000 !important; margin-bottom: 5px; letter-spacing: -1.5px; line-height: 1.2; }
        .sub-header   { font-size: 18px !important; color: #D4AF37 !important; margin-bottom: 35px; font-weight: 500; letter-spacing: 0.5px; }
        .metric-box { background-color: #FFFFFF !important; border: 1px solid #EAEAEA !important; border-left: 5px solid #D4AF37 !important; border-radius: 12px; text-align: center; padding: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.04); transition: all 0.3s ease; }
        .metric-box:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(212,175,55,0.15); }
        .metric-box h4 { color: #666666 !important; font-size: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
        .metric-box h2 { color: #000000 !important; font-weight: 800; font-size: 36px; margin-top: 0; letter-spacing: -1px; }
        h1, h2, h3, h4, h5, h6 { color: #000000 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
        div[data-testid="stFileUploadDropzone"] { border: 2px dashed #D4AF37 !important; }
        span[data-baseweb="tag"] { background-color: #000000 !important; color: #D4AF37 !important; }
        .stButton > button { background-color: #000000 !important; color: #D4AF37 !important; font-weight: 600; border-radius: 8px; border: 2px solid #000000; padding: 10px 24px; transition: all 0.3s ease; }
        .stButton > button:hover { background-color: #D4AF37 !important; color: #000000 !important; border-color: #D4AF37; }
        hr { border-color: #EAEAEA !important; }
        </style>
    """, unsafe_allow_html=True)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_pipeline import feature_engineering, train_and_persist_model

BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = BASE_DIR.parent.parent
processed_data_path = BASE_DIR / "data" / "processed" / "processed_data.csv"
model_path = BASE_DIR / "models" / "talent_scout_model.pkl"
default_raw_path = BASE_DIR / "data" / "Raw_data" / "Raw Dataset For DS & ML Project.xlsx"
age_data_path = REPOSITORY_DIR / "Data Set" / "Age Of Players.txt"


def _player_name_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _load_age_map():
    age_map = {}
    if age_data_path.exists():
        for line in age_data_path.read_text(encoding="utf-8").splitlines():
            match = re.search(r"^\s*\*?\s*(.+?)\s*=\s*(\d+)\s*$", line)
            if match:
                age_map[_player_name_key(match.group(1))] = int(match.group(2))
    return age_map


def _assign_role(row):
    runs = pd.to_numeric(row.get("Total_Runs", 0), errors="coerce")
    wickets = pd.to_numeric(row.get("Total_Wickets", 0), errors="coerce")
    runs = 0 if pd.isna(runs) else runs
    wickets = 0 if pd.isna(wickets) else wickets
    if runs > wickets * 150 * 1.2:
        return "Batsman"
    if wickets * 150 > runs * 1.2:
        return "Bowler"
    return "All-Rounder"


def _enrich_player_metadata(df):
    df = df.copy()
    age_map = _load_age_map()
    if "Player_Name" in df.columns:
        keys = df["Player_Name"].map(_player_name_key)
        mapped_ages = keys.map(age_map)
        existing_ages = (
            pd.to_numeric(df["Age"], errors="coerce")
            if "Age" in df.columns
            else pd.Series(index=df.index, dtype="float64")
        )
        df["Age"] = mapped_ages.fillna(existing_ages).fillna(25)
    if "Country" not in df.columns:
        df["Country"] = "Unknown"
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    if "Role" not in df.columns:
        df["Role"] = df.apply(_assign_role, axis=1)
    else:
        missing_roles = df["Role"].isna() | df["Role"].astype(str).str.strip().isin(["", "Unknown", "nan"])
        df.loc[missing_roles, "Role"] = df.loc[missing_roles].apply(_assign_role, axis=1)
    return df

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Your Data")
st.sidebar.markdown("Upload your own `.csv` or `.xlsx` player dataset to analyze real stats.")

if 'upload_widget_key' not in st.session_state:
    st.session_state.upload_widget_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload Player Dataset",
    type=["csv", "xlsx", "xls"],
    key=f"player_upload_{st.session_state.upload_widget_key}",
    help="Your file must have columns like: Player_Name, Country, Age, Runs, Batting_Avg, Strike_Rate, Wickets, Bowling_Avg, Economy_Rate"
)

def _standardize_flat_file(df):
    df = df.copy()
    df.columns = [str(column).strip().replace(" ", "_") for column in df.columns]
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ('player', 'name', 'player_name', 'cricketer'): col_map[col] = 'Player_Name'
        elif cl in ('country', 'nation', 'team'): col_map[col] = 'Country'
        elif cl == 'age': col_map[col] = 'Age'
        elif cl in ('tournament', 'competition', 'event'): col_map[col] = 'Tournament'
    df = df.rename(columns=col_map)
    if 'Player_Name' not in df.columns:
        raise ValueError("The uploaded workbook does not contain a Player column.")
    df['Player_Name'] = df['Player_Name'].astype(str).str.strip()
    df = df[~df['Player_Name'].str.lower().isin(['', 'nan', 'player'])].copy()
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df


def _dedup_cols(cols):
    seen = {}
    res = []
    for c in cols:
        n = str(c).strip().replace(' ', '_')
        if n in seen:
            seen[n] += 1
            res.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            res.append(n)
    return res

def _parse_stacked_sheets(xl):
    bat_frames, bowl_frames, field_frames = [], [], []
    for sheet in xl.sheet_names:
        try: df = pd.read_excel(xl, sheet_name=sheet, header=None)
        except Exception: continue
        mode = None
        header_idx = None
        data_rows = []
        headers = []
        for idx, row in df.iterrows():
            row_str = " ".join([str(x).strip() for x in row if pd.notna(x)]).lower()
            if 'batting' in row_str: mode = 'bat'; header_idx = None; continue
            elif 'bowling' in row_str: mode = 'bowl'; header_idx = None; continue
            elif 'fielding' in row_str: mode = 'field'; header_idx = None; continue
            if mode:
                if header_idx is None:
                    if 'player' in row_str:
                        header_idx = idx
                        headers = row.tolist()
                        data_rows = []
                else:
                    player_cell = row.iloc[0] if len(row) > 0 else None
                    if pd.isna(player_cell) or str(player_cell).strip() == '':
                        if data_rows:
                            chunk = pd.DataFrame(data_rows, columns=headers)
                            chunk.columns = _dedup_cols(chunk.columns)
                            chunk['_Tournament'] = sheet
                            if mode == 'bat': bat_frames.append(chunk)
                            elif mode == 'bowl': bowl_frames.append(chunk)
                            elif mode == 'field': field_frames.append(chunk)
                            data_rows = []
                            mode = None
                    else:
                        data_rows.append(row.values)
        if mode and data_rows:
            chunk = pd.DataFrame(data_rows, columns=headers)
            chunk.columns = _dedup_cols(chunk.columns)
            chunk['_Tournament'] = sheet
            if mode == 'bat': bat_frames.append(chunk)
            elif mode == 'bowl': bowl_frames.append(chunk)
            elif mode == 'field': field_frames.append(chunk)
    return bat_frames, bowl_frames, field_frames

def _process_and_merge_frames(bat_frames, bowl_frames, field_frames):
    def prepare_frames(frames):
        if not frames: return pd.DataFrame()
        # Drop columns that are completely empty or just nan
        for i in range(len(frames)):
            frames[i] = frames[i].loc[:, ~frames[i].columns.str.startswith('nan_')]
            if 'nan' in frames[i].columns:
                frames[i] = frames[i].drop(columns=['nan'])
        if not frames: return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
        if 'Player' in df.columns: df = df.rename(columns={'Player': 'Player_Name'})
        elif 'Name' in df.columns: df = df.rename(columns={'Name': 'Player_Name'})
        def extract_year(x):
            if pd.isna(x): return 'Unknown'
            m = re.search(r'20\d\d', str(x))
            return m.group(0) if m else 'Unknown'
        df['Year'] = df['_Tournament'].apply(extract_year)
        
        # We need to map standard columns so the pipeline understands them
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if cl in ('runs','run','r'): col_map[col] = 'Runs'
            elif cl in ('avg','average','batting_avg','bat_avg'): col_map[col] = 'Batting_Avg'
            elif cl in ('sr','strike_rate','strike rate','bsr'): col_map[col] = 'Strike_Rate'
            elif cl in ('mat','matches','m'): col_map[col] = 'Matches'
            elif cl in ('wkts','wkt','wickets','w'): col_map[col] = 'Wickets'
            elif cl in ('avg','average','bowling_avg','bowl_avg'): col_map[col] = 'Bowling_Avg'
            elif cl in ('econ','economy','economy_rate','er'): col_map[col] = 'Economy_Rate'
        df = df.rename(columns=col_map)
        return df

    bat_df = prepare_frames(bat_frames)
    bowl_df = prepare_frames(bowl_frames)
    field_df = prepare_frames(field_frames)
    
    # We must ensure we aggregate properly, just concatenating won't group by player.
    # We will compute the aggregate for each frame separately, then outer join.
    if not bat_df.empty:
        for c in ['Runs', 'Batting_Avg', 'Strike_Rate', 'Matches']:
            if c in bat_df.columns: bat_df[c] = pd.to_numeric(bat_df[c], errors='coerce').fillna(0)
        bat_agg = bat_df.groupby('Player_Name', as_index=False).agg({
            'Runs': 'sum', 'Batting_Avg': 'mean', 'Strike_Rate': 'mean',
            'Matches': 'sum' if 'Matches' in bat_df.columns else lambda x: len(x),
            'Year': lambda x: str(x.iloc[0]) if len(x)>0 else 'Unknown'
        })
    else: bat_agg = pd.DataFrame()

    if not bowl_df.empty:
        for c in ['Wickets', 'Bowling_Avg', 'Economy_Rate', 'Matches']:
            if c in bowl_df.columns: bowl_df[c] = pd.to_numeric(bowl_df[c], errors='coerce').fillna(0)
        bowl_agg = bowl_df.groupby('Player_Name', as_index=False).agg({
            'Wickets': 'sum', 'Bowling_Avg': 'mean', 'Economy_Rate': 'mean',
            'Year': lambda x: str(x.iloc[0]) if len(x)>0 else 'Unknown'
        })
    else: bowl_agg = pd.DataFrame()

    if not bat_agg.empty and not bowl_agg.empty:
        merged = bat_agg.merge(bowl_agg, on='Player_Name', how='outer', suffixes=('', '_bowl'))
        merged['Year'] = merged['Year'].fillna(merged['Year_bowl'])
    elif not bat_agg.empty: merged = bat_agg
    elif not bowl_agg.empty: merged = bowl_agg
    else: return pd.DataFrame()

    # Load custom country and age data
    merged = _enrich_player_metadata(merged)

    # Remap columns for the pipeline
    col_map = {
        'Runs': 'Total_Runs',
        'Wickets': 'Total_Wickets',
        'Matches': 'Total_Matches'
    }
    merged = merged.rename(columns=col_map)
    
    return merged

def _find_header_row(raw_df):
    for i, row in raw_df.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if 'player' in vals: return i
    return 0

def _read_sheet_with_auto_header(xl, sheet_name):
    raw = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw)
    df = pd.read_excel(xl, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    if 'Player' in df.columns:
        df = df[df['Player'].notna()]
        df = df[~df['Player'].astype(str).str.strip().str.lower().isin(['player', 'nan', ''])]
    return df.reset_index(drop=True)

def _infer_tournament_name(sheet_name):
    parts = sheet_name.upper().replace('_BAT','').replace('_BOWL','').replace('_FIELD','')
    return parts.replace('_', ' ').strip()

def _merge_cricket_sheets(file, xl, bat_sheets, bowl_sheets):
    bat_frames = []
    bowl_frames = []
    for sheet in bat_sheets:
        try:
            df = _read_sheet_with_auto_header(xl, sheet)
            df['_Tournament'] = _infer_tournament_name(sheet)
            bat_frames.append(df)
        except Exception: continue
    for sheet in bowl_sheets:
        try:
            df = _read_sheet_with_auto_header(xl, sheet)
            df['_Tournament'] = _infer_tournament_name(sheet)
            bowl_frames.append(df)
        except Exception: continue
    return _process_and_merge_frames(bat_frames, bowl_frames, [])

def smart_load_file(file):
    filename = file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
        return _standardize_flat_file(df)

    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names

    # The supplied workbooks use descriptive sheet names and place headers
    # several rows below the title. Read every player table instead of relying
    # on BAT/BOWL naming conventions or the first (dashboard) sheet.
    frames = []
    for sheet in sheet_names:
        if 'dashboard' in sheet.lower():
            continue
        try:
            frame = _read_sheet_with_auto_header(xl, sheet)
        except (ValueError, ImportError):
            continue
        try:
            frame = _standardize_flat_file(frame)
        except ValueError:
            continue
        frame['Tournament'] = frame.get('Tournament', sheet)
        frames.append(frame)

    if not frames:
        return _standardize_flat_file(pd.read_excel(file))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined['Player_Name'] = combined['Player_Name'].astype(str).str.strip()
    numeric_sum_columns = {
        'Runs', 'Total_Runs', 'Wickets', 'Wkts', 'Total_Wickets',
        'Matches', 'Mat', 'Catches', 'Stumpings', 'Run_Outs',
        'RunOuts', 'Dismissals'
    }
    numeric_columns = combined.select_dtypes(include='number').columns.tolist()
    aggregations = {
        column: ('sum' if column in numeric_sum_columns else 'mean')
        for column in numeric_columns
    }
    for column in combined.columns:
        if column not in aggregations and column != 'Player_Name':
            aggregations[column] = 'first'
    return combined.groupby('Player_Name', as_index=False).agg(aggregations)

def run_pipeline_on_df(raw_df):
    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    raw_df = _standardize_flat_file(raw_df)
    engineered = feature_engineering(raw_df.copy())
    engineered = _enrich_player_metadata(engineered)
    final = train_and_persist_model(engineered)
    return final

if 'clear_data' not in st.session_state:
    st.session_state.clear_data = False

df = None
if uploaded_file is not None:
    try:
        raw_df = smart_load_file(uploaded_file)
        df = run_pipeline_on_df(raw_df)
        st.session_state.clear_data = False
        st.sidebar.success(f"✅ Loaded **{len(df)} players** from your file!")
    except Exception as e:
        st.sidebar.error(f"❌ Error processing file: {e}")
        df = None
else:
    if not st.session_state.clear_data:
        try:
            if processed_data_path.exists():
                cached = pd.read_csv(processed_data_path)
                required = {'Player_Name', 'Age', 'Country', 'Role', 'Total_Runs', 'Batting_Impact', 'Scout_Rating'}
                invalid_names = cached["Player_Name"].astype(str).str.contains(
                    "total players|highest runs|statistics", case=False, regex=True, na=False
                ) if "Player_Name" in cached.columns else pd.Series(dtype=bool)
                missing_roles = (
                    cached["Role"].isna() | cached["Role"].astype(str).str.strip().isin(["", "Unknown", "nan"])
                ) if "Role" in cached.columns else pd.Series([True])
                if not required.issubset(cached.columns) or cached.empty or invalid_names.any() or missing_roles.any():
                    raise ValueError("Cached data is missing player statistics.")
                df = _enrich_player_metadata(cached)
            elif default_raw_path.exists():
                df = run_pipeline_on_df(smart_load_file(default_raw_path))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.sidebar.warning(f"Cached dashboard data was rebuilt: {exc}")
            if default_raw_path.exists():
                df = run_pipeline_on_df(smart_load_file(default_raw_path))

if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.clear_data = True
    for generated_path in (processed_data_path, model_path):
        if generated_path.exists():
            generated_path.unlink()
    st.session_state.upload_widget_key += 1
    st.rerun()

if df is not None:
    if 'Player_Name' in df.columns:
        df['Player_Name'] = df['Player_Name'].astype(str).str.strip()
    if 'Country' in df.columns:
        df['Country'] = df['Country'].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
    df = _enrich_player_metadata(df)
    for categorical_column in ('Role', 'Performance_Tier'):
        if categorical_column in df.columns:
            df[categorical_column] = df[categorical_column].fillna('Unknown').astype(str).str.strip()
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(25)

if df is None or df.empty:
    st.info("**👋 Welcome to Talent Scouting Portal!**")
    st.warning("No data loaded yet. Please upload your player dataset using the sidebar panel on the left.")
    demo_cols = ['Player_Name', 'Country', 'Age', 'Role', 'Scout_Rating', 'Performance_Tier', 'Total_Matches',
                 'Total_Runs', 'Batting_Avg', 'Strike_Rate', 'Total_Wickets',
                 'Bowling_Avg', 'Economy_Rate', 'Form_Index_Last5', 'Batting_Impact', 'Bowling_Impact']
    df = pd.DataFrame(columns=demo_cols)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Players")

_countries = sorted(df['Country'].unique()) if 'Country' in df.columns else []
countries = st.sidebar.multiselect("Select Country Pool", options=_countries, default=_countries)

_roles = list(df['Role'].unique()) if 'Role' in df.columns else []
roles_list = sorted(list(set(['Batsman', 'Bowler', 'All-Rounder'] + _roles)))
roles = st.sidebar.multiselect("Select Player Role", options=roles_list, default=roles_list)

_tiers = sorted(df['Performance_Tier'].unique()) if 'Performance_Tier' in df.columns else []
tiers = st.sidebar.multiselect("Select Model Ranked Tiers", options=_tiers, default=_tiers)

_years = ['All', '2024', '2025', '2026']
selected_year = st.sidebar.selectbox("Select player by years", options=_years, index=0)

u19_u23_only = st.sidebar.checkbox("Isolate Under-19 & Under-23 Talents Only", value=False)

if not df.empty:
    filtered_df = df.copy()
    if 'Country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    if 'Role' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Role'].isin(roles)]
    if 'Performance_Tier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Performance_Tier'].isin(tiers)]

    if u19_u23_only and 'Age' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Age'] <= 23]
    if selected_year != 'All' and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Year'].astype(str).str.strip() == selected_year]
else:
    filtered_df = df.copy()

m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f'<div class="metric-box"><h4>Total Scanned Assets</h4><h2>{len(filtered_df)}</h2></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-box"><h4>Tier 1 (Elite) Prospects</h4><h2>{len(filtered_df[filtered_df["Performance_Tier"] == "Tier_1"]) if not filtered_df.empty else 0}</h2></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-box"><h4>Mean Scout Rating</h4><h2>{round(filtered_df["Scout_Rating"].mean(), 2) if not filtered_df.empty else 0}</h2></div>', unsafe_allow_html=True)
with m4:
    form_col = 'Form_Index_Last5' if 'Form_Index_Last5' in filtered_df.columns else 'Form_Index_last5'
    st.markdown(f'<div class="metric-box"><h4>Global Squad Form Index</h4><h2>{filtered_df[form_col].mean() if not filtered_df.empty else 0:.1f}/10</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🏏 Machine Learning Evaluated Talent Pool")
display_cols = ['Player_Name', 'Country', 'Age', 'Year', 'Role', 'Scout_Rating', 'Performance_Tier', form_col, 'Total_Runs', 'Batting_Avg', 'Strike_Rate', 'Total_Wickets', 'Bowling_Avg', 'Economy_Rate']
available_cols = [c for c in display_cols if c in filtered_df.columns]
display_df = filtered_df[available_cols].sort_values(by="Scout_Rating", ascending=False).reset_index(drop=True) if not filtered_df.empty else filtered_df[available_cols]

if dark_mode:
    styled_df = display_df.style.set_properties(**{
        'background-color': '#000000',
        'color': '#FFD700',
        'border-color': '#333333'
    })
    st.dataframe(styled_df, use_container_width=True)
else:
    st.dataframe(display_df, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("  Performance Segmentation Space Mapping")
chart_template = "plotly_dark" if dark_mode else "plotly_white"
tier2_color = "#AAAAAA" if dark_mode else "#333333"

if not filtered_df.empty and 'Batting_Impact' in filtered_df.columns:
    fig = px.scatter(filtered_df, x="Batting_Impact", y="Bowling_Impact", color="Performance_Tier", size="Scout_Rating", hover_name='Player_Name', hover_data=['Age', 'Country', form_col], color_discrete_map={"Tier_1": "#D4AF37", "Tier_2": tier2_color, "Tier_3": "#808080"}, title="2D Performance Impact Scatter — Player Distribution", template=chart_template)
else:
    fig = px.scatter(title="2D Performance Impact Scatter (No Data)", template=chart_template)

fig.update_layout(
    paper_bgcolor="#1A1A1A" if dark_mode else "#FFFFFF",
    plot_bgcolor="#1E1E1E" if dark_mode else "#F9F9F9",
    font=dict(color="#FFFFFF" if dark_mode else "#000000", family="Outfit"),
    title_font=dict(size=18, color="#D4AF37"),
    legend=dict(bgcolor="#242424" if dark_mode else "#FFFFFF", bordercolor="#D4AF37", borderwidth=1, font=dict(color="#FFFFFF" if dark_mode else "#000000")),
    xaxis=dict(gridcolor="#333333" if dark_mode else "#EAEAEA", color="#FFFFFF" if dark_mode else "#000000"),
    yaxis=dict(gridcolor="#333333" if dark_mode else "#EAEAEA", color="#FFFFFF" if dark_mode else "#000000"),
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.subheader(" Real-Time Machine Learning Scouting Predictor")
st.markdown("Enter New Player Stats To Check Their Prediction")

with st.form("realtime_scouting_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        in_runs = st.number_input("Total Runs Scored", min_value=0,value=150)
        in_avg_bat = st.number_input("Batting Average Performance", min_value=0.0,value=32.5)
        in_sr = st.number_input("Batting Strike Rate Axis", min_value=0,value=130)
    with c2:
        in_wkts = st.number_input("Total Wickets Taken", min_value=0,value=12)
        in_avg_bowl = st.number_input("Bowling Average Performance", min_value=0.0,value=24.5)
        in_econ = st.number_input("Bowling Economy Rate Parameters", min_value=0.0,value=8.5)
    with c3:
        in_form = st.slider("Recent Form Track Index Rating", min_value=1.0, max_value=10.0 , value=7.0, step=0.1)

    eval_btn = st.form_submit_button("Run Analytics Inference Engine")

    if eval_btn:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            calc_bat_impact = (in_runs* 0.4) + (in_avg_bat * 0.3) + (in_sr * 0.3)
            calc_bowl_impact = (in_wkts* 0.5) + (100/ (in_avg_bowl + 1e-5) * 0.25) + (100/ (in_econ + 1e-5) * 0.25)
            calc_rating = (calc_bat_impact* 0.4) + (calc_bowl_impact * 0.4) + (in_form * 2.0)
            input_vector = np.array([[calc_bat_impact, calc_bowl_impact, 0.0, in_form, calc_rating]])
            predicted_cluster = model.predict(input_vector) [0]
            st.success(f"**Prediction Engine Inference Completed Successfully!**")
            st.markdown(f"**Calculated Core Scout Rating: ** '{calc_rating:.2f}'")
            st.info(f"The algorithm has classified this player profile as an asset belonging to the evaluated group cluster segment ID: ** Cluster {predicted_cluster} **")
        else:
            simulated_rating = min (((in_runs * 0.1) + (in_sr * 0.2) + (in_wkts * 2.5) + (in_form * 4)), 100.0)
            st.warning("Using Analytical Rule Based Inference (Trained Model File (.pkl) not found on disk). ")
            st.success(f" Estimated Rating Matrix Score: '{simulated_rating:.2f}'")
