import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os


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

        /* ===== GLOBAL FONT & COLOR ===== */
        html, body {
            font-family: 'Outfit', sans-serif !important;
        }

        /* ===== MAIN APP BACKGROUND ===== */
        .stApp {
            background-color: #1A1A1A !important;
        }
        div[data-testid="stAppViewContainer"],
        div[data-testid="stAppViewBlockContainer"],
        div.block-container,
        div[data-testid="block-container"] {
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
        }

        /* ===== TOP HEADER / NAVBAR ===== */
        header[data-testid="stHeader"],
        header[data-testid="stHeader"] > div {
            background-color: #111111 !important;
            border-bottom: 1px solid #2E2E2E !important;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: #141414 !important;
            border-right: 1px solid #2E2E2E !important;
        }
        section[data-testid="stSidebar"] {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select {
            background-color: #2A2A2A !important;
            color: #FFFFFF !important;
            border: 1px solid #444 !important;
        }

        /* ===== CUSTOM HEADER CLASSES ===== */
        .main-header { font-size: 48px !important; font-weight: 800; color: #FFFFFF !important; margin-bottom: 5px; letter-spacing: -1.5px; line-height: 1.2; }
        .sub-header   { font-size: 18px !important; color: #D4AF37 !important; margin-bottom: 35px; font-weight: 500; letter-spacing: 0.5px; }

        /* ===== METRIC BOXES ===== */
        .metric-box { background-color: #242424 !important; border: 1px solid #333 !important; border-left: 5px solid #D4AF37 !important; border-radius: 12px; text-align: center; padding: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.4); transition: all 0.3s ease; }
        .metric-box:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(212,175,55,0.25); }
        .metric-box h4 { color: #AAAAAA !important; font-size: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
        .metric-box h2 { color: #FFFFFF !important; font-weight: 800; font-size: 36px; margin-top: 0; letter-spacing: -1px; }

        /* ===== ALL TEXT ELEMENTS ===== */
        h1, h2, h3, h4, h5, h6,
        p, label, li, td, th {
            color: #FFFFFF !important;
        }

        /* ===== DATAFRAME / TABLE ===== */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] iframe {
            background-color: #1E1E1E !important;
            border-radius: 10px !important;
        }
        /* Arrow table (newer Streamlit) */
        .stDataFrame [data-testid="glideDataEditor"],
        .stDataFrame canvas { background-color: #1E1E1E !important; }

        /* ===== FORM & INPUT FIELDS ===== */
        div[data-testid="stForm"] {
            background-color: #222222 !important;
            border: 1px solid #333333 !important;
            border-radius: 12px !important;
            padding: 16px !important;
        }
        input[type="number"],
        input[type="text"],
        textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            background-color: #2A2A2A !important;
            color: #FFFFFF !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stNumberInput"],
        div[data-testid="stSlider"] { color: #FFFFFF !important; }
        div[data-testid="stSlider"] * { color: #FFFFFF !important; }
        .stSlider .st-ae { background-color: #D4AF37 !important; }

        /* ===== FILE UPLOADER ===== */
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] * {
            background-color: #222222 !important;
            color: #FFFFFF !important;
            border-color: #444 !important;
        }
        div[data-testid="stFileUploadDropzone"] {
            background-color: #1E1E1E !important;
            border: 2px dashed #D4AF37 !important;
        }

        /* ===== MULTISELECT / SELECTBOX ===== */
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"] * {
            background-color: #2A2A2A !important;
            color: #FFFFFF !important;
        }
        span[data-baseweb="tag"] {
            background-color: #D4AF37 !important;
            color: #1A1A1A !important;
        }

        /* ===== ALERTS (info, warning, success, error) ===== */
        div[data-testid="stAlert"],
        div[data-testid="stInfo"],
        div[data-testid="stWarning"],
        div[data-testid="stSuccess"],
        div[data-testid="stException"] {
            background-color: #222222 !important;
            border-left: 4px solid #D4AF37 !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stAlert"] * { color: #FFFFFF !important; }

        /* ===== BUTTON ===== */
        .stButton > button {
            background-color: #D4AF37 !important; color: #1A1A1A !important;
            font-weight: 600; border-radius: 8px; border: 2px solid #D4AF37; padding: 10px 24px; transition: all 0.3s ease;
        }
        .stButton > button:hover { background-color: #1A1A1A !important; color: #D4AF37 !important; }

        /* ===== TOGGLE / CHECKBOX ===== */
        label[data-testid="stToggleLabel"] * { color: #FFFFFF !important; }

        /* ===== PLOTLY CHART WRAPPER ===== */
        div[data-testid="stPlotlyChart"] {
            background-color: #1A1A1A !important;
            border-radius: 12px !important;
        }
        div[data-testid="stPlotlyChart"] > div { background-color: #1A1A1A !important; }

        /* ===== DIVIDERS ===== */
        hr { border-color: #333333 !important; }

        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* ===== GLOBAL FONT ===== */
        html, body {
            font-family: 'Outfit', sans-serif !important;
        }

        /* ===== MAIN APP BACKGROUND ===== */
        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stAppViewBlockContainer"],
        div.block-container,
        div[data-testid="block-container"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        /* ===== TOP HEADER / NAVBAR ===== */
        header[data-testid="stHeader"],
        header[data-testid="stHeader"] > div {
            background-color: #FFFFFF !important;
            border-bottom: 1px solid #EAEAEA !important;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: #F8F8F8 !important;
            border-right: 1px solid #EAEAEA !important;
        }

        /* ===== CUSTOM HEADER CLASSES ===== */
        .main-header { font-size: 48px !important; font-weight: 800; color: #000000 !important; margin-bottom: 5px; letter-spacing: -1.5px; line-height: 1.2; }
        .sub-header   { font-size: 18px !important; color: #D4AF37 !important; margin-bottom: 35px; font-weight: 500; letter-spacing: 0.5px; }

        /* ===== METRIC BOXES ===== */
        .metric-box { background-color: #FFFFFF !important; border: 1px solid #EAEAEA !important; border-left: 5px solid #D4AF37 !important; border-radius: 12px; text-align: center; padding: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.04); transition: all 0.3s ease; }
        .metric-box:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(212,175,55,0.15); }
        .metric-box h4 { color: #666666 !important; font-size: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
        .metric-box h2 { color: #000000 !important; font-weight: 800; font-size: 36px; margin-top: 0; letter-spacing: -1px; }

        /* ===== HEADINGS & TEXT ===== */
        h1, h2, h3, h4, h5, h6 { color: #000000 !important; font-weight: 700 !important; letter-spacing: -0.5px; }

        /* ===== FILE UPLOADER ===== */
        div[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #D4AF37 !important;
        }

        /* ===== MULTISELECT TAGS ===== */
        span[data-baseweb="tag"] {
            background-color: #000000 !important;
            color: #D4AF37 !important;
        }

        /* ===== BUTTON ===== */
        .stButton > button {
            background-color: #000000 !important; color: #D4AF37 !important;
            font-weight: 600; border-radius: 8px; border: 2px solid #000000; padding: 10px 24px; transition: all 0.3s ease;
        }
        .stButton > button:hover { background-color: #D4AF37 !important; color: #000000 !important; border-color: #D4AF37; }

        /* ===== DIVIDERS ===== */
        hr { border-color: #EAEAEA !important; }

        </style>
    """, unsafe_allow_html=True)


# ---- Import pipeline functions ----
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_pipeline import feature_engineering, train_and_persist_model

# Data Verfication and Pipeline Loading
processed_data_path = os.path.join("data", "processed", "processed_data.csv")
model_path = os.path.join("models", "talent_scout_model.pkl")


# ---- Sidebar: File Upload Section ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Your Data")
st.sidebar.markdown("Upload your own `.csv` or `.xlsx` player dataset to analyze real stats.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Player Dataset",
    type=["csv", "xlsx", "xls"],
    help="Your file must have columns like: Player_Name, Country, Age_Group, Role, Total_Runs, Batting_Avg, Strike_Rate, Total_Wickets, Bowling_Avg, Economy_Rate, Form_Index_Last5"
)


def smart_load_file(file):
    """
    Smart loader: handles both simple CSV/Excel AND complex multi-sheet datasets.
    Auto-detects the data format and returns a clean, merged DataFrame ready for the pipeline.
    """
    filename = file.name.lower()

    # --- SIMPLE CSV ---
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        return df

    # --- EXCEL: detect if multi-sheet cricket dataset ---
    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names

    # Check if it's the complex multi-sheet cricket format
    bat_sheets  = [s for s in sheet_names if '_BAT' in s.upper() or 'BAT' in s.upper()]
    bowl_sheets = [s for s in sheet_names if '_BOWL' in s.upper() or 'BOWL' in s.upper()]

    if bat_sheets or bowl_sheets:
        return _merge_cricket_sheets(file, xl, bat_sheets, bowl_sheets)
    else:
        # Simple single-sheet Excel
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        return df


def _find_header_row(raw_df):
    """
    Scan rows to find the actual header row.
    A valid header MUST contain the word 'Player' as an exact cell value (case-insensitive).
    """
    for i, row in raw_df.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        # The header row must contain 'player' as one of its cell values
        if 'player' in vals:
            return i
    return 0


def _read_sheet_with_auto_header(xl, sheet_name):
    """Read a sheet, find the real header row, return a clean DataFrame."""
    raw = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw)
    df = pd.read_excel(xl, sheet_name=sheet_name, header=header_row)
    # Clean column names
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    # Drop rows that are just repeats of the header text
    if 'Player' in df.columns:
        df = df[df['Player'].notna()]
        df = df[~df['Player'].astype(str).str.strip().str.lower().isin(['player', 'nan', ''])]
    return df.reset_index(drop=True)


def _infer_tournament_name(sheet_name):
    """Extract a short tournament label from sheet name like 'T20_WC_2024_BAT' -> 'T20 WC 2024'."""
    parts = sheet_name.upper().replace('_BAT','').replace('_BOWL','').replace('_FIELD','')
    return parts.replace('_', ' ').strip()


def _infer_age_group(tournament):
    """Guess age group from tournament name."""
    t = tournament.upper()
    if 'U19' in t:
        return 'U19'
    elif 'U23' in t:
        return 'U23'
    else:
        return 'Senior'


def _merge_cricket_sheets(file, xl, bat_sheets, bowl_sheets):
    """Merge all batting and bowling sheets into one unified player DataFrame."""
    bat_frames  = []
    bowl_frames = []

    # ---- Load batting sheets ----
    for sheet in bat_sheets:
        try:
            df = _read_sheet_with_auto_header(xl, sheet)
            if 'Player' not in df.columns:
                continue
            # Standardise column names
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl == 'player':       col_map[col] = 'Player_Name'
                elif cl == 'country':    col_map[col] = 'Country'
                elif cl in ('runs','run','r'): col_map[col] = 'Runs'
                elif cl in ('avg','average','batting_avg','bat_avg'): col_map[col] = 'Batting_Avg'
                elif cl in ('sr','strike_rate','strike rate','bsr'): col_map[col] = 'Strike_Rate'
                elif cl in ('mat','matches','m'):   col_map[col] = 'Matches'
            df = df.rename(columns=col_map)
            df['_Tournament'] = _infer_tournament_name(sheet)
            df['Age_Group']   = _infer_age_group(sheet)
            bat_frames.append(df[['Player_Name','Country','Matches','Runs','Batting_Avg','Strike_Rate','_Tournament','Age_Group']
                                  if all(c in df.columns for c in ['Player_Name','Country','Runs'])
                                  else [c for c in ['Player_Name','Country','Matches','Runs','Batting_Avg','Strike_Rate','_Tournament','Age_Group'] if c in df.columns]])
        except Exception:
            continue

    # ---- Load bowling sheets ----
    for sheet in bowl_sheets:
        try:
            df = _read_sheet_with_auto_header(xl, sheet)
            if 'Player' not in df.columns:
                continue
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl == 'player':          col_map[col] = 'Player_Name'
                elif cl == 'country':       col_map[col] = 'Country'
                elif cl in ('wkts','wkt','wickets','w'): col_map[col] = 'Wickets'
                elif cl in ('avg','average','bowling_avg','bowl_avg'): col_map[col] = 'Bowling_Avg'
                elif cl in ('econ','economy','economy_rate','er'): col_map[col] = 'Economy_Rate'
                elif cl in ('mat','matches','m'): col_map[col] = 'Matches'
            df = df.rename(columns=col_map)
            df['_Tournament'] = _infer_tournament_name(sheet)
            bowl_frames.append(df[['Player_Name','Country','Wickets','Bowling_Avg','Economy_Rate','_Tournament']
                                   if all(c in df.columns for c in ['Player_Name','Country'])
                                   else [c for c in ['Player_Name','Country','Wickets','Bowling_Avg','Economy_Rate','_Tournament'] if c in df.columns]])
        except Exception:
            continue

    if not bat_frames and not bowl_frames:
        raise ValueError("Could not extract player data from any sheet. Please check your file format.")

    # ---- Aggregate: one row per player ----
    # Batting: sum runs, weighted avg of batting_avg and SR
    bat_all  = pd.concat(bat_frames,  ignore_index=True) if bat_frames  else pd.DataFrame()
    bowl_all = pd.concat(bowl_frames, ignore_index=True) if bowl_frames else pd.DataFrame()

    # Convert numeric columns
    for col in ['Runs','Batting_Avg','Strike_Rate','Matches']:
        if col in bat_all.columns:
            bat_all[col] = pd.to_numeric(bat_all[col], errors='coerce').fillna(0)
    for col in ['Wickets','Bowling_Avg','Economy_Rate']:
        if col in bowl_all.columns:
            bowl_all[col] = pd.to_numeric(bowl_all[col], errors='coerce').fillna(0)

    # Remove header-repeat junk rows (where Player_Name is literally "Player")
    if not bat_all.empty and 'Player_Name' in bat_all.columns:
        bat_all = bat_all[~bat_all['Player_Name'].astype(str).str.strip().isin(['Player','player','PLAYER'])]
    if not bowl_all.empty and 'Player_Name' in bowl_all.columns:
        bowl_all = bowl_all[~bowl_all['Player_Name'].astype(str).str.strip().isin(['Player','player','PLAYER'])]

    # Aggregate batting per player
    if not bat_all.empty:
        bat_agg = bat_all.groupby('Player_Name', as_index=False).agg(
            Country      = ('Country',     lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'),
            Total_Runs   = ('Runs',        'sum'),
            Batting_Avg  = ('Batting_Avg', 'mean'),
            Strike_Rate  = ('Strike_Rate', 'mean'),
            Total_Matches= ('Matches',     'sum') if 'Matches' in bat_all.columns else ('Player_Name', 'count'),
            Age_Group    = ('Age_Group',   lambda x: x.mode().iloc[0] if 'Age_Group' in bat_all.columns and len(x) > 0 else 'Senior'),
        )
    else:
        bat_agg = pd.DataFrame()

    # Aggregate bowling per player
    if not bowl_all.empty:
        bowl_agg = bowl_all.groupby('Player_Name', as_index=False).agg(
            Total_Wickets = ('Wickets',     'sum'),
            Bowling_Avg   = ('Bowling_Avg', 'mean'),
            Economy_Rate  = ('Economy_Rate','mean'),
        )
    else:
        bowl_agg = pd.DataFrame()

    # Merge batting + bowling
    if not bat_agg.empty and not bowl_agg.empty:
        merged = bat_agg.merge(bowl_agg, on='Player_Name', how='outer')
    elif not bat_agg.empty:
        merged = bat_agg
    else:
        merged = bowl_agg

    # Fill missing values with sensible defaults
    merged['Total_Runs']    = merged.get('Total_Runs',    pd.Series([0]*len(merged))).fillna(0)
    merged['Batting_Avg']   = merged.get('Batting_Avg',  pd.Series([0]*len(merged))).fillna(0)
    merged['Strike_Rate']   = merged.get('Strike_Rate',  pd.Series([100]*len(merged))).fillna(100)
    merged['Total_Wickets'] = merged.get('Total_Wickets',pd.Series([0]*len(merged))).fillna(0)
    merged['Bowling_Avg']   = merged.get('Bowling_Avg',  pd.Series([30]*len(merged))).fillna(30)
    merged['Economy_Rate']  = merged.get('Economy_Rate', pd.Series([8]*len(merged))).fillna(8)
    merged['Total_Matches'] = merged.get('Total_Matches',pd.Series([5]*len(merged))).fillna(5)
    merged['Age_Group']     = merged.get('Age_Group',    pd.Series(['Senior']*len(merged))).fillna('Senior')
    merged['Country']       = merged.get('Country',      pd.Series(['Unknown']*len(merged))).fillna('Unknown')

    # Assign Role based on stats
    def assign_role(row):
        runs = row.get('Total_Runs', 0)
        wkts = row.get('Total_Wickets', 0)
        
        # Calculate relative impact (1 wicket ~ 25 runs)
        bat_score = runs
        bowl_score = wkts * 25
        
        # Elite in both with relatively balanced impact
        if bat_score > 300 and bowl_score > 300 and abs(bat_score - bowl_score) < max(bat_score, bowl_score) * 0.6:
            return 'All-Rounder'
            
        # Significant difference in impact dictates primary role
        if bat_score > bowl_score * 1.5:
            return 'Batsman'
        elif bowl_score > bat_score * 1.5:
            return 'Bowler'
        else:
            return 'All-Rounder'
            
    merged['Role'] = merged.apply(assign_role, axis=1)

    # Add form index (simulate based on batting avg percentile since real form not in dataset)
    bat_avg_vals = pd.to_numeric(merged['Batting_Avg'], errors='coerce').fillna(0)
    max_avg = bat_avg_vals.max() if bat_avg_vals.max() > 0 else 1
    merged['Form_Index_Last5'] = ((bat_avg_vals / max_avg) * 8 + 2).round(1).clip(2, 9.5)

    # Drop duplicates
    merged = merged.drop_duplicates(subset='Player_Name').reset_index(drop=True)

    return merged


def run_pipeline_on_df(raw_df):
    """Run feature engineering + clustering on a raw dataframe and return processed df."""
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    engineered = feature_engineering(raw_df.copy())
    final = train_and_persist_model(engineered)
    return final


# ---- Load Data: uploaded file takes priority, then fallback to processed CSV ----
# ---- State Management ----
if 'clear_data' not in st.session_state:
    st.session_state.clear_data = False

# ---- Load Data: uploaded file takes priority, then fallback to processed CSV ----
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
    # Fallback: load from pre-processed CSV if it exists
    if not st.session_state.clear_data and os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)

if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.clear_data = True
    st.rerun()

if df is None or df.empty:
    st.info("**👋 Welcome to Talent Scouting Portal!**")
    st.warning("No data loaded yet. Please upload your player dataset using the sidebar panel on the left.")
    demo_cols = ['Player_Name', 'Country', 'Age_Group', 'Role', 'Scout_Rating', 'Performance_Tier', 'Total_Matches',
                 'Total_Runs', 'Batting_Avg', 'Strike_Rate', 'Total_Wickets',
                 'Bowling_Avg', 'Economy_Rate', 'Form_Index_Last5', 'Batting_Impact', 'Bowling_Impact']
    df = pd.DataFrame(columns=demo_cols)

# ---- Sidebar Interactive Parameters Matrix ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Players")

countries = st.sidebar.multiselect("Select Country Pool", options=sorted(df['Country'].unique()), default=sorted(df['Country'].unique()))
roles_list = sorted(list(set(['Batsman', 'Bowler', 'All-Rounder'] + list(df['Role'].unique()))))
roles = st.sidebar.multiselect("Select Player Role", options=roles_list, default=roles_list)
tiers = st.sidebar.multiselect("Select Model Ranked Tiers", options=sorted(df['Performance_Tier'].unique()), default=sorted(df['Performance_Tier'].unique()))

# Checkbox logic for analytical classification isolating young stars
u19_u23_only = st.sidebar.checkbox("Isolate Under-19 & Under-23 Talents Only", value=False)

# Query Structural processing execulations
if not df.empty:
    filtered_df = df[
        (df['Country'].isin(countries)) &
        (df['Role'].isin(roles)) &
        (df['Performance_Tier'].isin(tiers))
    ]
    if u19_u23_only:
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(['U19', 'U23'])]
else:
    filtered_df = df.copy()

# Metrics Wrapper Rows
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><h4>Total Scanned Assets</h4><h2>{len(filtered_df)}</h2></div>', unsafe_allow_html=True)
with m2:
    tier1_count = len(filtered_df[filtered_df['Performance_Tier'] == 'Tier_1']) if not filtered_df.empty else 0
    st.markdown(f'<div class="metric-box"><h4>Tier 1 (Elite) Prospects</h4><h2>{tier1_count}</h2></div>', unsafe_allow_html=True)
with m3:
    avg_rating = round(filtered_df['Scout_Rating'].mean(), 2) if not filtered_df.empty else 0
    st.markdown(f'<div class="metric-box"><h4>Mean Scout Rating</h4><h2>{avg_rating}</h2></div>', unsafe_allow_html=True)
with m4:
    form_col = 'Form_Index_Last5' if 'Form_Index_Last5' in filtered_df.columns else 'Form_Index_last5'
    avg_form = filtered_df[form_col].mean() if not filtered_df.empty else 0
    st.markdown(f'<div class="metric-box"><h4>Global Squad Form Index</h4><h2>{avg_form:.1f}/10</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- HIGH-DIMENTIONAL DATA FRAMES VIEW ---
st.subheader("🏏 Machine Learning Evaluated Talent Pool")
display_cols = ['Player_Name', 'Country', 'Age_Group', 'Role', 'Scout_Rating', 'Performance_Tier', form_col]
available_cols = [c for c in display_cols if c in filtered_df.columns]
st.dataframe(filtered_df[available_cols].sort_values(by="Scout_Rating", ascending=False).reset_index(drop=True) if not filtered_df.empty else filtered_df[available_cols], use_container_width=True)

# ---- GRAPHICAL SCATTER PLOT ANALYSIS --- 
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("  Performance Segmentation Space Mapping")

chart_template = "plotly_dark" if dark_mode else "plotly_white"
tier2_color = "#AAAAAA" if dark_mode else "#333333"

if not filtered_df.empty:
    fig = px.scatter(
        filtered_df,
        x="Batting_Impact",
        y="Bowling_Impact",
        color="Performance_Tier",
        size="Scout_Rating",
        hover_name='Player_Name',
        hover_data=['Age_Group', 'Country', form_col],
        color_discrete_map={"Tier_1": "#D4AF37", "Tier_2": tier2_color, "Tier_3": "#808080"},
        title="2D Performance Impact Scatter — Player Distribution",
        template=chart_template
    )
else:
    # Create empty scatter plot
    fig = px.scatter(
        title="2D Performance Impact Scatter — Player Distribution (No Data)",
        template=chart_template
    )

fig.update_layout(
    paper_bgcolor="#1A1A1A" if dark_mode else "#FFFFFF",
    plot_bgcolor="#1E1E1E" if dark_mode else "#F9F9F9",
    font=dict(color="#FFFFFF" if dark_mode else "#000000", family="Outfit"),
    title_font=dict(size=18, color="#D4AF37"),
    legend=dict(
        bgcolor="#242424" if dark_mode else "#FFFFFF",
        bordercolor="#D4AF37",
        borderwidth=1,
        font=dict(color="#FFFFFF" if dark_mode else "#000000")
    ),
    xaxis=dict(gridcolor="#333333" if dark_mode else "#EAEAEA", color="#FFFFFF" if dark_mode else "#000000"),
    yaxis=dict(gridcolor="#333333" if dark_mode else "#EAEAEA", color="#FFFFFF" if dark_mode else "#000000"),
)
st.plotly_chart(fig, use_container_width=True)

# --- Real-Time Machine Learning Scouting Predictor ---
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
            # Load Persistent Serialization engine architecture
            model = joblib.load(model_path)

            # MAth Transformations aligning to feature scaling baselines
            calc_bat_impact = (in_runs* 0.4) + (in_avg_bat * 0.3) + (in_sr * 0.3)
            calc_bowl_impact = (in_wkts* 0.5) + (100/ (in_avg_bowl + 1e-5) * 0.25) + (100/ (in_econ + 1e-5) * 0.25)

            # Composite Score Calculations
            calc_rating = (calc_bat_impact* 0.4) + (calc_bowl_impact * 0.4) + (in_form * 2.0)

            # Array translation formatting for model input array matching
            input_vector = np.array ([[calc_bat_impact, calc_bowl_impact, in_form, calc_rating]])

            # Predict Vector Point Label Class 
            predicted_cluster = model.predict(input_vector) [0]


            # Context Aware Fallback Ranking Mapping
            st.success(f"**Prediction Engine Inference Completed Successfully!**")
            st.markdown(f"**Calculated Core Scout Rating: ** '{calc_rating:.2f}'")
            st.info(f"The algorithm has classified this player profile as an asset belonging to the evaluated group cluster segment ID: ** Cluster {predicted_cluster} **")
        else:
            # Basic Procedural Calculation Logic Framework Safely Running When Model Object Are Not Cached
            simulated_rating = min (((in_runs * 0.1) + (in_sr * 0.2) + (in_wkts * 2.5) + (in_form * 4)), 100.0)
            st.warning("Using Analytical Rule Based Inference (Trained Model File (.pkl) not found on disk). ")
            st.success(f" Estimated Rating Matrix Score: '{simulated_rating:.2f}'")
