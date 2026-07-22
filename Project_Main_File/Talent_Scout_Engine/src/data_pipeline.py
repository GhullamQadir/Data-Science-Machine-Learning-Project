import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deduplicate_columns(columns):
    counts = {}
    result = []
    for column in columns:
        name = str(column).strip() or "unnamed"
        count = counts.get(name, 0)
        result.append(name if count == 0 else f"{name}_{count}")
        counts[name] = count + 1
    return result


def _safe_col(df, primary, aliases=None, default=0.0):
    """
    Safely read a numeric column from df.
    - Tries `primary` first.
    - Falls back through `aliases` if primary is missing.
    - Returns a Series filled with `default` if nothing matches.
    This prevents KeyError when an uploaded file has different column names.
    """
    normalized = {str(column).strip().lower().replace(' ', '_'): column for column in df.columns}
    candidates = [primary] + (aliases or [])
    for candidate in candidates:
        source = normalized.get(str(candidate).strip().lower().replace(' ', '_'))
        if source is not None:
            return pd.to_numeric(df[source], errors='coerce').fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def load_and_merge_data():
    """
    Load Excel files from data/raw folder and merge into one DataFrame.
    """
    raw_dir = os.path.join(BASE_DIR, "data", "Raw_data")
    all_data = []

    if not os.path.exists(raw_dir):
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        os.makedirs(os.path.join("models"), exist_ok=True)
        return None

    for file in os.listdir(raw_dir):
        if file.lower().endswith((".xlsx", ".xls")):
            file_path = os.path.join(raw_dir, file)
            print(f"[INFO] Loading raw file: {file_path}")
            excel_data = pd.ExcelFile(file_path)
            for sheet_name in excel_data.sheet_names:
                raw_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                header_row = next(
                    (
                        index for index, row in raw_sheet.iterrows()
                        if any(str(value).strip().lower() in {"player", "player_name", "player name"}
                               for value in row if pd.notna(value))
                    ),
                    None,
                )
                if header_row is None:
                    continue
                df_sheet = raw_sheet.iloc[header_row + 1:].copy()
                df_sheet.columns = _deduplicate_columns([
                    str(value).strip().replace(" ", "_") if pd.notna(value) else f"unnamed_{index}"
                    for index, value in enumerate(raw_sheet.iloc[header_row])
                ])
                player_column = next(
                    (column for column in df_sheet.columns
                     if str(column).lower() in {"player", "player_name", "player_name_1"}),
                    None,
                )
                if player_column is None:
                    continue
                df_sheet = df_sheet[df_sheet[player_column].notna()]
                df_sheet["_Tournament"] = sheet_name
                all_data.append(df_sheet)

    if len(all_data) == 0:
        return None

    merged = pd.concat(all_data, ignore_index=True, sort=False)
    column_lookup = {str(column).strip().lower(): column for column in merged.columns}
    player_column = next(
        (column_lookup[name] for name in ("player", "player_name", "player name")
         if name in column_lookup),
        None,
    )
    if player_column is None:
        raise ValueError("No player column was found in the raw workbook.")
    if player_column != "Player_Name":
        merged = merged.rename(columns={player_column: "Player_Name"})
    if "Team" in merged.columns and "Country" not in merged.columns:
        merged = merged.rename(columns={"Team": "Country"})
    merged["Player_Name"] = merged["Player_Name"].astype(str).str.strip()
    merged = merged[~merged["Player_Name"].str.lower().isin({"", "nan", "player"})]
    return merged.reset_index(drop=True)


def feature_engineering(df):
    """
    Cricket Domain Specific Feature Engineering — Batting, Bowling, and Fielding.
    Uses safe column access so uploads with varied column names never crash.
    """
    print("[INFO] Executing Feature Engineering Pipeline")

    # ---- Batting (safe: accepts Total_Runs or Runs) ----
    df['Total_Runs']  = _safe_col(df, 'Total_Runs',  aliases=['Runs', 'Run', 'R'],                     default=0.0)
    df['Batting_Avg'] = _safe_col(df, 'Batting_Avg', aliases=['Avg', 'Average', 'Bat_Avg'],            default=0.0)
    df['Strike_Rate'] = _safe_col(df, 'Strike_Rate', aliases=['SR', 'BSR', 'StrikeRate'],              default=0.0)

    # ---- Bowling (safe: accepts Total_Wickets or Wickets) ----
    df['Total_Wickets'] = _safe_col(df, 'Total_Wickets', aliases=['Wickets', 'Wkts', 'W'],             default=0.0)
    df['Bowling_Avg']   = _safe_col(df, 'Bowling_Avg',   aliases=['Bowl_Avg', 'Avg'],                  default=30.0)
    df['Economy_Rate']  = _safe_col(df, 'Economy_Rate',  aliases=['Economy', 'Econ', 'ER'],            default=8.0)

    # ---- Form (safe: accepts Form_Index_Last5 or Form_Index_last5) ----
    df['Form_Index_last5'] = _safe_col(df, 'Form_Index_Last5',
                                        aliases=['Form_Index_last5', 'Form_Index', 'Form'],             default=5.0)

    # ---- Fielding (safe: defaults to 0 if columns absent) ----
    df['Total_Catches']    = _safe_col(df, 'Total_Catches',    aliases=['Catches', 'Catch', 'CT'],     default=0.0)
    df['Total_Stumpings']  = _safe_col(df, 'Total_Stumpings',  aliases=['Stumpings', 'Stumping', 'ST'],default=0.0)
    df['Total_RunOuts']    = _safe_col(df, 'Total_RunOuts',    aliases=['Run_Outs', 'RunOuts', 'RO'],  default=0.0)
    df['Total_Dismissals'] = _safe_col(df, 'Total_Dismissals', aliases=['Dismissals', 'Dis'],          default=0.0)

    # ---- Impact Score Formulas ----
    df['Batting_Impact'] = df['Total_Runs'] * 0.4 + df['Batting_Avg'] * 0.3 + df['Strike_Rate'] * 0.3

    df['Bowling_Impact'] = (
        df['Total_Wickets'] * 0.5 +
        (100 / (df['Bowling_Avg'] + 1e-5)) * 0.25 +
        (df['Economy_Rate'] + 1e-5)
    ) * 0.25

    df['Fielding_Impact'] = (
        df['Total_Catches']   * 0.4 +
        df['Total_Stumpings'] * 0.3 +
        df['Total_RunOuts']   * 0.3
    )

    # ---- Scale all impact scores to 0-100 ----
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['Batting_Impact', 'Bowling_Impact', 'Fielding_Impact']] = scaler.fit_transform(
        df[['Batting_Impact', 'Bowling_Impact', 'Fielding_Impact']]
    )

    # ---- Composite Scout Rating: Bat 35% + Bowl 35% + Field 10% + Form 20% ----
    df['Scout_Rating'] = (
        df['Batting_Impact']   * 0.35 +
        df['Bowling_Impact']   * 0.35 +
        df['Fielding_Impact']  * 0.10 +
        df['Form_Index_last5'] * 2.0
    )
    df['Scout_Rating'] = scaler.fit_transform(df[['Scout_Rating']])

    return df


def train_and_persist_model(df):
    """
    Train KMeans clustering model and persist model + processed CSV to disk.
    """
    print("[INFO] Training Clustering Model (KMeans)")

    features = ['Batting_Impact', 'Bowling_Impact', 'Fielding_Impact', 'Form_Index_last5', 'Scout_Rating']
    X = df[features]

    cluster_count = min(3, len(df))
    if cluster_count < 1:
        raise ValueError("At least one player is required to train the scouting model.")
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    df['Cluster_Labels'] = kmeans.fit_predict(X)

    mean_ratings = df.groupby('Cluster_Labels')['Scout_Rating'].mean().sort_values(ascending=False)
    tier_mapping = {cluster_id: f'Tier_{rank+1}' for rank, cluster_id in enumerate(mean_ratings.index)}
    df['Performance_Tier'] = df['Cluster_Labels'].map(tier_mapping)

    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    processed_path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"[INFO] Processed data saved to: {processed_path}")

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "models", "talent_scout_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"[SUCCESS] ML model checkpoint saved at: {model_path}")

    return df


if __name__ == "__main__":
    raw_df = load_and_merge_data()
    if raw_df is not None:
        engineered_df = feature_engineering(raw_df)
        final_df = train_and_persist_model(engineered_df)
        print("[FINISHED] Pipeline process completed successfully.")
    else:
        print("[ERROR] No Excel files found in 'data/raw'. Please add your dataset and retry.")
