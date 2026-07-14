import os  
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib 


def load_and_merge_data():
    """
    data/raw folder will load excel files to merge into one dataframe
    """
    raw_dir = os.path.join("data","raw")
    all_data = []

    # Check the directory exists
    if not os.path.exists(raw_dir):
        # local path to the raw data folder
        os.makedirs(os.path.join("data","processed"), exist_ok= True)
        os.makedirs(os.path.join("models"), exist_ok= True)
        return None

    for file in os.listdir(raw_dir):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(raw_dir, file)
            print(f"[INFO] Loading raw file: {file_path}")

            # Excel files can have multiple sheets so we need to load all sheets and merge them
            excel_data = pd.ExcelFile(file_path)
            for sheet_name in excel_data.sheet_names:
                df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
                all_data.append(df_sheet)
                # ensure baseline string mapping cleans spaces
                df_sheet.columns = df_sheet.columns.str.strip()
                all_data.append(df_sheet)

    if len(all_data) == 0:
        return None

    return pd.concat(all_data, ignore_index=True)

def feature_engineering(df):
    """
    Cricket Domain Specific Feature Engineering
    """
    print("[INFO] Executing Feature Engineering Pipeline")

    # Handle missing values / Data Cleaning
    df['Total_Runs'] = pd.to_numeric(df['Total_Runs'], errors='coerce').fillna(0)
    df['Batting_Avg'] = pd.to_numeric(df['Batting_Avg'], errors='coerce').fillna(0)
    df['Strike_Rate'] = pd.to_numeric(df['Strike_Rate'], errors='coerce').fillna(0)


    df['Total_Wickets'] = pd.to_numeric(df['Total_Wickets'], errors='coerce').fillna(0)
    df['Bowling_Avg'] = pd.to_numeric(df['Bowling_Avg'], errors='coerce').fillna(30.0)
    df['Economy_Rate'] = pd.to_numeric(df['Economy_Rate'], errors='coerce').fillna(25.0)

    df['Form_Index_last5'] = pd.to_numeric(df['Form_Index_Last5'], errors='coerce').fillna(5.0)

    # Mathematical Domain Formulation (Impact Scores)
    # Batting Impact Score Formula
    df['Batting_Impact'] = (df['Total_Runs'] * 0.4 +df['Batting_Avg'] * 0.3 + df['Strike_Rate'] * 0.3)

    # Bowling Impact Score Formula (Adding 1e-5 epsilon value to prevent mathematical Inf/NaN syntax errors)
    df['Bowling_Impact'] = (df['Total_Wickets'] * 0.5 + (100 / (df['Bowling_Avg'] + 1e-5)) * 0.25 + df['Economy_Rate'] + 1e-5) * 0.25

    # Scaling the Impact Scores using MinMaxScaler (0-100 Range Boundary)
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['Batting_Impact', 'Bowling_Impact']] = scaler.fit_transform(df[['Batting_Impact', 'Bowling_Impact']])

    # Composite Scout Rating Formula 
    df['Scout_Rating'] = (df['Batting_Impact'] * 0.4 + df['Bowling_Impact'] * 0.4) + df['Form_Index_last5'] * 2.0
    df['Scout_Rating'] = scaler.fit_transform(df[['Scout_Rating']])

    return df


def train_and_persist_model(df):
    """
    Unsupervised Learning Model Training and Persistence
    """
    print("[INFO] Training Clustering Model (KMeans)")

    # Selecting numerical vectors for ML Matrix MApping
    features = ['Batting_Impact', 'Bowling_Impact','Form_Index_last5', 'Scout_Rating']
    X = df[features]

    # Training Clustering Node
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_Labels'] = kmeans.fit_predict(X)

    # Dynamic Sorting Strategy : Map Raw cluster labels to meaningful cluster names based on Scout Rating 
    mean_ratings = df.groupby('Cluster_Labels')['Scout_Rating'].mean().sort_values(ascending=False)
    tier_mapping = {cluster_id: f'Tier_{rank+1}' for rank, cluster_id in enumerate(mean_ratings.index)}

    df['Performance_Tier'] = df['Cluster_Labels'].map(tier_mapping)

    # Save Processed Dataset
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"[INFO] Processed data saved to: {processed_path}")

    # Persist serialized training checkpoints
    model_path = os.path.join("models", "talent_scout_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"[SUCCESS] Serialized ML Pipeline checkpoint saved at: {model_path}")

    return df   

if __name__ == "__main__":
    # Execution Block Pipeline Flow Controller
    raw_df = load_and_merge_data()
    if raw_df is not None:
        engineered_df = feature_engineering(raw_df)
        final_df = train_and_persist_model(engineered_df)
        print("[FINISHED] Pipeline process is completed successfully.")  
    else:
        print("[ERROR] No Excel dataset sheets detected inside 'data/raw' directory. Please add Excel files to proceed.")






































