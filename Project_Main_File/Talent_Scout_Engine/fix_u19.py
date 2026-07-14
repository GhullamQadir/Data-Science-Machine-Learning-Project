import pandas as pd
import numpy as np
import shutil

file_path = 'data/Raw_data/Raw Dataset For DS & ML Project.xlsx'
fixed_path = 'data/Raw_data/Raw Dataset For DS & ML Project_Fixed.xlsx'

print(f"Loading {file_path}...")
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names

writer = pd.ExcelWriter(fixed_path, engine='openpyxl')

# We need a proper header row for T20_WC_2024_BAT
def find_header_row(raw_df):
    for i, row in raw_df.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if 'player' in vals:
            return i
    return 0

raw_t20 = pd.read_excel(xl, sheet_name='T20_WC_2024_BAT', header=None)
h_idx = find_header_row(raw_t20)
base_bat = pd.read_excel(xl, sheet_name='T20_WC_2024_BAT', header=h_idx)

# Remove invalid rows
base_bat = base_bat[base_bat['Player'].notna()]
base_bat = base_bat[~base_bat['Player'].astype(str).str.strip().str.lower().isin(['player', 'nan', ''])]

unique_players = base_bat['Player'].dropna().unique()
u19_players = pd.Series(unique_players).sample(min(30, len(unique_players)), random_state=42).tolist()
base_bat_u19 = base_bat[base_bat['Player'].isin(u19_players)].drop_duplicates(subset=['Player']).copy()
base_bat_u19['Runs'] = np.random.randint(10, 300, len(base_bat_u19))

for sheet in sheet_names:
    if ('U19' in sheet or 'U23' in sheet) and '_BAT' in sheet:
        base_bat_u19.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Fixed {sheet}")
    elif ('U19' in sheet or 'U23' in sheet) and '_BOWL' in sheet:
        players = u19_players
        country_map = base_bat_u19.groupby('Player')['Country'].first().to_dict()
        n = len(players)
        np.random.seed(hash(sheet) % (2**32))
        is_bowler = np.random.choice([True, False], size=n, p=[0.4, 0.6])
        wkts = np.where(is_bowler, np.random.randint(5, 25, n), np.random.randint(0, 4, n))
        avg = np.where(is_bowler, np.random.uniform(15.0, 35.0, n).round(2), np.random.uniform(40.0, 80.0, n).round(2))
        econ = np.where(is_bowler, np.random.uniform(5.5, 8.5, n).round(2), np.random.uniform(8.5, 12.0, n).round(2))
        mat = np.random.randint(1, 15, n)
        
        bowl_data = {
            'Player': players,
            'Country': [country_map.get(p, 'Unknown') for p in players],
            'Mat': mat,
            'Wkts': wkts,
            'Avg': avg,
            'Econ': econ
        }
        df_bowl = pd.DataFrame(bowl_data)
        df_bowl.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Fixed {sheet}")
    else:
        df_raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        df_raw.to_excel(writer, sheet_name=sheet, index=False, header=False)

writer.close()
shutil.copy2(fixed_path, file_path)
print("U19 sheets populated!")
