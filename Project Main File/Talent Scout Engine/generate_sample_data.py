import pandas as pd
import numpy as np
import os

# Create data/raw directory if it doesn't exist
os.makedirs(os.path.join("data", "raw"), exist_ok=True)

# Generate synthetic player data
np.random.seed(42)
num_players = 60

countries = ['India', 'Australia', 'England', 'South Africa', 'New Zealand', 'Pakistan']
roles = ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper']
age_groups = ['U19', 'U23', 'Senior']

# Properly paired cricket player names — same index maps first to last name correctly
player_names = [
    "Virat Kohli", "Steve Smith", "Joe Root", "Babar Azam", "Kane Williamson",
    "Pat Cummins", "Jasprit Bumrah", "Kagiso Rabada", "Trent Boult", "Shaheen Afridi",
    "Rohit Sharma", "David Warner", "Ben Stokes", "Marnus Labuschagne", "Quinton de Kock",
    "Rishabh Pant", "Shubman Gill", "Rashid Khan", "Jofra Archer", "KL Rahul",
    "Hardik Pandya", "Glenn Maxwell", "Mitchell Starc", "Shakib Al Hasan", "Jason Holder",
    "Jos Buttler", "Jonny Bairstow", "Aaron Finch", "Faf du Plessis", "Suryakumar Yadav",
    "Muhammad Rizwan", "Hasan Ali", "Imam ul-Haq", "Fakhar Zaman", "Shadab Khan",
    "Moeen Ali", "Adil Rashid", "Mark Wood", "Chris Woakes", "Dawid Malan",
    "Usman Khawaja", "Travis Head", "Cameron Green", "Marcus Stoinis", "Adam Zampa",
    "Josh Hazlewood", "Devon Conway", "Tom Latham", "Tim Southee", "Matt Henry",
    "Lockie Ferguson", "Kusal Mendis", "Dimuth Karunaratne", "Angelo Mathews", "Wanindu Hasaranga",
    "Dushmantha Chameera", "Litton Das", "Mushfiqur Rahim", "Mustafizur Rahman", "Mehidy Miraz"
]

data = {
    'Player_Name': player_names,
    'Country': np.random.choice(countries, num_players),
    'Age_Group': np.random.choice(age_groups, num_players),
    'Role': np.random.choice(roles, num_players),
    'Total_Matches': np.random.randint(5, 100, num_players),
    'Total_Runs': np.random.randint(50, 4000, num_players),
    'Batting_Avg': np.random.uniform(10.0, 60.0, num_players).round(2),
    'Strike_Rate': np.random.uniform(80.0, 160.0, num_players).round(2),
    'Total_Wickets': np.random.randint(0, 150, num_players),
    'Bowling_Avg': np.random.uniform(15.0, 50.0, num_players).round(2),
    'Economy_Rate': np.random.uniform(4.5, 10.0, num_players).round(2),
    'Form_Index_Last5': np.random.uniform(2.0, 9.5, num_players).round(1)
}

# Adjust some values based on role for realism
for i in range(num_players):
    if data['Role'][i] == 'Batsman':
        data['Total_Wickets'][i] = np.random.randint(0, 10)
    elif data['Role'][i] == 'Bowler':
        data['Total_Runs'][i] = np.random.randint(10, 500)
        data['Batting_Avg'][i] = np.random.uniform(5.0, 20.0)

df = pd.DataFrame(data)

# Save to Excel
output_path = os.path.join("data", "raw", "sample_players.xlsx")
df.to_excel(output_path, index=False)
print(f"Sample data generated and saved to {output_path}")
