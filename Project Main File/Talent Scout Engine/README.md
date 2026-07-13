# 🏏 Global Cricket Talent Scout Engine

## Project Overview
Talent Scout Engine ek Machine Learning aur Data Science based project hai jo ICC tournaments aur T20 Franchise leagues (2025-26) ke data ka analysis karta hai. Ye system specially Under-19 aur Under-23 ke emerging talents ko unki form, batting impact, aur bowling impact ke basis par evaluate aur tier-rank karta hai.

## Features
- **Data Pipeline:** Raw Excel/CSV files ko clean aur process karta hai.
- **K-Means Clustering:** Unsupervised learning ka use karke players ko automatically 3 Performance Tiers (Elite, Mid, Low) mein divide karta hai.
- **Streamlit Dashboard:** Ek interactive web app jahan users players ko filter, analyze aur visually map kar sakte hain.
- **Real-Time Predictor:** Naye player stats input karke unka real-time scout rating aur tier check karne ka feature.

## How to Run This Project Locally
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Place your raw data in `data/raw/` directory.
6. Run the ML pipeline: `python src/data_pipeline.py`
7. Start the dashboard: `streamlit run app.py`