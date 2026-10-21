import pandas as pd
import numpy as np

try:
    df = pd.read_csv('/Users/franklinsantaella/CascadeProjects/nba_advanced_stats/ML/data/predictions/betting_tracking.csv')
    print(f"Total records in CSV: {len(df)}")

    # Filtrar solo los que tienen resultados reales
    df_eval = df[df['actual_home'].notna()].copy()
    print(f"Records with actual results: {len(df_eval)}")

    for market in ['Visitante', 'Local', 'Partido']:
        m_df = df_eval[df_eval['market'] == market].copy()
        if len(m_df) > 0:
            correct = m_df['was_correct'].map({'True': True, 'False': False, True: True, False: False}).sum()
            total = len(m_df)
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"Market {market}: {correct}/{total} ({accuracy:.2f}%)")

    print("\nValue counts of 'was_correct':")
    print(df_eval['was_correct'].value_counts())
    
    print("\nExample records (first 5 with results):")
    print(df_eval[['game_date', 'market', 'bet_type', 'threshold', 'was_correct']].head())

except Exception as e:
    print(f"Error: {e}")
