import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

def load_and_clean_data():
    print("--- Fetching Actuarial Datasets ---")
    # freq contains policy characteristics and number of claims
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser='pandas').frame
    # sev contains the cost of the claims
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser='pandas').frame

    # 1. Aggregate Severity to Policy ID level
    df_sev_adj = df_sev.groupby('IDpol').agg({'ClaimAmount': 'sum'}).reset_index()

    # 2. Merge with Frequency data
    df = pd.merge(df_freq, df_sev_adj, on='IDpol', how='left')
    
    # 3. Fill missing claim amounts with 0
    df['ClaimAmount'] = df['ClaimAmount'].fillna(0)
    
    # 4. Filter Exposure: Actuarially, we only care about records with time at risk
    df = df[df['Exposure'] > 0].copy()
    
    # 5. Actuarial Cap: Cap at 99th percentile for stability
    cap = df['ClaimAmount'].quantile(0.99)
    df['ClaimAmount_Capped'] = df['ClaimAmount'].clip(upper=cap)
    
    # Drop raw amount to ensure we don't use it by mistake later
    df = df.drop(columns=['ClaimAmount'])

    print(f"Dataset Loaded: {df.shape[0]} rows. Capped claims at: €{cap:.2f}")
    return df

def perform_eda(df):
    print("--- Running Actuarial EDA ---")
    
    # Average Pure Premium (Total Cost / Total Exposure)
    avg_pure_premium = df['ClaimAmount_Capped'].sum() / df['Exposure'].sum()
    print(f"Market Average Pure Premium: {avg_pure_premium:.2f}€ per year")

    # Plot Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df[df['ClaimAmount_Capped'] > 0]['ClaimAmount_Capped'], bins=50, kde=True)
    plt.title("Distribution of Non-Zero Claim Amounts (Severity)")
    plt.xlabel("Claim Amount (€)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    df = load_and_clean_data()
    perform_eda(df)
    
    # Save for Step 2
    df.to_pickle("processed_step_1.pkl")
    print("\nStep 1 Complete. Data saved to 'processed_step_1.pkl'")