import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

print("Starting COVID-19 analysis...")

try:
    # --- 1. LOAD DATA ---
    cases_df = pd.read_csv('C:\Project\Covid-19\covid_19_india.csv')
    vaccine_df = pd.read_csv('C:\Project\Covid-19\covid_vaccine_statewise.csv')

    # --- 2. PREPARE CASES DATA (covid_19_india.csv) ---
    cases_df['Date'] = pd.to_datetime(cases_df['Date'], format='%Y-%m-%d')
    cases_df = cases_df.dropna(subset=['Date', 'State/UnionTerritory'])
    cases_df = cases_df.sort_values(by=['State/UnionTerritory', 'Date'])
    
    # Calculate daily new cases/deaths from cumulative data
    cases_df['Daily Confirmed'] = cases_df.groupby('State/UnionTerritory')['Confirmed'].diff().fillna(cases_df['Confirmed'])
    cases_df['Daily Deaths'] = cases_df.groupby('State/UnionTerritory')['Deaths'].diff().fillna(cases_df['Deaths'])
    
    # Correct for any data anomalies (e.g., negative new cases)
    cases_df['Daily Confirmed'] = cases_df['Daily Confirmed'].clip(lower=0)
    cases_df['Daily Deaths'] = cases_df['Daily Deaths'].clip(lower=0)
    
    cases_df_daily = cases_df[['Date', 'State/UnionTerritory', 'Daily Confirmed', 'Daily Deaths']]

    # --- 3. PREPARE VACCINATION DATA (covid_vaccine_statewise.csv) ---
    vaccine_df['Date'] = pd.to_datetime(vaccine_df['Updated On'], format='%d/%m/%Y')
    vaccine_df = vaccine_df.rename(columns={'State': 'State/UnionTerritory'})
    
    # Filter out the aggregate "India" row to avoid double-counting
    vaccine_df_states = vaccine_df[vaccine_df['State/UnionTerritory'] != 'India'].copy()
    
    vaccine_cols = ['Date', 'State/UnionTerritory', 'Total Individuals Vaccinated']
    vaccine_df_clean = vaccine_df_states[vaccine_cols]

    # --- 4. MERGE DATASETS ---
    # Merge daily cases with daily vaccination data
    merged_df = pd.merge(
        cases_df_daily, 
        vaccine_df_clean, 
        on=['Date', 'State/UnionTerritory'], 
        how='left'
    )
    
    # Fill vaccine data with 0 for dates before the drive started (Jan 2021)
    merged_df['Total Individuals Vaccinated'] = merged_df['Total Individuals Vaccinated'].fillna(0)

    # --- 5. AGGREGATE TO NATIONAL LEVEL ---
    # Sum all state data to get a daily national total
    national_df = merged_df.groupby('Date').sum().reset_index()

    print("Data loaded and merged successfully.")

    # --- 6. ANALYSIS 1: CORRELATION WITH TIME LAG ---
    # We test if vaccinations (with a lag) correlate with daily deaths.
    # Immunity isn't instant. Let's use a 21-day lag.
    lag_days = 21
    national_df['Vax_Lagged'] = national_df['Total Individuals Vaccinated'].shift(lag_days).fillna(0)
    
    # We can only analyze the period *after* vaccinations started
    analysis_df = national_df[national_df['Vax_Lagged'] > 0].copy()
    
    if not analysis_df.empty:
        correlation = analysis_df['Vax_Lagged'].corr(analysis_df['Daily Deaths'])
        print("\n--- Correlation Analysis ---")
        print(f"Correlation between {lag_days}-day lagged vaccinations and Daily Deaths: {correlation:.4f}")
    else:
        print("\n--- Correlation Analysis ---")
        print("Not enough lagged data to perform correlation analysis.")

    # --- 7. ANALYSIS 2: LINEAR REGRESSION ---
    if not analysis_df.empty:
        print("\n--- Linear Regression Analysis ---")
        
        # Prepare data for scikit-learn
        # We need to predict 'Daily Deaths' (y) using 'Vax_Lagged' (X)
        X = analysis_df[['Vax_Lagged']].values  # .values creates a NumPy array
        y = analysis_df['Daily Deaths'].values
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # y = mx + c
        m = model.coef_[0]
        c = model.intercept_
        
        print(f"Regression Model: Daily Deaths = ({m:.6f} * Lagged_Vaccinations) + {c:.2f}")
        print("Interpretation: For every 1 million additional people vaccinated, the model predicts a change of",
              f"{m * 1_000_000:.0f} daily deaths {lag_days} days later.")

        # --- 8. VISUALIZATION ---
        print("\nGenerating plots... (Close the plot window to exit the script)")
        
        # Plot 1: National Daily Cases vs. Vaccinations
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_title('National Daily Cases vs. Total Vaccinations')
        ax1.set_xlabel('Date')
        
        ax1.plot(national_df['Date'], national_df['Daily Confirmed'], color='blue', label='Daily Confirmed Cases')
        ax1.set_ylabel('Daily Confirmed Cases', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a second y-axis for vaccinations
        ax2 = ax1.twinx()
        ax2.plot(national_df['Date'], national_df['Total Individuals Vaccinated'], color='green', label='Total Individuals Vaccinated')
        ax2.set_ylabel('Total Individuals Vaccinated (in 100-Millions)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        fig.tight_layout()
        plt.legend()
        plt.show()

        # Plot 2: Regression Plot (The "line algorithm" visualization)
        plt.figure(figsize=(14, 7))
        sns.regplot(data=analysis_df, x='Vax_Lagged', y='Daily Deaths', line_kws={"color":"red"})
        plt.title(f'Regression: {lag_days}-Day Lagged Vaccinations vs. Daily Deaths')
        plt.xlabel('Total Individuals Vaccinated (Lagged by 21 Days)')
        plt.ylabel('Daily Deaths')
        plt.ticklabel_format(style='plain', axis='x') # Disable scientific notation
        plt.show()

    else:
        print("Cannot perform regression or plotting due to insufficient data.")

except FileNotFoundError as e:
    print(f"\n--- ERROR ---")
    print(f"File not found: {e.filename}")
    print("Please make sure 'covid_19_india.csv' and 'covid_vaccine_statewise.csv' are in the same folder as the script.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\nAnalysis complete.")