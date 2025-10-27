import streamlit as st
import pandas as pd
import altair as alt
from scipy import stats  # Import stats for the T-test

# Set page title and layout
st.set_page_config(page_title="COVID-19 India Analysis", layout="wide")

@st.cache_data  # Cache the data to speed up the app
def load_data():
    """Loads, cleans, and merges both COVID cases and vaccination datasets."""
    try:
        # --- 1. Load Cases Data ---
        cases_df = pd.read_csv('covid_19_india.csv')
        cases_df['Date'] = pd.to_datetime(cases_df['Date'], format='%Y-%m-%d')
        cases_df = cases_df.dropna(subset=['Date', 'State/UnionTerritory'])
        cases_df = cases_df.sort_values(by=['State/UnionTerritory', 'Date'])
        
        # Calculate daily new cases/deaths
        cases_df['Daily Confirmed'] = cases_df.groupby('State/UnionTerritory')['Confirmed'].diff().fillna(0).clip(lower=0)
        cases_df['Daily Deaths'] = cases_df.groupby('State/UnionTerritory')['Deaths'].diff().fillna(0).clip(lower=0)
        cases_df_daily = cases_df[['Date', 'State/UnionTerritory', 'Daily Confirmed', 'Daily Deaths']]
        
        # --- 2. Load Vaccination Data ---
        vaccine_df = pd.read_csv('covid_vaccine_statewise.csv')
        vaccine_df.columns = vaccine_df.columns.str.strip()
        vaccine_df = vaccine_df[vaccine_df['State'] != 'India'].copy()
        
        # Drop rows where key vaccination data is missing
        vaccine_df.dropna(subset=['Total Individuals Vaccinated'], inplace=True)
        
        vaccine_df['Date'] = pd.to_datetime(vaccine_df['Updated On'], format='%d/%m/%Y')
        vaccine_df_clean = vaccine_df[['Date', 'State', 'Total Individuals Vaccinated']]
        vaccine_df_clean = vaccine_df_clean.rename(columns={'State': 'State/UnionTerritory'})
        
        # --- 3. Merge Dataframes ---
        merged_df = pd.merge(
            cases_df_daily,
            vaccine_df_clean,
            on=['Date', 'State/UnionTerritory'],
            how='left'
        )
        
        # Forward-fill vaccination data (assumes vax numbers are cumulative and non-decreasing)
        merged_df['Total Individuals Vaccinated'] = merged_df.groupby('State/UnionTerritory')['Total Individuals Vaccinated'].ffill().fillna(0)
        
        return merged_df

    except FileNotFoundError as e:
        st.error(f"Error: {e.filename} not found. Please make sure both 'covid_19_india.csv' and 'covid_vaccine_statewise.csv' are in the same folder as app.py")
        return pd.DataFrame()

# --- Build the Streamlit Page ---
st.title("🇮🇳 COVID-19 India Analysis Dashboard")

df_merged = load_data()

if not df_merged.empty:
    
    # --- SECTION 1: State-wise Vaccination Charts ---
    st.header("State-wise Vaccination Performance")
    
    # Find the latest data for each state from the merged file
    df_latest = df_merged.sort_values('Date').drop_duplicates(subset=['State/UnionTerritory'], keep='last')
    
    # --- Top 15 States ---
    st.subheader("Top 15 States by Total Individuals Vaccinated")
    top_15_states = df_latest.nlargest(15, 'Total Individuals Vaccinated')

    # Create the Interactive Bar Chart for Top 15
    chart_top = alt.Chart(top_15_states).mark_bar(color="cornflowerblue").encode(
        y=alt.Y('State/UnionTerritory', sort='-x', title='State'),
        x=alt.X('Total Individuals Vaccinated', title='Total Individuals Vaccinated'),
        tooltip=['State/UnionTerritory', 'Total Individuals Vaccinated', 'Date']
    ).properties(
        title='Top 15 States by Total Individuals Vaccinated (as of latest data)'
    ).interactive()
    st.altair_chart(chart_top, use_container_width=True)
    
    # --- Bottom 10 States ---
    st.subheader("Bottom 10 States by Total Individuals Vaccinated")
    
    # Filter out states with 0 vaccinations (likely data issues or territories not started)
    df_latest_filtered = df_latest[df_latest['Total Individuals Vaccinated'] > 0]
    bottom_10_states = df_latest_filtered.nsmallest(10, 'Total Individuals Vaccinated')

    # Create the Interactive Bar Chart for Bottom 10
    chart_bottom = alt.Chart(bottom_10_states).mark_bar(color="tomato").encode(
        # Sort y-axis ascending to show lowest at the top
        y=alt.Y('State/UnionTerritory', sort='x', title='State'), 
        x=alt.X('Total Individuals Vaccinated', title='Total Individuals Vaccinated'),
        tooltip=['State/UnionTerritory', 'Total Individuals Vaccinated', 'Date']
    ).properties(
        title='Bottom 10 States by Total Individuals Vaccinated (as of latest data)'
    ).interactive()
    st.altair_chart(chart_bottom, use_container_width=True)

    with st.expander("Show data for Bottom 10 states"):
        st.dataframe(bottom_10_states[['State/UnionTerritory', 'Date', 'Total Individuals Vaccinated']])
    
    st.divider() # Add a horizontal line

    # --- SECTION 2: T-TEST ANALYSIS ---
    st.header("Analysis: T-Test for High vs. Low Vaccination States")
    st.markdown("""
    This analysis tests if states with **higher vaccination rates** experienced 
    significantly **lower daily deaths** during the peak of the second wave.

    - **Snapshot Date:** We check vaccination levels on **April 30, 2021**.
    - **Outcome Period:** We compare average daily deaths from **May 1 - May 19, 2021**.
    - **Groups:** We compare the 10 states with the *most* vaccinations against the 10 states with the *least*.
    """)
    
    if st.button("Run T-Test Analysis"):
        # ... (T-Test code remains the same as before) ...
        try:
            snapshot_date = '2021-04-30'
            outcome_start = '2021-05-01'
            outcome_end = '2021-05-19'
            snapshot_data = df_merged[df_merged['Date'] == snapshot_date].sort_values('Total Individuals Vaccinated', ascending=False)
            snapshot_data = snapshot_data[snapshot_data['Total Individuals Vaccinated'] > 0]
            high_vax_states = snapshot_data.head(10)['State/UnionTerritory']
            low_vax_states = snapshot_data.tail(10)['State/UnionTerritory']
            outcome_df = df_merged[(df_merged['Date'] >= outcome_start) & (df_merged['Date'] <= outcome_end)]
            high_vax_deaths = outcome_df[outcome_df['State/UnionTerritory'].isin(high_vax_states)]['Daily Deaths']
            low_vax_deaths = outcome_df[outcome_df['State/UnionTerritory'].isin(low_vax_states)]['Daily Deaths']
            t_statistic, p_value = stats.ttest_ind(high_vax_deaths, low_vax_deaths, equal_var=False)
            
            st.subheader("T-Test Results")
            col1, col2 = st.columns(2)
            col1.metric("Avg. Daily Deaths (High Vax States)", f"{high_vax_deaths.mean():.2f}")
            col2.metric("Avg. Daily Deaths (Low Vax States)", f"{low_vax_deaths.mean():.2f}")
            st.metric("P-Value", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("The result is statistically significant (p < 0.05).")
            else:
                st.warning("The result is not statistically significant (p >= 0.05).")
        except Exception as e:
            st.error(f"An error occurred during the T-test: {e}")
            
    st.divider()

    # --- SECTION 3: STATE-WISE CORRELATION (NEW) ---
    st.header("Analysis: State-wise Lagged Correlation")
    st.markdown("""
    This analysis calculates the **21-day lagged correlation** between *Total Individuals Vaccinated* and *Daily Deaths* for each state. 
    
    A **negative** correlation is desired (as vaccinations go up, deaths go down 3 weeks later). 
    A **positive** correlation suggests other factors (like the severity of a wave) had a stronger impact.
    """)
    
    if st.button("Run State-wise Correlation Analysis"):
        with st.spinner("Calculating correlations for all states..."):
            try:
                # 1. Create the analysis dataframe
                # We start from March 2021 to ensure vaccination data is present
                analysis_start_date = '2021-03-01'
                df_corr_analysis = df_merged[df_merged['Date'] >= analysis_start_date].copy()
                
                # 2. Create the 21-day lagged vaccination column for each state
                df_corr_analysis['Vax_Lagged_21d'] = df_corr_analysis.groupby('State/UnionTerritory')['Total Individuals Vaccinated'].shift(21).fillna(0)
                
                # 3. Filter for only the period where lagged data exists
                df_corr_analysis = df_corr_analysis[df_corr_analysis['Vax_Lagged_21d'] > 0]
                
                # 4. Group by state and calculate the correlation
                # This returns a multi-index dataframe
                state_correlations_multi = df_corr_analysis.groupby('State/UnionTerritory')[['Vax_Lagged_21d', 'Daily Deaths']].corr(method='pearson')
                
                # 5. Extract just the correlation value we want
                correlations = state_correlations_multi.unstack().loc[:, ('Vax_Lagged_21d', 'Daily Deaths')]
                correlations = correlations.reset_index(name='Correlation (Vax vs. Deaths)')
                correlations = correlations.sort_values('Correlation (Vax vs. Deaths)', ascending=True).dropna()

                st.subheader("Lagged Correlation Results by State")
                
                # 6. Display Top and Bottom states
                col1, col2 = st.columns(2)
                col1.write("States with Strongest *Negative* Correlation (Desired Effect):")
                col1.dataframe(correlations.head(10))
                
                col2.write("States with Strongest *Positive* Correlation (Spurious Effect):")
                col2.dataframe(correlations.tail(10))

                with st.expander("Show Full Correlation List for All States"):
                    st.dataframe(correlations)
                    
            except Exception as e:
                st.error(f"An error occurred during the correlation analysis: {e}")

else:
    st.warning("Data could not be loaded. Please check your CSV files.")