import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ALS-EMG Biomarker Integration Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load ALS datasets"""
    try:
        longitudinal = pd.read_csv('als_longitudinal_data.csv')
        demographics = pd.read_csv('als_patient_demographics.csv')
        
        # Merge datasets
        df = longitudinal.merge(demographics, on='patient_id', how='left')
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        
        return df
    except FileNotFoundError:
        st.error("Dataset files not found. Please run the data generation script first.")
        return None

def calculate_biomarker_correlations(df):
    """Calculate correlations between EMG parameters and ALSFRS-R"""
    biomarker_cols = [
        'mune_count', 'cmap_amplitude_mv', 'fasciculation_frequency_hz',
        'neurophysiology_index', 'neurofilament_light_pg_ml', 
        'creatinine_mg_dl', 'uric_acid_mg_dl', 'fvc_percent_predicted'
    ]
    
    correlations = {}
    p_values = {}
    
    for biomarker in biomarker_cols:
        if biomarker in df.columns:
            corr, p_val = stats.pearsonr(df[biomarker].dropna(), 
                                       df.loc[df[biomarker].notna(), 'alsfrs_r_total'])
            correlations[biomarker] = corr
            p_values[biomarker] = p_val
    
    return correlations, p_values

def create_patient_timeline(df, patient_id):
    """Create timeline visualization for a specific patient"""
    patient_data = df[df['patient_id'] == patient_id].sort_values('visit_month')
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('ALSFRS-R Progression', 'EMG Parameters', 'Biomarkers'),
        vertical_spacing=0.1
    )
    
    # ALSFRS-R progression
    fig.add_trace(
        go.Scatter(x=patient_data['visit_month'], y=patient_data['alsfrs_r_total'],
                  mode='lines+markers', name='ALSFRS-R Total',
                  line=dict(color='red', width=3)),
        row=1, col=1
    )
    
    # EMG parameters
    fig.add_trace(
        go.Scatter(x=patient_data['visit_month'], y=patient_data['mune_count'],
                  mode='lines+markers', name='MUNE Count',
                  line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=patient_data['visit_month'], y=patient_data['cmap_amplitude_mv'],
                  mode='lines+markers', name='CMAP Amplitude',
                  line=dict(color='green'), yaxis='y3'),
        row=2, col=1
    )
    
    # Biomarkers
    fig.add_trace(
        go.Scatter(x=patient_data['visit_month'], y=patient_data['neurofilament_light_pg_ml'],
                  mode='lines+markers', name='Neurofilament Light',
                  line=dict(color='purple')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, title=f'Patient {patient_id} Timeline')
    fig.update_xaxes(title_text="Months from Study Start")
    
    return fig

# Main Application
def main():
    st.markdown('<h1 class="main-header">ALS-EMG Biomarker Integration Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **For Dr. Ghazala Hayat - SLUCare ALS Clinic**
    
    Advanced biomarker correlation analysis platform integrating EMG neurophysiology with functional outcomes for ALS research.
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar controls
    st.sidebar.markdown('<h2 class="sub-header">Analysis Controls</h2>', unsafe_allow_html=True)
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Population Overview", "Biomarker Correlations", "Patient Timeline", 
         "Progression Modeling", "Research Insights"]
    )
    
    if analysis_type == "Population Overview":
        st.markdown('<h2 class="sub-header">Population Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df['patient_id'].unique()))
        with col2:
            st.metric("Study Duration", f"{df['visit_month'].max()} months")
        with col3:
            avg_decline = df.groupby('patient_id')['alsfrs_r_total'].apply(
                lambda x: (x.iloc[0] - x.iloc[-1]) / len(x) if len(x) > 1 else 0
            ).mean()
            st.metric("Avg ALSFRS-R Decline", f"{avg_decline:.2f} pts/month")
        with col4:
            completion_rate = (df.groupby('patient_id')['visit_month'].max() >= 18).mean() * 100
            st.metric("18+ Month Follow-up", f"{completion_rate:.1f}%")
        
        # Demographics
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age = px.histogram(df[df['visit_month']==0], x='age_at_onset', 
                                 title='Age at Onset Distribution')
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            onset_counts = df[df['visit_month']==0]['onset_type'].value_counts()
            fig_onset = px.pie(values=onset_counts.values, names=onset_counts.index,
                             title='Onset Type Distribution')
            st.plotly_chart(fig_onset, use_container_width=True)
    
    elif analysis_type == "Biomarker Correlations":
        st.markdown('<h2 class="sub-header">Biomarker-ALSFRS-R Correlations</h2>', unsafe_allow_html=True)
        
        correlations, p_values = calculate_biomarker_correlations(df)
        
        # Create correlation dataframe
        corr_df = pd.DataFrame({
            'Biomarker': list(correlations.keys()),
            'Correlation': list(correlations.values()),
            'P-Value': list(p_values.values()),
            'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                           for p in p_values.values()]
        })
        
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Display correlation table
        st.markdown("**Statistical Correlations with ALSFRS-R Total Score:**")
        st.dataframe(corr_df.style.format({'Correlation': '{:.3f}', 'P-Value': '{:.3e}'}))
        
        # Visualization
        fig_corr = px.bar(corr_df, x='Biomarker', y='Correlation', 
                         color='Significance',
                         title='Biomarker Correlations with ALSFRS-R',
                         color_discrete_map={'***': 'red', '**': 'orange', '*': 'yellow', 'ns': 'gray'})
        fig_corr.update_xaxes(tickangle=45)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots
        biomarker_select = st.selectbox("Select biomarker for detailed analysis:", 
                                       list(correlations.keys()))
        
        fig_scatter = px.scatter(df, x=biomarker_select, y='alsfrs_r_total',
                               color='onset_type', trendline='ols',
                               title=f'{biomarker_select} vs ALSFRS-R')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif analysis_type == "Patient Timeline":
        st.markdown('<h2 class="sub-header">Individual Patient Analysis</h2>', unsafe_allow_html=True)
        
        patient_id = st.selectbox("Select Patient ID:", 
                                 sorted(df['patient_id'].unique()))
        
        # Patient info
        patient_info = df[df['patient_id'] == patient_id].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Age at Onset:** {patient_info['age_at_onset']}")
        with col2:
            st.info(f"**Onset Type:** {patient_info['onset_type']}")
        with col3:
            st.info(f"**Sex:** {patient_info['sex']}")
        with col4:
            st.info(f"**BMI:** {patient_info['bmi']:.1f}")
        
        # Timeline visualization
        fig_timeline = create_patient_timeline(df, patient_id)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Patient data table
        patient_data = df[df['patient_id'] == patient_id][
            ['visit_month', 'alsfrs_r_total', 'mune_count', 'cmap_amplitude_mv', 
             'neurofilament_light_pg_ml', 'fvc_percent_predicted']
        ].sort_values('visit_month')
        
        st.markdown("**Patient Data Table:**")
        st.dataframe(patient_data)
    
    elif analysis_type == "Progression Modeling":
        st.markdown('<h2 class="sub-header">Disease Progression Analysis</h2>', unsafe_allow_html=True)
        
        # Progression rates by onset type
        progression_rates = []
        for pid in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == pid].sort_values('visit_month')
            if len(patient_data) > 1:
                rate = (patient_data['alsfrs_r_total'].iloc[0] - patient_data['alsfrs_r_total'].iloc[-1]) / \
                       (patient_data['visit_month'].iloc[-1] - patient_data['visit_month'].iloc[0])
                progression_rates.append({
                    'patient_id': pid,
                    'progression_rate': rate,
                    'onset_type': patient_data['onset_type'].iloc[0],
                    'baseline_alsfrs': patient_data['alsfrs_r_total'].iloc[0]
                })
        
        prog_df = pd.DataFrame(progression_rates)
        
        # Box plot of progression rates
        fig_prog = px.box(prog_df, x='onset_type', y='progression_rate',
                         title='ALSFRS-R Progression Rates by Onset Type')
        fig_prog.update_yaxes(title='ALSFRS-R Points per Month')
        st.plotly_chart(fig_prog, use_container_width=True)
        
        # Survival analysis
        st.markdown("**Milestone Analysis:**")
        
        milestones = {
            'ALSFRS-R ≤ 30': df[df['alsfrs_r_total'] <= 30]['patient_id'].nunique(),
            'ALSFRS-R ≤ 20': df[df['alsfrs_r_total'] <= 20]['patient_id'].nunique(),
            'Feeding Tube': df[df['feeding_tube'] == True]['patient_id'].nunique(),
            'Ventilation': df[df['ventilation_support'] == True]['patient_id'].nunique()
        }
        
        milestone_df = pd.DataFrame(list(milestones.items()), columns=['Milestone', 'Patients'])
        fig_milestones = px.bar(milestone_df, x='Milestone', y='Patients',
                               title='Disease Milestone Frequencies')
        st.plotly_chart(fig_milestones, use_container_width=True)
    
    elif analysis_type == "Research Insights":
        st.markdown('<h2 class="sub-header">Research Insights & Export</h2>', unsafe_allow_html=True)
        
        # Key findings
        st.markdown("### Key Research Findings:")
        
        correlations, p_values = calculate_biomarker_correlations(df)
        
        # Strongest correlations
        strong_corrs = [(k, v) for k, v in correlations.items() if abs(v) > 0.5 and p_values[k] < 0.01]
        
        for biomarker, corr in sorted(strong_corrs, key=lambda x: abs(x[1]), reverse=True):
            direction = "positively" if corr > 0 else "negatively"
            st.write(f"• **{biomarker}** is {direction} correlated with ALSFRS-R (r={corr:.3f}, p<0.01)")
        
        # Export options
        st.markdown("### Data Export Options:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Correlation Analysis"):
                corr_data = pd.DataFrame({
                    'Biomarker': list(correlations.keys()),
                    'Correlation': list(correlations.values()),
                    'P_Value': list(p_values.values())
                })
                csv = corr_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "als_correlations.csv", "text/csv")
        
        with col2:
            if st.button("Export Patient Summary"):
                summary_data = df.groupby('patient_id').agg({
                    'alsfrs_r_total': ['first', 'last', 'min'],
                    'visit_month': 'max',
                    'onset_type': 'first',
                    'age_at_onset': 'first'
                }).round(2)
                csv = summary_data.to_csv()
                st.download_button("Download CSV", csv, "patient_summary.csv", "text/csv")
        
        with col3:
            if st.button("Export Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "full_als_dataset.csv", "text/csv")
        
        # Statistical summary
        st.markdown("### Statistical Summary:")
        st.write(df.describe())

if __name__ == "__main__":
    main()
