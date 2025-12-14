import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. ENTERPRISE CONFIGURATION ---
st.set_page_config(
    page_title="PharmAI Pro | Safety Intelligence", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Look
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val {font-size: 28px; font-weight: 700; color: #1e40af;}
    .metric-lbl {font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 600;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; color: #0f172a;}
    .stSelectbox label {font-weight: bold; color: #334155;}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADER ---
@st.cache_data
def load_data():
    if not os.path.exists('global_safety_summary.parquet'):
        return None
    return pd.read_parquet('global_safety_summary.parquet')

df = load_data()

# --- 3. SIDEBAR (PROFESSIONAL TOOLS) ---
with st.sidebar:
    st.title("🧬 PharmAI Pro")
    st.caption("Global Pharmacovigilance Suite")
    st.markdown("---")
    
    if df is None:
        st.error("🚨 Database Offline. Please upload 'global_safety_summary.parquet'.")
        st.stop()

    # Search Tool
    all_drugs = sorted(df['drugname'].unique())
    selected_drug = st.selectbox("Select Therapeutic Agent:", ["Type to search..."] + all_drugs)
    
    # Filter Tool (Time Range)
    years = sorted(df['year'].unique())
    selected_years = st.select_slider("Analysis Period:", options=years, value=(min(years), max(years)))
    
    st.markdown("---")
    st.markdown("### 🛠️ Analyst Tools")
    show_raw = st.checkbox("Show Raw Source Data")
    high_contrast = st.checkbox("High Contrast Mode (Vis)")

# --- 4. MAIN ANALYTICS ENGINE ---
if selected_drug and selected_drug != "Type to search...":
    
    # Filter Data based on User Selection
    mask = (df['drugname'] == selected_drug) & (df['year'].between(selected_years[0], selected_years[1]))
    data = df[mask]
    
    # Header
    c1, c2 = st.columns([3, 1])
    c1.title(f"Safety Profile: {selected_drug}")
    c1.markdown(f"**Data Source:** FDA FAERS | **Period:** {selected_years[0]} - {selected_years[1]}")
    
    # Export Button
    csv = data.to_csv(index=False).encode('utf-8')
    c2.download_button(
        label="📥 Download Report",
        data=csv,
        file_name=f"{selected_drug}_safety_report.csv",
        mime="text/csv"
    )

    if data.empty:
        st.warning("No reports found for this time period.")
        st.stop()

    st.markdown("---")

    # KPIs (Key Performance Indicators)
    total_events = data['count'].sum()
    top_event_row = data.loc[data['count'].idxmax()]
    unique_pt = data['pt'].nunique()
    
    # Yearly Growth Calculation
    yearly_counts = data.groupby('year')['count'].sum()
    if len(yearly_counts) > 1:
        growth = ((yearly_counts.iloc[-1] - yearly_counts.iloc[0]) / yearly_counts.iloc[0]) * 100
        growth_str = f"{growth:+.1f}%"
        growth_color = "red" if growth > 0 else "green"
    else:
        growth_str = "N/A"
        growth_color = "black"

    # KPI Display
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(f'<div class="metric-card"><div class="metric-lbl">Total ICSRs</div><div class="metric-val">{total_events:,}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-card"><div class="metric-lbl">Unique Signals</div><div class="metric-val">{unique_pt:,}</div></div>', unsafe_allow_html=True)
    kpi3.markdown(f'<div class="metric-card"><div class="metric-lbl">Top Adverse Event</div><div class="metric-val" style="font-size:20px">{top_event_row["pt"]}</div></div>', unsafe_allow_html=True)
    kpi4.markdown(f'<div class="metric-card"><div class="metric-lbl">Volume Trend</div><div class="metric-val" style="color:{growth_color}">{growth_str}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Visual Intelligence")

    # Tabbed Interface for Cleaner Look
    tab1, tab2 = st.tabs(["Signal Detection", "Temporal Analysis"])

    with tab1:
        # Interactive Bar Chart (Top 10)
        top_10 = data.groupby('pt')['count'].sum().nlargest(10).reset_index()
        fig_bar = px.bar(
            top_10, x='count', y='pt', orientation='h',
            title=f"Top 10 Most Frequent Adverse Events for {selected_drug}",
            labels={'count': 'Report Count', 'pt': 'MedDRA Preferred Term'},
            text='count',
            color='count',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        # Interactive Line Chart
        trend_data = data.groupby('year')['count'].sum().reset_index()
        fig_line = px.line(
            trend_data, x='year', y='count', markers=True,
            title="Reporting Volume Over Time",
            labels={'count': 'Total Reports', 'year': 'Year'},
            line_shape='spline'
        )
        fig_line.update_traces(line_color='#2563eb', line_width=4)
        fig_line.update_xaxes(type='category') # Ensure years don't show as decimals
        st.plotly_chart(fig_line, use_container_width=True)

    # Raw Data Expander
    if show_raw:
        st.markdown("### 📂 Source Data Inspector")
        st.dataframe(data.sort_values(by='count', ascending=False), use_container_width=True)

else:
    # Landing Page State
    st.info("👈 Please select a therapeutic agent from the sidebar to initialize the Intelligence Engine.")
    st.markdown("""
    ### System Capabilities:
    * **Universal Search:** Access data for 200,000+ drug names.
    * **Live Analytics:** Real-time aggregation of FDA reports.
    * **Trend Detection:** Identify spikes in reporting volume.
    * **Export:** Download audit-ready CSV reports.
    """)