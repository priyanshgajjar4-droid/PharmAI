import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="PharmAI Global", layout="wide")
st.title("🌍 PharmAI Global Drug Surveillance")

@st.cache_data
def load_data():
    if not os.path.exists('global_safety_summary.parquet'): return None
    return pd.read_parquet('global_safety_summary.parquet')

df = load_data()

if df is None:
    st.error("Upload 'global_safety_summary.parquet'")
    st.stop()

# Sidebar
all_drugs = sorted(df['drugname'].unique())
selected_drug = st.sidebar.selectbox("Select Drug:", ["Search..."] + all_drugs)

if selected_drug != "Search...":
    data = df[df['drugname'] == selected_drug]
    
    # Metrics
    total = data['count'].sum()
    top = data.loc[data['count'].idxmax()]['pt']
    
    c1, c2 = st.columns(2)
    c1.metric("Total Reports", f"{total:,}")
    c2.metric("Top Signal", top)
    
    # Charts
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("Top Reactions")
        # Sum across years
        top_reacs = data.groupby('pt')['count'].sum().nlargest(10).reset_index()
        fig = px.bar(top_reacs, x='count', y='pt', orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c_right:
        st.subheader("Yearly Trend")
        # Ensure year is treated as category/string for line chart
        yearly = data.groupby('year')['count'].sum().reset_index()
        fig2 = px.line(yearly, x='year', y='count', markers=True)
        fig2.update_xaxes(type='category') # Force year labels (2022, 2023)
        st.plotly_chart(fig2, use_container_width=True)