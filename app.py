"""
Streamlit Dashboard for Juvenile Crime Analysis in Maharashtra (2017-2022)
Interactive visualization and analysis tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Juvenile Crime Analysis - Maharashtra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Juvenile Crime Analysis in Maharashtra (2017-2022)</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load cleaned Maharashtra crime data"""
    try:
        df = pd.read_csv('outputs/data/maharashtra_cleaned.csv')
        return df
    except:
        st.error("Please run main.ipynb first to generate the cleaned data!")
        return None

@st.cache_data
def load_forecast():
    """Load Prophet forecast data"""
    try:
        forecast = pd.read_csv('outputs/data/prophet_forecast.csv')
        return forecast
    except:
        return None

@st.cache_data
def load_model_comparison():
    """Load model comparison data"""
    try:
        comparison = pd.read_csv('outputs/data/comprehensive_model_comparison.csv')
        return comparison
    except:
        try:
            comparison = pd.read_csv('outputs/data/model_comparison.csv')
            return comparison
        except:
            return None

# Load data
df = load_data()
if df is None:
    st.stop()

forecast_df = load_forecast()
comparison_df = load_model_comparison()

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["📈 Overview", "🏘️ District Analysis", "📅 Yearly Trends", "🔬 Model Performance", "🔮 Forecasting"]
)

# Overview Page
if page == "📈 Overview":
    st.header("📈 Overview Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_crimes = df['total_crime'].sum()
    avg_crimes_per_year = df.groupby('year')['total_crime'].sum().mean()
    num_districts = df['district_name'].nunique()
    years_covered = f"{df['year'].min()}-{df['year'].max()}"
    
    with col1:
        st.metric("Total Crimes", f"{total_crimes:,.0f}")
    with col2:
        st.metric("Avg Crimes/Year", f"{avg_crimes_per_year:,.0f}")
    with col3:
        st.metric("Districts", num_districts)
    with col4:
        st.metric("Years", years_covered)
    
    st.markdown("---")
    
    # Top 10 Districts
    st.subheader("🏆 Top 10 Districts by Total Crimes")
    district_totals = df.groupby('district_name')['total_crime'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=district_totals.values,
        y=district_totals.index,
        orientation='h',
        labels={'x': 'Total Crimes', 'y': 'District'},
        color=district_totals.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500, showlegend=False, title="Top 10 Districts by Total Crimes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly Trend
    st.subheader("📅 Yearly Crime Trend")
    yearly_data = df.groupby('year')['total_crime'].sum().reset_index()
    
    fig = px.line(
        yearly_data,
        x='year',
        y='total_crime',
        markers=True,
        labels={'year': 'Year', 'total_crime': 'Total Crimes'},
        title="Total Crimes Over Time"
    )
    fig.update_traces(line_width=3, marker_size=10)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    crime_cols = [col for col in df.columns if col not in ['year', 'state_name', 'state_code', 
                                                             'district_name', 'district_code', 'total_crime']]
    top_crimes = df[crime_cols].sum().nlargest(10).index.tolist()
    corr_data = df[top_crimes + ['total_crime']].corr()
    
    fig = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        x=corr_data.columns,
        y=corr_data.columns,
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig.update_layout(height=600, title="Correlation Matrix of Top Crime Types")
    st.plotly_chart(fig, use_container_width=True)

# District Analysis Page
elif page == "🏘️ District Analysis":
    st.header("🏘️ District Analysis")
    
    # District selector
    districts = sorted(df['district_name'].unique())
    selected_district = st.selectbox("Select District", districts)
    
    # Filter data
    district_data = df[df['district_name'] == selected_district].copy()
    
    if len(district_data) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Crimes", f"{district_data['total_crime'].sum():,.0f}")
        with col2:
            st.metric("Avg Crimes/Year", f"{district_data['total_crime'].mean():,.0f}")
        with col3:
            st.metric("Years of Data", len(district_data))
        
        # Yearly trend for selected district
        st.subheader(f"📈 Yearly Trend: {selected_district}")
        district_yearly = district_data.groupby('year')['total_crime'].sum().reset_index()
        
        fig = px.bar(
            district_yearly,
            x='year',
            y='total_crime',
            labels={'year': 'Year', 'total_crime': 'Total Crimes'},
            title=f"Crime Trend in {selected_district}"
        )
        fig.update_traces(marker_color='steelblue')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top crime types for district
        st.subheader(f"🔍 Top Crime Types: {selected_district}")
        crime_cols = [col for col in district_data.columns if col not in ['year', 'state_name', 'state_code', 
                                                                          'district_name', 'district_code', 'total_crime']]
        district_crimes = district_data[crime_cols].sum().sort_values(ascending=False).head(10)
        
        fig = px.pie(
            values=district_crimes.values,
            names=district_crimes.index,
            title=f"Top 10 Crime Types in {selected_district}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Yearly Trends Page
elif page == "📅 Yearly Trends":
    st.header("📅 Yearly Trends Analysis")
    
    # Year selector
    years = sorted(df['year'].unique())
    selected_year = st.selectbox("Select Year", years)
    
    # Filter data
    year_data = df[df['year'] == selected_year].copy()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Crimes", f"{year_data['total_crime'].sum():,.0f}")
    with col2:
        st.metric("Districts", len(year_data))
    with col3:
        st.metric("Avg Crimes/District", f"{year_data['total_crime'].mean():,.0f}")
    
    # Top districts for selected year
    st.subheader(f"🏆 Top Districts in {selected_year}")
    year_districts = year_data.groupby('district_name')['total_crime'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=year_districts.values,
        y=year_districts.index,
        orientation='h',
        labels={'x': 'Total Crimes', 'y': 'District'},
        title=f"Top 10 Districts in {selected_year}"
    )
    fig.update_traces(marker_color='crimson')
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year comparison
    st.subheader("📊 Year-over-Year Comparison")
    all_years_data = df.groupby('year')['total_crime'].sum().reset_index()
    
    fig = px.line(
        all_years_data,
        x='year',
        y='total_crime',
        markers=True,
        labels={'year': 'Year', 'total_crime': 'Total Crimes'},
        title="Total Crimes Across All Years"
    )
    fig.add_scatter(
        x=all_years_data['year'],
        y=all_years_data['total_crime'],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Data Points'
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Performance Page
elif page == "🔬 Model Performance":
    st.header("🔬 Model Performance Comparison")
    
    if comparison_df is not None:
        st.subheader("📊 Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        st.subheader("📈 Performance Metrics Visualization")
        
        metric = st.selectbox("Select Metric", ["Test R²", "Test MAE", "Test RMSE"])
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=metric,
            color=metric,
            color_continuous_scale='Viridis',
            title=f"{metric} by Model"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        if metric == "Test R²":
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            best_score = comparison_df[metric].max()
            st.success(f"🏆 Best Model: **{best_model}** with {metric} = {best_score:.4f}")
        else:
            best_model = comparison_df.loc[comparison_df[metric].idxmin(), 'Model']
            best_score = comparison_df[metric].min()
            st.success(f"🏆 Best Model: **{best_model}** with {metric} = {best_score:.4f}")
    else:
        st.warning("Model comparison data not available. Please run main.ipynb to generate model results.")

# Forecasting Page
elif page == "🔮 Forecasting":
    st.header("🔮 Crime Forecasting")
    
    if forecast_df is not None:
        st.subheader("📈 Prophet Forecast: Next 3 Years")
        
        # Prepare data for visualization
        historical = df.groupby('year')['total_crime'].sum().reset_index()
        historical['ds'] = pd.to_datetime(historical['year'], format='%Y')
        historical['type'] = 'Historical'
        
        forecast_plot = forecast_df.copy()
        forecast_plot['ds'] = pd.to_datetime(forecast_plot['ds'])
        forecast_plot['type'] = 'Forecast'
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['total_crime'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        # Forecast
        future_forecast = forecast_plot[forecast_plot['ds'] > historical['ds'].max()]
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title="Prophet Forecast: Total Crimes in Maharashtra (Next 3 Years)",
            xaxis_title="Year",
            yaxis_title="Total Crimes",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.subheader("📋 Forecast Details")
        forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_display.columns = ['Year', 'Forecast', 'Lower Bound', 'Upper Bound']
        forecast_display['Year'] = forecast_display['Year'].dt.year
        forecast_display = forecast_display.round(2)
        st.dataframe(forecast_display, use_container_width=True)
        
        # Insights
        st.subheader("💡 Forecast Insights")
        latest_forecast = future_forecast.iloc[-1]
        latest_historical = historical.iloc[-1]
        
        change = ((latest_forecast['yhat'] - latest_historical['total_crime']) / latest_historical['total_crime']) * 100
        
        if change > 0:
            st.info(f"📈 Forecast indicates a **{change:.1f}% increase** in total crimes by {int(latest_forecast['ds'].year)} compared to {int(latest_historical['ds'].year)}")
        else:
            st.info(f"📉 Forecast indicates a **{abs(change):.1f}% decrease** in total crimes by {int(latest_forecast['ds'].year)} compared to {int(latest_historical['ds'].year)}")
    else:
        st.warning("Forecast data not available. Please run main.ipynb to generate forecasts.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Juvenile Crime Analysis Dashboard | Data Source: India Data Portal</p>
        <p>Maharashtra State (2017-2022)</p>
    </div>
""", unsafe_allow_html=True)

