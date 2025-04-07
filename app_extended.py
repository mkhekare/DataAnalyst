import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import base64
import io
import tempfile
from datetime import datetime
import time
import os
import json
import re
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import google.generativeai as genai
from io import BytesIO

# Configure the page
st.set_page_config(
    layout="wide",
    page_title="🔍 DataInsight Pro",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a80d2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .reportgen {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .reportgen:hover {
        background-color: #218838;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .insight-card {
        background-color: #f1f8ff;
        border-left: 4px solid #4a90e2;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .stat-box {
        background-color: white;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4a90e2;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #4a90e2 !important;
        border-bottom: 2px solid #4a90e2;
    }
    footer {
        margin-top: 50px;
        padding: 20px 0;
        border-top: 1px solid #e9ecef;
        text-align: center;
        font-size: 14px;
        color: #6c757d;
    }
    .file-uploader {
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.3s;
    }
    .file-uploader:hover {
        border-color: #4a90e2;
        background-color: #f1f8ff;
    }
    .sql-card {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'sql_queries' not in st.session_state:
    st.session_state.sql_queries = []
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = None
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'bar'
if 'univariate_figs' not in st.session_state:
    st.session_state.univariate_figs = {}
if 'bivariate_figs' not in st.session_state:
    st.session_state.bivariate_figs = {}
if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = True  # Default to enabled since API is backend

# ========== GEMINI AI SETUP ==========
api_key = os.getenv("GEMINI_API_KEY")

genai_available = True
st.session_state.ai_enabled = True

try:
    genai.configure(api_key=api_key)  # <-- This line makes the key-based setup work
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    genai_available = False
    st.sidebar.error(f"⚠️ AI features unavailable: {str(e)}")

# Function to generate downloadable link for any file
def get_download_link(file_path, link_text, file_format="csv"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = "csv" if file_format == "csv" else "xlsx" if file_format == "excel" else "pdf"
    filename = f"datainisght_export_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
    href = f'<a href="data:application/{file_format};base64,{b64}" download="{filename}" class="downloadBtn">{link_text}</a>'
    return href

# Function to save plotly figure as image
def fig_to_bytes(fig, format="png"):
    """Convert a plotly figure to a bytes object"""
    img_bytes = fig.to_image(format=format, scale=2)
    return img_bytes

# Function to create enhanced PDF report
def generate_enhanced_pdf_report(df, report_title, include_charts=True, include_summary=True, 
                                include_data=True, include_insights=True, include_sql=True):
    try:
        # Create a PDF document
        class PDF(FPDF):
            def header(self):
                # Logo or title
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, report_title, 0, 1, 'C')
                # Line break
                self.ln(5)
                
            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                # Arial italic 8
                self.set_font('Arial', 'I', 8)
                # Page number
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
                # Time stamp
                self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'R')
                
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Cover page
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, report_title, ln=True, align='C')
        pdf.ln(10)
        
        # Add timestamp
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(30)
        
        # Add small description
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, "This report contains a comprehensive analysis of the dataset including statistical summaries, data visualizations, and AI-powered insights.")
        pdf.ln(20)
        
        # Add dataset information
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Dataset Overview", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Dataset Name: {st.session_state.filename if st.session_state.filename else 'Unnamed Dataset'}", ln=True)
        pdf.cell(0, 10, f"Rows: {len(df)}", ln=True)
        pdf.cell(0, 10, f"Columns: {len(df.columns)}", ln=True)
        pdf.cell(0, 10, f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB", ln=True)
        
        # Table of contents
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Table of Contents", ln=True)
        pdf.ln(5)
        
        toc_items = []
        page_num = 3  # Start from page 3 (after cover and TOC)
        
        if include_summary:
            toc_items.append(("1. Data Summary", page_num))
            page_num += 3  # Estimate pages for this section
            
        if include_data:
            toc_items.append(("2. Data Sample", page_num))
            page_num += 1
            
        if include_charts:
            section_num = len(toc_items) + 1
            toc_items.append((f"{section_num}. Univariate Analysis", page_num))
            page_num += 3
            
            toc_items.append((f"{section_num+1}. Bivariate Analysis", page_num))
            page_num += 3
            
            toc_items.append((f"{section_num+2}. Correlation Analysis", page_num))
            page_num += 2
            
        if include_insights and st.session_state.insights:
            section_num = len(toc_items) + 1
            toc_items.append((f"{section_num}. AI Insights", page_num))
            page_num += 1
            
        if include_sql and st.session_state.sql_queries:
            section_num = len(toc_items) + 1
            toc_items.append((f"{section_num}. SQL Queries", page_num))
            page_num += 1
        
        # Print TOC
        pdf.set_font('Arial', '', 12)
        for title, page in toc_items:
            pdf.cell(0, 10, f"{title} {'.' * (70 - len(title))} {page}", ln=True)
        
        # Add data summary if selected
        if include_summary:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, "1. Data Summary", ln=True)
            pdf.ln(5)
            
            # Calculate numerical summary
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Statistics for numerical columns
            if not numeric_cols.empty:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, "1.1 Numeric Columns Summary", ln=True)
                pdf.ln(5)
                
                summary = df[numeric_cols].describe().transpose()
                
                # Create a summary table
                pdf.set_font('Arial', 'B', 10)
                # Header
                col_width = 30
                stats_width = 25
                pdf.cell(col_width, 10, "Column", 1, 0, 'C')
                for stat in ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
                    pdf.cell(stats_width, 10, stat, 1, 0, 'C')
                pdf.ln()
                
                # Data rows
                pdf.set_font('Arial', '', 10)
                for col in numeric_cols[:15]:  # Limit to 15 columns for readability
                    pdf.cell(col_width, 10, col[:20], 1, 0)
                    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                        val = summary.loc[col, stat]
                        pdf.cell(stats_width, 10, f"{val:.2f}", 1, 0, 'R')
                    pdf.ln()
                
                if len(numeric_cols) > 15:
                    pdf.cell(0, 10, f"... {len(numeric_cols) - 15} more columns not shown", ln=True)
                
                pdf.ln(10)
            
            # Information about categorical columns
            if not categorical_cols.empty:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, "1.2 Categorical Columns Summary", ln=True)
                pdf.ln(5)
                
                pdf.set_font('Arial', 'B', 10)
                # Header
                pdf.cell(60, 10, "Column", 1, 0, 'C')
                pdf.cell(30, 10, "Unique Values", 1, 0, 'C')
                pdf.cell(30, 10, "Top Value", 1, 0, 'C')
                pdf.cell(30, 10, "Frequency", 1, 0, 'C')
                pdf.ln()
                
                # Data rows
                pdf.set_font('Arial', '', 10)
                for col in categorical_cols[:15]:  # Limit to 15 columns
                    value_counts = df[col].value_counts()
                    top_value = value_counts.index[0] if not value_counts.empty else "N/A"
                    top_freq = value_counts.iloc[0] if not value_counts.empty else 0
                    
                    pdf.cell(60, 10, col[:25], 1, 0)
                    pdf.cell(30, 10, f"{df[col].nunique()}", 1, 0, 'C')
                    pdf.cell(30, 10, f"{str(top_value)[:15]}", 1, 0, 'C')
                    pdf.cell(30, 10, f"{top_freq}", 1, 0, 'C')
                    pdf.ln()
                
                if len(categorical_cols) > 15:
                    pdf.cell(0, 10, f"... {len(categorical_cols) - 15} more columns not shown", ln=True)
                    
                pdf.ln(10)
            
            # Missing values information
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "1.3 Missing Values Analysis", ln=True)
            pdf.ln(5)
            
            missing_data = df.isna().sum().sort_values(ascending=False)
            missing_percent = (missing_data / len(df) * 100).round(2)
            missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if not missing_df.empty:
                pdf.set_font('Arial', 'B', 10)
                # Header
                pdf.cell(60, 10, "Column", 1, 0, 'C')
                pdf.cell(40, 10, "Missing Values", 1, 0, 'C')
                pdf.cell(40, 10, "Missing (%)", 1, 0, 'C')
                pdf.ln()
                
                # Data rows
                pdf.set_font('Arial', '', 10)
                for col, row in missing_df.iterrows():
                    pdf.cell(60, 10, col[:25], 1, 0)
                    pdf.cell(40, 10, f"{row['Missing Values']}", 1, 0, 'C')
                    pdf.cell(40, 10, f"{row['Percentage']}%", 1, 0, 'C')
                    pdf.ln()
                
                if len(missing_df) > 15:
                    pdf.cell(0, 10, f"... {len(missing_df) - 15} more columns not shown", ln=True)
            else:
                pdf.cell(0, 10, "No missing values found in the dataset.", ln=True)
            
            pdf.ln(10)
            
            # Duplicate rows information
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "1.4 Duplicate Rows Analysis", ln=True)
            pdf.ln(5)
            
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                pdf.cell(0, 10, f"Number of duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}% of total)", ln=True)
            else:
                pdf.cell(0, 10, "No duplicate rows found in the dataset.", ln=True)
        
        # Add data sample if selected
        if include_data:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            section_num = 2 if include_summary else 1
            pdf.cell(0, 10, f"{section_num}. Data Sample", ln=True)
            pdf.ln(5)
            
            # Show first 10 rows
            sample_data = df.head(10)
            cols = sample_data.columns.tolist()
            
            # Add column headers
            pdf.set_font('Arial', 'B', 8)
            col_width = min(25, 180 / min(len(cols), 8))  # Limit to first 8 columns if more
            
            # First set of columns (up to 8)
            for col in cols[:8]:
                pdf.cell(col_width, 10, str(col)[:15], border=1)
            pdf.ln()
            
            # Add data rows
            pdf.set_font('Arial', '', 8)
            for i, row in sample_data.iterrows():
                for col in cols[:8]:
                    value = str(row[col])
                    if len(value) > 15:
                        value = value[:12] + "..."
                    pdf.cell(col_width, 10, value, border=1)
                pdf.ln()
            
            # If more columns exist, show on next page
            if len(cols) > 8:
                pdf.cell(0, 10, f"... {len(cols) - 8} more columns not shown", ln=True)
        
        # Add charts and visualizations if selected
        if include_charts:
            section_base = (2 if include_summary else 1) + (1 if include_data else 0)
            
            # Univariate Analysis
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{section_base}. Univariate Analysis", ln=True)
            pdf.ln(5)
            
            # Add univariate analysis for numerical columns
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f"{section_base}.1 Numerical Distributions", ln=True)
            pdf.ln(5)
            
            # Check if univariate figures are in the session state
            if 'univariate_figs' in st.session_state and st.session_state.univariate_figs:
                # We'll try to add up to 4 charts per page
                chart_count = 0
                charts_per_page = 3
                
                for col_name, fig in st.session_state.univariate_figs.items():
                    if chart_count % charts_per_page == 0 and chart_count > 0:
                        pdf.add_page()
                    
                    try:
                        # Save figure to temporary file
                        img_bytes = fig_to_bytes(fig)
                        img_path = f"temp_univariate_{col_name}.png"
                        with open(img_path, 'wb') as f:
                            f.write(img_bytes)
                        
                        # Add to PDF
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, f"Distribution of {col_name}", ln=True)
                        pdf.image(img_path, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(60)  # Space for the image
                        
                        # Remove temp file
                        os.remove(img_path)
                        chart_count += 1
                    except Exception as e:
                        pdf.cell(0, 10, f"Error adding chart for {col_name}: {e}", ln=True)
            else:
                # Generate some basic univariate plots if none exist
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for i, col in enumerate(numeric_cols[:6]):  # Limit to first 6 columns
                    if i % 3 == 0 and i > 0:
                        pdf.add_page()
                    
                    try:
                        # Create histograms using matplotlib
                        plt.figure(figsize=(10, 4))
                        sns.histplot(df[col].dropna(), kde=True)
                        plt.title(f"Distribution of {col}")
                        plt.tight_layout()
                        
                        # Save to temp file
                        img_path = f"temp_hist_{col}.png"
                        plt.savefig(img_path)
                        plt.close()
                        
                        # Add to PDF
                        pdf.cell(0, 10, f"Distribution of {col}", ln=True)
                        pdf.image(img_path, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(60)  # Space for the image
                        
                        # Remove temp file
                        os.remove(img_path)
                    except Exception as e:
                        pdf.cell(0, 10, f"Error creating histogram for {col}: {e}", ln=True)
            
            # Add bivariate analysis
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{section_base+1}. Bivariate Analysis", ln=True)
            pdf.ln(5)
            
            # Check if bivariate figures are in the session state
            if 'bivariate_figs' in st.session_state and st.session_state.bivariate_figs:
                # We'll try to add up to 2 charts per page
                chart_count = 0
                charts_per_page = 2
                
                for chart_name, fig in st.session_state.bivariate_figs.items():
                    if chart_count % charts_per_page == 0 and chart_count > 0:
                        pdf.add_page()
                    
                    try:
                        # Save figure to temporary file
                        img_bytes = fig_to_bytes(fig)
                        img_path = f"temp_bivariate_{chart_count}.png"
                        with open(img_path, 'wb') as f:
                            f.write(img_bytes)
                        
                        # Add to PDF
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, chart_name, ln=True)
                        pdf.image(img_path, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(90)  # Space for the image
                        
                        # Remove temp file
                        os.remove(img_path)
                        chart_count += 1
                    except Exception as e:
                        pdf.cell(0, 10, f"Error adding chart: {e}", ln=True)
            else:
                # Generate some basic scatter plots if none exist
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) >= 2:
                    for i in range(min(3, len(numeric_cols) - 1)):  # Up to 3 pairs
                        try:
                            # Create scatter plot using matplotlib
                            plt.figure(figsize=(10, 6))
                            plt.scatter(df[numeric_cols[i]], df[numeric_cols[i+1]], alpha=0.5)
                            plt.title(f"{numeric_cols[i]} vs {numeric_cols[i+1]}")
                            plt.xlabel(numeric_cols[i])
                            plt.ylabel(numeric_cols[i+1])
                            plt.tight_layout()
                            
                            # Save to temp file
                            img_path = f"temp_scatter_{i}.png"
                            plt.savefig(img_path)
                            plt.close()
                            
                            # Add to PDF
                            if i > 0:
                                pdf.add_page()
                            pdf.cell(0, 10, f"{numeric_cols[i]} vs {numeric_cols[i+1]}", ln=True)
                            pdf.image(img_path, x=10, y=pdf.get_y(), w=180)
                            pdf.ln(100)  # Space for the image
                            
                            # Remove temp file
                            os.remove(img_path)
                        except Exception as e:
                            pdf.cell(0, 10, f"Error creating scatter plot: {e}", ln=True)
                            
            # Add correlation matrix
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{section_base+2}. Correlation Analysis", ln=True)
            pdf.ln(5)
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                try:
                    # Create correlation matrix using matplotlib
                    plt.figure(figsize=(10, 8))
                    corr = df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                                square=True, linewidths=.5)
                    plt.title("Correlation Matrix")
                    plt.tight_layout()
                    
                    # Save to temp file
                    corr_path = "temp_correlation.png"
                    plt.savefig(corr_path)
                    plt.close()
                    
                    # Add to PDF
                    pdf.image(corr_path, x=10, y=pdf.get_y(), w=180)
                    
                    # Remove temp file
                    os.remove(corr_path)
                except Exception as e:
                    pdf.cell(0, 10, f"Error creating correlation matrix: {e}", ln=True)
            else:
                pdf.cell(0, 10, "Not enough numerical columns for correlation analysis.", ln=True)
        
        # Add insights if available and selected
        if include_insights and st.session_state.insights:
            section_num = section_base + 3 if include_charts else section_base
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{section_num}. AI Insights", ln=True)
            pdf.ln(10)
            
            pdf.set_font('Arial', '', 12)
            for idx, insight in enumerate(st.session_state.insights):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f"Insight {idx+1}:", ln=True)
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 10, insight)
                pdf.ln(5)
        
        # Add SQL queries if available and selected
        if include_sql and st.session_state.sql_queries:
            section_num = section_base + 4 if include_charts else section_base + 1
            if include_insights and st.session_state.insights:
                section_num += 1
                
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"{section_num}. SQL Queries", ln=True)
            pdf.ln(10)
            
            pdf.set_font('Arial', '', 12)
            for idx, query in enumerate(st.session_state.sql_queries):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f"Query {idx+1}:", ln=True)
                pdf.set_font('Courier', '', 10)
                pdf.multi_cell(0, 10, query)
                pdf.ln(5)
        
        # Save the PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_path = tmp_file.name
            pdf.output(pdf_path)
        
        return pdf_path
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

# Function to generate SQL queries for the dataset
def generate_sql_queries(df):
    if not genai_available:
        # Return sample SQL queries if AI is not enabled
        return [
            "SELECT * FROM data LIMIT 10;",
            "SELECT COUNT(*) FROM data;",
            "SELECT category, COUNT(*) FROM data GROUP BY category;",
            "SELECT * FROM data WHERE revenue > 10000;",
            "SELECT product, MAX(profit) FROM data GROUP BY product ORDER BY MAX(profit) DESC LIMIT 5;"
        ]
    
    try:
        # Generate table schema from dataframe
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = "NUMBER"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = "DATE"
            else:
                column_types[col] = "TEXT"
        
        schema = ", ".join([f"{col} {dtype}" for col, dtype in column_types.items()])
        
        prompt = f"""
        Create 5 useful SQL queries for exploring and analyzing this dataset.
        
        Table name: data
        Schema: {schema}
        
        Include basic queries and more complex analytical queries.
        Format: Only provide the SQL queries, one per line, with no explanations.
        """
        
        response = model.generate_content(prompt)
        queries_text = response.text
        
        # Extract SQL queries
        queries = [q.strip() for q in queries_text.split(";") if q.strip()]
        queries = [f"{q};" for q in queries if q]
        
        return queries[:5]  # Return at most 5 queries
        
    except Exception as e:
        st.error(f"Error generating SQL queries: {e}")
        return ["-- Could not generate SQL queries. Please check your API key or try again later."]

# Function to generate AI insights using Google Gemini
def generate_ai_insights(df):
    if not genai_available:
        # Return sample insights if AI is not enabled
        return [
            "Electronics category has the highest average profit margin at 36%.",
            "Sales show a seasonal pattern with peaks in January and July.",
            "Products in the Food category have the highest sales volume but lowest profit margin.",
            "There is a strong positive correlation between marketing spend and sales volume.",
            "Customer retention rate is higher for electronics compared to other categories."
        ]
    
    try:
        # Prepare data summary for the prompt
        summary = df.describe().to_string()
        head = df.head(5).to_string()
        dtypes = df.dtypes.to_string()
        missing = df.isna().sum().to_string()

        # Create a detailed prompt
        prompt = f"""
        I need in-depth analytical insights for this dataset. Here are some metadata and a preview:

        Data types:
        {dtypes}

        Summary statistics:
        {summary}

        First 5 rows:
        {head}

        Missing values:
        {missing}

        Please analyze this data and return **8 to 10 specific, analytical, and data-driven insights**.
        Each insight should be:
        - A single, clear sentence
        - Based on trends, relationships, seasonality, anomalies, correlations, or clustering
        - Focused on actionable or surprising findings
        - Not generic advice or vague interpretations

        Do NOT include any preamble, explanation, or bullet points. Just list the insights.
        """

        response = model.generate_content(prompt)

        # Process response to extract insights
        insights_text = response.text
        insights = [line.strip() for line in insights_text.split('\n') if line.strip() and not line.strip().startswith('•') and len(line.strip()) > 10]

        # Limit to top 10 insights
        insights = insights[:10]

        return insights

    except Exception as e:
        st.error(f"Error generating AI insights: {e}")
        return [f"Could not generate insights due to an error: {str(e)}"]

# Function to create data profile
def create_data_profile(df):
    if df is None:
        return None
    
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": len(df.select_dtypes(include=['number']).columns),
        "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": len(df.select_dtypes(include=['datetime']).columns),
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        "column_stats": {}
    }
    
    # Get detailed stats for each numerical column
    for col in df.select_dtypes(include=['number']).columns:
        profile["column_stats"][col] = {
            "mean": df[col].mean() if not pd.isna(df[col].mean()) else 0,
            "median": df[col].median() if not pd.isna(df[col].median()) else 0,
            "std": df[col].std() if not pd.isna(df[col].std()) else 0,
            "min": df[col].min() if not pd.isna(df[col].min()) else 0,
            "max": df[col].max() if not pd.isna(df[col].max()) else 0,
            "nulls": df[col].isna().sum(),
            "nulls_percentage": (df[col].isna().sum() / len(df)) * 100
        }
    
    # Get categorical columns distribution
    profile["categorical_distribution"] = {}
    for col in df.select_dtypes(include=['object', 'category']).columns[:5]:  # Limit to first 5 categorical columns
        value_counts = df[col].value_counts().head(5).to_dict()
        profile["categorical_distribution"][col] = value_counts
    
    # Get correlation matrix
    if len(df.select_dtypes(include=['number']).columns) > 1:
        profile["correlation"] = df.select_dtypes(include=['number']).corr().round(2).to_dict()
    
    return profile

# Function to generate univariate analysis charts
def generate_univariate_charts(df, max_charts=6):
    charts = {}
    
    # Generate histograms for numeric columns
    for col in df.select_dtypes(include=['number']).columns[:max_charts]:
        try:
            # Create histogram with KDE
            fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Frequency",
                template="plotly_white"
            )
            charts[col] = fig
            
        except Exception as e:
            st.warning(f"Could not create histogram for {col}: {e}")
    
    # Generate bar charts for categorical columns with few unique values
    for col in df.select_dtypes(include=['object', 'category']).columns[:max_charts]:
        try:
            if df[col].nunique() <= 15:  # Only for columns with few unique values
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'count']
                
                fig = px.bar(
                    value_counts, 
                    x=col, 
                    y='count', 
                    title=f"Distribution of {col}",
                    template="plotly_white"
                )
                fig.update_layout(
                    xaxis_title=col,
                    yaxis_title="Count"
                )
                charts[f"{col} (categorical)"] = fig
                
        except Exception as e:
            st.warning(f"Could not create bar chart for {col}: {e}")
    
    return charts

# Function to generate bivariate analysis charts
def generate_bivariate_charts(df, max_charts=6):
    charts = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) >= 2:
        # Generate scatter plots between numeric columns
        pairs_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if pairs_count >= max_charts:
                    break
                    
                try:
                    fig = px.scatter(
                        df, 
                        x=col1, 
                        y=col2, 
                        title=f"{col1} vs {col2}",
                        template="plotly_white",
                        trendline="ols"
                    )
                    fig.update_layout(
                        xaxis_title=col1,
                        yaxis_title=col2
                    )
                    charts[f"{col1} vs {col2}"] = fig
                    pairs_count += 1
                    
                except Exception as e:
                    st.warning(f"Could not create scatter plot: {e}")
    
    # Generate box plots for numeric vs categorical
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
            for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                if df[cat_col].nunique() <= 10:  # Only for columns with few unique values
                    try:
                        fig = px.box(
                            df, 
                            x=cat_col, 
                            y=num_col, 
                            title=f"{num_col} by {cat_col}",
                            template="plotly_white"
                        )
                        fig.update_layout(
                            xaxis_title=cat_col,
                            yaxis_title=num_col
                        )
                        charts[f"{num_col} by {cat_col}"] = fig
                        
                    except Exception as e:
                        st.warning(f"Could not create box plot: {e}")
    
    return charts

# Sidebar with app info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dashboard-layout.png", width=80)
    st.title("DataInsight Pro")
    st.markdown("---")
    
    st.subheader("🔍 About")
    st.write("DataInsight Pro is an advanced data analysis and visualization platform designed to help you quickly extract insights from your data.")
    
    st.markdown("---")
    
    # Reset app button
    if st.button("🔄 Reset App"):
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Header
st.markdown("""
<div style="background-color:#4a90e2; padding:20px; border-radius:10px; margin-bottom:30px">
    <h1 style="color:white; text-align:center; margin:0">📊 DataInsight Pro</h1>
    <p style="color:white; text-align:center; margin:0">Powerful data visualization and analysis platform</p>
</div>
""", unsafe_allow_html=True)

# File uploader card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📁 Data Source")

uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, or JSON)", 
                                  type=["csv", "xlsx", "xls", "json"],
                                  help="Maximum file size: 200MB")

if 'filename' in st.session_state and st.session_state.filename:
    st.info(f"Current file: {st.session_state.filename}")

# Fix for DataFrame truth value error - use "is None" instead
sample_data = st.checkbox("Use sample data instead", value=(not uploaded_file and (st.session_state.data is None)))

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        st.session_state.data = df
        st.session_state.filename = uploaded_file.name
        st.success(f"✅ Successfully loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
elif sample_data:
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Sports']
    products = [f'Product {chr(65+i)}' for i in range(15)]
    
    data = {
        'date': np.random.choice(dates, 100),
        'product': np.random.choice(products, 100),
        'category': np.random.choice(categories, 100),
        'sales': np.random.randint(100, 2000, 100),
        'revenue': np.random.uniform(1000, 50000, 100).round(2),
        'cost': np.random.uniform(500, 30000, 100).round(2),
        'units': np.random.randint(10, 200, 100),
        'returns': np.random.randint(0, 20, 100),
        'customer_rating': np.random.uniform(1, 5, 100).round(1)
    }
    
    # Add some missing values
    for col in ['sales', 'revenue', 'customer_rating']:
        idx = np.random.choice(range(100), 5, replace=False)
        data[col] = pd.Series(data[col])
        data[col].iloc[idx] = np.nan
        
    df = pd.DataFrame(data)
    # Calculate profit
    df['profit'] = df['revenue'] - df['cost']
    
    st.session_state.data = df
    st.session_state.filename = "sample_data.csv"
    st.success("✅ Sample data loaded successfully")

st.markdown('</div>', unsafe_allow_html=True)

# Main tabs
if st.session_state.data is not None:
    tabs = st.tabs(["🔢 Data Table", "📊 Visualizations", "📋 Data Profile", "💡 AI Insights", "🔎 SQL Explorer", "📄 Report Generator"])
    
    # Data Table tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Add search and filter functionalities
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search data", "")
        with col_filter:
            num_rows = st.slider("Rows", min_value=5, max_value=100, value=10, step=5)
        
        # Apply search filter if provided
        if search_term:
            filtered_df = st.session_state.data[
                st.session_state.data.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False).any(), axis=1
                )
            ]
            st.write(f"Showing {len(filtered_df)} of {len(st.session_state.data)} rows")
            st.dataframe(filtered_df.head(num_rows), use_container_width=True, height=400)
        else:
            st.dataframe(st.session_state.data.head(num_rows), use_container_width=True, height=400)
        
        # Export options
        st.markdown("### Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("Export as CSV", key="export_csv"):
                # Export data as CSV
                csv = st.session_state.data.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="data_export.csv" class="downloadBtn">📥 Download CSV</a>'
                st.markdown(href_csv, unsafe_allow_html=True)
        
        with export_col2:
            if st.button("Export as Excel", key="export_excel"):
                # Export data as Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.data.to_excel(writer, index=False)
                b64_excel = base64.b64encode(buffer.getvalue()).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="data_export.xlsx" class="downloadBtn">📥 Download Excel</a>'
                st.markdown(href_excel, unsafe_allow_html=True)
        
        with export_col3:
            if st.button("Show Data Stats", key="data_stats"):
                st.write("### Data Statistics")
                st.write(st.session_state.data.describe())
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Visualization")
        
        # Column selection and chart type
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            chart_type = st.selectbox(
                "Chart Type", 
                ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap"],
                key="chart_type_select"
            )
            st.session_state.chart_type = chart_type.lower().replace(" ", "_")
        
        with viz_col2:
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            x_axis = st.selectbox("X-Axis", st.session_state.data.columns.tolist(), key="x_axis_select")
            
        with viz_col3:
            if chart_type != "Pie Chart" and chart_type != "Histogram":
                y_axis = st.selectbox("Y-Axis", numeric_cols, key="y_axis_select")
            else:
                y_axis = st.selectbox("Value", numeric_cols, key="value_select")
        
        # Additional options based on chart type
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            group_col = st.selectbox(
                "Group By (Color)", 
                ["None"] + [col for col in st.session_state.data.columns if st.session_state.data[col].nunique() <= 10],
                key="group_col_select"
            )
        
        # Generate the selected visualization
        st.markdown("### Chart Preview")
        
        try:
            if chart_type == "Bar Chart":
                if 'group_col' in locals() and group_col != "None":
                    fig = px.bar(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis, 
                        color=group_col,
                        title=f"{y_axis} by {x_axis}",
                        template="plotly_white"
                    )
                else:
                    fig = px.bar(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis, 
                        title=f"{y_axis} by {x_axis}",
                        template="plotly_white"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Line Chart":
                if 'group_col' in locals() and group_col != "None":
                    fig = px.line(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis, 
                        color=group_col,
                        markers=True,
                        title=f"{y_axis} Over {x_axis}",
                        template="plotly_white"
                    )
                else:
                    fig = px.line(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis,
                        markers=True, 
                        title=f"{y_axis} Over {x_axis}",
                        template="plotly_white"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Scatter Plot":
                if 'group_col' in locals() and group_col != "None":
                    fig = px.scatter(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis, 
                        color=group_col,
                        title=f"{y_axis} vs {x_axis}",
                        template="plotly_white"
                    )
                else:
                    fig = px.scatter(
                        st.session_state.data, 
                        x=x_axis, 
                        y=y_axis,
                        title=f"{y_axis} vs {x_axis}",
                        template="plotly_white"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Histogram":
                fig = px.histogram(
                    st.session_state.data, 
                    x=x_axis,
                    title=f"Distribution of {x_axis}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Box Plot":
                if 'group_col' in locals() and group_col != "None" and group_col in st.session_state.data.columns:
                    fig = px.box(
                        st.session_state.data, 
                        x=group_col, 
                        y=y_axis,
                        title=f"Box Plot of {y_axis} by {group_col}",
                        template="plotly_white"
                    )
                else:
                    fig = px.box(
                        st.session_state.data, 
                        y=y_axis,
                        title=f"Box Plot of {y_axis}",
                        template="plotly_white"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Pie Chart":
                # For pie charts, we need to aggregate the data
                if x_axis in st.session_state.data.columns:
                    pie_data = st.session_state.data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.pie(
                        pie_data, 
                        names=x_axis, 
                        values=y_axis,
                        title=f"Distribution of {y_axis} by {x_axis}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Column {x_axis} not found in the data.")
            
            elif chart_type == "Heatmap":
                # For heatmap, let's create correlation matrix
                corr_data = st.session_state.data.select_dtypes(include=['number']).corr()
                fig = px.imshow(
                    corr_data,
                    text_auto=True,
                    title="Correlation Heatmap",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating chart: {e}")
            st.info("Try selecting different columns or chart types.")
        
        # Advanced Visualization Section - Univariate & Bivariate Analysis
        st.markdown("### Advanced Analysis")
        
        analysis_tabs = st.tabs(["Univariate Analysis", "Bivariate Analysis"])
        
        with analysis_tabs[0]:
            st.markdown("#### Univariate Analysis")
            st.info("Analyze the distribution of individual variables")
            
            if st.button("Generate Univariate Plots"):
                with st.spinner("Generating univariate analysis..."):
                    univariate_figs = generate_univariate_charts(st.session_state.data)
                    st.session_state.univariate_figs = univariate_figs
                    st.success(f"Generated {len(univariate_figs)} univariate plots!")
            
            if 'univariate_figs' in st.session_state and st.session_state.univariate_figs:
                for name, fig in st.session_state.univariate_figs.items():
                    st.subheader(f"Distribution of {name}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[1]:
            st.markdown("#### Bivariate Analysis")
            st.info("Analyze relationships between pairs of variables")
            
            if st.button("Generate Bivariate Plots"):
                with st.spinner("Generating bivariate analysis..."):
                    bivariate_figs = generate_bivariate_charts(st.session_state.data)
                    st.session_state.bivariate_figs = bivariate_figs
                    st.success(f"Generated {len(bivariate_figs)} bivariate plots!")
            
            if 'bivariate_figs' in st.session_state and st.session_state.bivariate_figs:
                for name, fig in st.session_state.bivariate_figs.items():
                    st.subheader(name)
                    st.plotly_chart(fig, use_container_width=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Data Profile tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Data Profile")
        
        # Generate profile button
        if st.button("Generate Data Profile"):
            with st.spinner("Generating data profile..."):
                # Actual profile generation
                st.session_state.profile_data = create_data_profile(st.session_state.data)
                st.success("✅ Profile generated successfully!")
        
        # Display profile if available
        if st.session_state.profile_data:
            profile = st.session_state.profile_data
            
            # Overview metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="stat-box">
                    <p class="metric-value">{}</p>
                    <p class="metric-label">Total Rows</p>
                </div>
                """.format(profile["rows"]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-box">
                    <p class="metric-value">{}</p>
                    <p class="metric-label">Total Columns</p>
                </div>
                """.format(profile["columns"]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stat-box">
                    <p class="metric-value">{}</p>
                    <p class="metric-label">Missing Values</p>
                </div>
                """.format(profile["missing_values"]), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="stat-box">
                    <p class="metric-value">{}</p>
                    <p class="metric-label">Duplicate Rows</p>
                </div>
                """.format(profile["duplicate_rows"]), unsafe_allow_html=True)
            
            # Column type distribution
            st.markdown("### Column Composition")
            
            # Column type distribution chart using columns
            col_types_data = {
                "Numerical": profile["numerical_columns"],
                "Categorical": profile["categorical_columns"],
                "Datetime": profile["datetime_columns"]
            }
            
            # Create a DataFrame for the column type distribution
            col_types_df = pd.DataFrame({
                'Type': list(col_types_data.keys()),
                'Count': list(col_types_data.values())
            })
            
            fig_col_types = px.pie(
                col_types_df,
                values='Count',
                names='Type',
                title="Column Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_col_types, use_container_width=True)
            
            # Detailed column statistics
            st.markdown("### Numerical Column Statistics")
            
            if profile["column_stats"]:
                # Create a DataFrame from column stats for easier display
                stats_data = []
                for col_name, stats in profile["column_stats"].items():
                    stats_data.append({
                        "Column": col_name,
                        "Mean": round(stats["mean"], 2),
                        "Median": round(stats["median"], 2),
                        "Std Dev": round(stats["std"], 2),
                        "Min": round(stats["min"], 2),
                        "Max": round(stats["max"], 2),
                        "Missing (%)": round(stats["nulls_percentage"], 2)
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Display missing values chart for numerical columns
                missing_data = {col: stats["nulls"] for col, stats in profile["column_stats"].items()}
                if any(missing_data.values()):
                    # Create a DataFrame for the missing values
                    missing_df = pd.DataFrame({
                        'Column': list(missing_data.keys()),
                        'Missing Values': list(missing_data.values())
                    })
                    
                    fig_missing = px.bar(
                        missing_df,
                        x='Column',
                        y='Missing Values',
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.info("No numerical columns found in the dataset.")
            
            # Categorical data distributions
            if "categorical_distribution" in profile and profile["categorical_distribution"]:
                st.markdown("### Categorical Column Distributions")
                for col_name, distribution in profile["categorical_distribution"].items():
                    if distribution:  # Check if distribution isn't empty
                        # Convert the distribution to a DataFrame
                        dist_df = pd.DataFrame({
                            'Value': list(distribution.keys()),
                            'Count': list(distribution.values())
                        })
                        
                        fig = px.bar(
                            dist_df,
                            x='Value',
                            y='Count',
                            title=f"Distribution of {col_name}",
                            color='Count'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            if "correlation" in profile and profile["correlation"]:
                st.markdown("### Correlation Matrix")
                # Convert the nested dictionary to a DataFrame
                corr_df = pd.DataFrame(profile["correlation"])
                
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=True,
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Memory usage
            st.markdown("### Memory Usage")
            st.info(f"Total memory usage: {profile['memory_usage']:.2f} MB")
            
        else:
            st.info("👆 Click 'Generate Data Profile' to analyze your dataset")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    # AI Insights tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 AI Insights")
        
        # Generate insights button
        if st.button("Generate AI Insights"):
            with st.spinner("AI is analyzing your data..."):
                # Generate insights
                insights = generate_ai_insights(st.session_state.data)
                
                if not genai_available:
                    st.warning("⚠️ AI features unavailable. Using sample insights instead.")
                
                st.session_state.insights = insights
                st.success("✅ Insights generated successfully!")
        
        # Display insights if available
        if st.session_state.insights:
            for i, insight in enumerate(st.session_state.insights):
                st.markdown(f"""
                <div class="insight-card">
                    <strong>Insight {i+1}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
                
            # Visual summary of insights
            st.markdown("### Key Findings Visualization")
            
            try:
                # Create a simple gauge chart showing the insight strength (just for visual effect)
                insight_strengths = np.random.uniform(0.6, 1.0, len(st.session_state.insights))
                insight_df = pd.DataFrame({
                    'Insight': [f"Insight {i+1}" for i in range(len(st.session_state.insights))],
                    'Strength': insight_strengths
                })
                
                fig = px.bar(
                    insight_df,
                    x='Insight',
                    y='Strength',
                    title="Insight Confidence Levels",
                    color='Strength',
                    color_continuous_scale='Viridis',
                    range_y=[0, 1]
                )
                fig.update_layout(yaxis_title="Confidence Level")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create insight visualization: {e}")
                
        else:
            st.info("👆 Click 'Generate AI Insights' to get AI-powered analysis of your data")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    # SQL Explorer tab
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔎 SQL Explorer")
        
        if st.button("Generate Example SQL Queries"):
            with st.spinner("Generating SQL queries..."):
                # Generate SQL queries
                sql_queries = generate_sql_queries(st.session_state.data)
                st.session_state.sql_queries = sql_queries
                st.success("✅ SQL queries generated successfully!")
        
        # Display SQL queries
        if 'sql_queries' in st.session_state and st.session_state.sql_queries:
            st.markdown("### Example SQL Queries")
            st.markdown("These SQL queries can be used to explore your dataset:")
            
            for i, query in enumerate(st.session_state.sql_queries):
                st.markdown(f"""
                <div class="sql-card">
                    <strong>Query {i+1}:</strong><br>
                    <code>{query}</code>
                </div>
                """, unsafe_allow_html=True)
            
            # Custom SQL query input
            st.markdown("### Write Your Own Query")
            custom_query = st.text_area(
                "Enter SQL Query",
                "SELECT * FROM data LIMIT 10;",
                height=100
            )
            
            if st.button("Execute Query (Demo)"):
                st.info("This is a demo feature. In a real application, this would execute your SQL query on the dataset.")
                st.code(custom_query, language="sql")
        else:
            st.info("👆 Click 'Generate Example SQL Queries' to get SQL examples for your dataset")
        
        st.markdown('</div>', unsafe_allow_html=True)
            
    # Report Generator tab (NEW)
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Report Generator")
        
        report_title = st.text_input("Report Title", "Data Analysis Report")
        
        # Report options
        report_format = st.selectbox("Report Format", ["PDF"])
        
        # Elements to include
        st.markdown("### Include in Report")
        include_col1, include_col2 = st.columns(2)
        
        with include_col1:
            include_charts = st.checkbox("Data Visualizations", value=True)
            include_summary = st.checkbox("Statistical Summary", value=True) 
            include_data = st.checkbox("Sample Data", value=True)
        
        with include_col2:
            include_insights = st.checkbox("AI Insights", value=True)
            include_sql = st.checkbox("SQL Queries", value=True)
        
        # Generate comprehensive report
        if st.button("Generate Comprehensive Report", key="gen_report", type="primary"):
            if st.session_state.data is not None:
                with st.spinner("Generating comprehensive report... This may take a minute."):
                    try:
                        # Create univariate and bivariate plots if not already created
                        if not 'univariate_figs' in st.session_state or not st.session_state.univariate_figs:
                            st.session_state.univariate_figs = generate_univariate_charts(st.session_state.data)
                            
                        if not 'bivariate_figs' in st.session_state or not st.session_state.bivariate_figs:
                            st.session_state.bivariate_figs = generate_bivariate_charts(st.session_state.data)
                        
                        # Create insights if not already generated
                        if not st.session_state.insights:
                            st.session_state.insights = generate_ai_insights(st.session_state.data)
                            
                        # Create SQL queries if not already generated
                        if not st.session_state.sql_queries:
                            st.session_state.sql_queries = generate_sql_queries(st.session_state.data)
                        
                        # Generate the enhanced PDF report
                        pdf_path = generate_enhanced_pdf_report(
                            st.session_state.data,
                            report_title,
                            include_charts,
                            include_summary,
                            include_data,
                            include_insights,
                            include_sql
                        )
                        
                        if pdf_path:
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            
                            st.download_button(
                                label="📥 Download Comprehensive Report",
                                data=pdf_bytes,
                                file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key="download_report"
                            )
                            st.success("✅ Comprehensive report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.error("Please upload or use sample data first.")

        st.markdown('<div style="margin-top: 20px; padding: 15px; border: 1px solid #4a90e2; border-radius: 5px;">', unsafe_allow_html=True)
        st.write("""
        #### Report Features
        - Comprehensive data summary with statistics and column distributions
        - Visual data profile with charts and graphs
        - AI-generated insights and recommendations
        - SQL queries for further analysis
        - Univariate and bivariate analysis visualizations
        - Correlation analysis and missing data reports
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Data Transformations Card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔄 Data Transformations")

if st.session_state.data is not None:
    transform_option = st.selectbox(
        "Select Transformation", 
        [
            "Filter Data",
            "Sort Data",
            "Handle Missing Values",
            "Create New Column",
            "Convert Data Types",
            "Group and Aggregate",
        ]
    )
    
    if transform_option == "Filter Data":
        filter_col = st.selectbox("Column to filter", st.session_state.data.columns)
        
        # Different filter options based on column type
        if pd.api.types.is_numeric_dtype(st.session_state.data[filter_col]):
            min_val = float(st.session_state.data[filter_col].min())
            max_val = float(st.session_state.data[filter_col].max())
            filter_range = st.slider(
                "Range", 
                min_value=min_val, 
                max_value=max_val,
                value=(min_val, max_val)
            )
            if st.button("Apply Filter"):
                filtered = st.session_state.data[(st.session_state.data[filter_col] >= filter_range[0]) & 
                                                (st.session_state.data[filter_col] <= filter_range[1])]
                st.session_state.data = filtered
                st.success(f"Data filtered to {len(filtered)} rows")
            
        else:
            unique_vals = st.session_state.data[filter_col].unique()
            selected_vals = st.multiselect("Select values", unique_vals, unique_vals)
            
            if st.button("Apply Filter"):
                filtered = st.session_state.data[st.session_state.data[filter_col].isin(selected_vals)]
                st.session_state.data = filtered
                st.success(f"Data filtered to {len(filtered)} rows")
            
    elif transform_option == "Sort Data":
        sort_col = st.selectbox("Sort by", st.session_state.data.columns)
        sort_order = st.radio("Order", ["Ascending", "Descending"])
        
        if st.button("Apply Sort"):
            st.session_state.data = st.session_state.data.sort_values(
                by=sort_col, 
                ascending=(sort_order == "Ascending")
            )
            st.success("Data sorted successfully")
            
    elif transform_option == "Handle Missing Values":
        missing_cols = st.session_state.data.columns[st.session_state.data.isna().any()]
        if len(missing_cols) == 0:
            st.info("No missing values found in the dataset")
        else:
            missing_col = st.selectbox("Column with missing values", missing_cols)
            missing_count = st.session_state.data[missing_col].isna().sum()
            st.write(f"Missing values in {missing_col}: {missing_count}")
            
            strategy = st.selectbox(
                "Strategy", 
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"]
            )
            
            if st.button("Apply Strategy"):
                if strategy == "Drop rows":
                    st.session_state.data = st.session_state.data.dropna(subset=[missing_col])
                    st.success(f"Dropped rows with missing values in {missing_col}")
                elif strategy == "Fill with mean":
                    if pd.api.types.is_numeric_dtype(st.session_state.data[missing_col]):
                        mean_val = st.session_state.data[missing_col].mean()
                        st.session_state.data[missing_col] = st.session_state.data[missing_col].fillna(mean_val)
                        st.success(f"Filled missing values with mean: {mean_val:.2f}")
                    else:
                        st.error("Cannot use mean for non-numeric column")
                elif strategy == "Fill with median":
                    if pd.api.types.is_numeric_dtype(st.session_state.data[missing_col]):
                        median_val = st.session_state.data[missing_col].median()
                        st.session_state.data[missing_col] = st.session_state.data[missing_col].fillna(median_val)
                        st.success(f"Filled missing values with median: {median_val:.2f}")
                    else:
                        st.error("Cannot use median for non-numeric column")
                elif strategy == "Fill with mode":
                    mode_val = st.session_state.data[missing_col].mode()[0]
                    st.session_state.data[missing_col] = st.session_state.data[missing_col].fillna(mode_val)
                    st.success(f"Filled missing values with mode: {mode_val}")
                elif strategy == "Fill with value":
                    fill_val = st.text_input("Value to fill")
                    if fill_val:
                        # Convert to appropriate type
                        if pd.api.types.is_numeric_dtype(st.session_state.data[missing_col]):
                            try:
                                fill_val = float(fill_val)
                            except:
                                st.error("Please enter a valid number")
                        st.session_state.data[missing_col] = st.session_state.data[missing_col].fillna(fill_val)
                        st.success(f"Filled missing values with: {fill_val}")
            
    elif transform_option == "Create New Column":
        st.write("Define a new column based on existing columns")
        
        new_col_name = st.text_input("New column name")
        
        formula_type = st.selectbox(
            "Formula type", 
            ["Basic arithmetic", "Conditional logic", "Text operations"]
        )
        
        if formula_type == "Basic arithmetic":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                col1_name = st.selectbox("First column", numeric_cols, key="arith_col1")
                operation = st.selectbox("Operation", ["+", "-", "*", "/"], key="arith_op")
                col2_name = st.selectbox("Second column", numeric_cols, key="arith_col2")
                
                if st.button("Create Column"):
                    try:
                        if operation == "+":
                            st.session_state.data[new_col_name] = st.session_state.data[col1_name] + st.session_state.data[col2_name]
                        elif operation == "-":
                            st.session_state.data[new_col_name] = st.session_state.data[col1_name] - st.session_state.data[col2_name]
                        elif operation == "*":
                            st.session_state.data[new_col_name] = st.session_state.data[col1_name] * st.session_state.data[col2_name]
                        elif operation == "/":
                            st.session_state.data[new_col_name] = st.session_state.data[col1_name] / st.session_state.data[col2_name]
                        
                        st.success(f"Created new column: {new_col_name}")
                    except Exception as e:
                        st.error(f"Error creating column: {e}")
            else:
                st.error("Need at least two numeric columns for arithmetic operations")
            
        elif formula_type == "Conditional logic":
            condition_col = st.selectbox("Condition column", st.session_state.data.columns)
            
            if pd.api.types.is_numeric_dtype(st.session_state.data[condition_col]):
                condition_op = st.selectbox(
                    "Condition", 
                    ["greater than (>)", "less than (<)", "equal to (==)", "not equal to (!=)"]
                )
                condition_val = st.number_input("Value")
            else:
                condition_op = st.selectbox(
                    "Condition", 
                    ["equal to (==)", "not equal to (!=)", "contains", "starts with"]
                )
                condition_val = st.text_input("Value")
            
            true_val = st.text_input("Value if condition is true")
            false_val = st.text_input("Value if condition is false")
            
            if st.button("Create Column"):
                try:
                    if condition_op == "greater than (>)":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col] > float(condition_val), 
                            true_val, 
                            false_val
                        )
                    elif condition_op == "less than (<)":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col] < float(condition_val), 
                            true_val, 
                            false_val
                        )
                    elif condition_op == "equal to (==)":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col] == condition_val, 
                            true_val, 
                            false_val
                        )
                    elif condition_op == "not equal to (!=)":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col] != condition_val, 
                            true_val, 
                            false_val
                        )
                    elif condition_op == "contains":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col].astype(str).str.contains(str(condition_val)), 
                            true_val, 
                            false_val
                        )
                    elif condition_op == "starts with":
                        st.session_state.data[new_col_name] = np.where(
                            st.session_state.data[condition_col].astype(str).str.startswith(str(condition_val)), 
                            true_val, 
                            false_val
                        )
                    
                    st.success(f"Created new column: {new_col_name}")
                except Exception as e:
                    st.error(f"Error creating column: {e}")
            
        elif formula_type == "Text operations":
            text_cols = st.session_state.data.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                text_operation = st.selectbox(
                    "Text operation", 
                    ["Extract substring", "Convert to uppercase", "Convert to lowercase", "String length", "Concatenate columns"]
                )
                
                if text_operation == "Extract substring":
                    text_col = st.selectbox("Text column", text_cols)
                    start_pos = st.number_input("Start position", min_value=0, value=0)
                    end_pos = st.number_input("End position", min_value=0, value=5)
                    
                    if st.button("Create Column"):
                        try:
                            st.session_state.data[new_col_name] = st.session_state.data[text_col].str[start_pos:end_pos]
                            st.success(f"Created new column: {new_col_name}")
                        except Exception as e:
                            st.error(f"Error creating column: {e}")
                
                elif text_operation in ["Convert to uppercase", "Convert to lowercase"]:
                    text_col = st.selectbox("Text column", text_cols)
                    
                    if st.button("Create Column"):
                        try:
                            if text_operation == "Convert to uppercase":
                                st.session_state.data[new_col_name] = st.session_state.data[text_col].str.upper()
                            else:
                                st.session_state.data[new_col_name] = st.session_state.data[text_col].str.lower()
                            st.success(f"Created new column: {new_col_name}")
                        except Exception as e:
                            st.error(f"Error creating column: {e}")
                
                elif text_operation == "String length":
                    text_col = st.selectbox("Text column", text_cols)
                    
                    if st.button("Create Column"):
                        try:
                            st.session_state.data[new_col_name] = st.session_state.data[text_col].str.len()
                            st.success(f"Created new column: {new_col_name}")
                        except Exception as e:
                            st.error(f"Error creating column: {e}")
                
                elif text_operation == "Concatenate columns":
                    text_col1 = st.selectbox("First column", st.session_state.data.columns)
                    separator = st.text_input("Separator", " ")
                    text_col2 = st.selectbox("Second column", st.session_state.data.columns)
                    
                    if st.button("Create Column"):
                        try:
                            st.session_state.data[new_col_name] = st.session_state.data[text_col1].astype(str) + separator + st.session_state.data[text_col2].astype(str)
                            st.success(f"Created new column: {new_col_name}")
                        except Exception as e:
                            st.error(f"Error creating column: {e}")
            else:
                st.error("No text columns found in the dataset")
            
    elif transform_option == "Convert Data Types":
        col_to_convert = st.selectbox("Column to convert", st.session_state.data.columns)
        current_type = st.session_state.data[col_to_convert].dtype
        st.write(f"Current type: {current_type}")
        
        new_type = st.selectbox(
            "New type", 
            ["String", "Integer", "Float", "Boolean", "Datetime"]
        )
        
        if st.button("Convert Type"):
            try:
                if new_type == "String":
                    st.session_state.data[col_to_convert] = st.session_state.data[col_to_convert].astype(str)
                elif new_type == "Integer":
                    st.session_state.data[col_to_convert] = pd.to_numeric(st.session_state.data[col_to_convert], errors='coerce').astype('Int64')
                elif new_type == "Float":
                    st.session_state.data[col_to_convert] = pd.to_numeric(st.session_state.data[col_to_convert], errors='coerce')
                elif new_type == "Boolean":
                    st.session_state.data[col_to_convert] = st.session_state.data[col_to_convert].astype(bool)
                elif new_type == "Datetime":
                    st.session_state.data[col_to_convert] = pd.to_datetime(st.session_state.data[col_to_convert], errors='coerce')
                
                st.success(f"Converted {col_to_convert} to {new_type}")
            except Exception as e:
                st.error(f"Error converting type: {e}")
            
    elif transform_option == "Group and Aggregate":
        group_cols = st.multiselect("Group by columns", st.session_state.data.columns)
        if group_cols:
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns
            agg_col = st.selectbox("Column to aggregate", numeric_cols)
            agg_func = st.selectbox(
                "Aggregation function", 
                ["Sum", "Mean", "Median", "Min", "Max", "Count", "Standard Deviation"]
            )
            
            if st.button("Apply Aggregation"):
                try:
                    agg_func_map = {
                        "Sum": "sum",
                        "Mean": "mean",
                        "Median": "median",
                        "Min": "min",
                        "Max": "max",
                        "Count": "count",
                        "Standard Deviation": "std"
                    }
                    
                    result = st.session_state.data.groupby(group_cols)[agg_col].agg(agg_func_map[agg_func]).reset_index()
                    result.columns = group_cols + [f"{agg_col}_{agg_func_map[agg_func]}"]
                    
                    st.session_state.data = result
                    st.success("Aggregation applied successfully")
                except Exception as e:
                    st.error(f"Error during aggregation: {e}")
            
else:
    st.info("Please upload or generate sample data first")
    
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>© 2025 DataInsight Pro - Powerful Data Analysis Platform</p>
    <p>Made with ❤️ for data enthusiasts</p>
</footer>
""", unsafe_allow_html=True)


# Extra panel 0
with st.expander('Advanced Setting Panel 0'):
    st.text_input('Parameter 0', value='Default 0')
    st.slider('Threshold 0', 0, 100, 50)
    st.checkbox('Enable Option 0')


# Extra panel 1
with st.expander('Advanced Setting Panel 1'):
    st.text_input('Parameter 1', value='Default 1')
    st.slider('Threshold 1', 0, 100, 50)
    st.checkbox('Enable Option 1')


# Extra panel 2
with st.expander('Advanced Setting Panel 2'):
    st.text_input('Parameter 2', value='Default 2')
    st.slider('Threshold 2', 0, 100, 50)
    st.checkbox('Enable Option 2')


# Extra panel 3
with st.expander('Advanced Setting Panel 3'):
    st.text_input('Parameter 3', value='Default 3')
    st.slider('Threshold 3', 0, 100, 50)
    st.checkbox('Enable Option 3')


# Extra panel 4
with st.expander('Advanced Setting Panel 4'):
    st.text_input('Parameter 4', value='Default 4')
    st.slider('Threshold 4', 0, 100, 50)
    st.checkbox('Enable Option 4')


# Extra panel 5
with st.expander('Advanced Setting Panel 5'):
    st.text_input('Parameter 5', value='Default 5')
    st.slider('Threshold 5', 0, 100, 50)
    st.checkbox('Enable Option 5')


# Extra panel 6
with st.expander('Advanced Setting Panel 6'):
    st.text_input('Parameter 6', value='Default 6')
    st.slider('Threshold 6', 0, 100, 50)
    st.checkbox('Enable Option 6')


# Extra panel 7
with st.expander('Advanced Setting Panel 7'):
    st.text_input('Parameter 7', value='Default 7')
    st.slider('Threshold 7', 0, 100, 50)
    st.checkbox('Enable Option 7')


# Extra panel 8
with st.expander('Advanced Setting Panel 8'):
    st.text_input('Parameter 8', value='Default 8')
    st.slider('Threshold 8', 0, 100, 50)
    st.checkbox('Enable Option 8')


# Extra panel 9
with st.expander('Advanced Setting Panel 9'):
    st.text_input('Parameter 9', value='Default 9')
    st.slider('Threshold 9', 0, 100, 50)
    st.checkbox('Enable Option 9')


# Extra panel 10
with st.expander('Advanced Setting Panel 10'):
    st.text_input('Parameter 10', value='Default 10')
    st.slider('Threshold 10', 0, 100, 50)
    st.checkbox('Enable Option 10')


# Extra panel 11
with st.expander('Advanced Setting Panel 11'):
    st.text_input('Parameter 11', value='Default 11')
    st.slider('Threshold 11', 0, 100, 50)
    st.checkbox('Enable Option 11')


# Extra panel 12
with st.expander('Advanced Setting Panel 12'):
    st.text_input('Parameter 12', value='Default 12')
    st.slider('Threshold 12', 0, 100, 50)
    st.checkbox('Enable Option 12')


# Extra panel 13
with st.expander('Advanced Setting Panel 13'):
    st.text_input('Parameter 13', value='Default 13')
    st.slider('Threshold 13', 0, 100, 50)
    st.checkbox('Enable Option 13')


# Extra panel 14
with st.expander('Advanced Setting Panel 14'):
    st.text_input('Parameter 14', value='Default 14')
    st.slider('Threshold 14', 0, 100, 50)
    st.checkbox('Enable Option 14')


# Extra panel 15
with st.expander('Advanced Setting Panel 15'):
    st.text_input('Parameter 15', value='Default 15')
    st.slider('Threshold 15', 0, 100, 50)
    st.checkbox('Enable Option 15')


# Extra panel 16
with st.expander('Advanced Setting Panel 16'):
    st.text_input('Parameter 16', value='Default 16')
    st.slider('Threshold 16', 0, 100, 50)
    st.checkbox('Enable Option 16')
