import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from ydata_profiling import ProfileReport
from fpdf import FPDF, FPDFException
from io import BytesIO
import tempfile
from datetime import datetime
import google.generativeai as genai
import os
import base64
from PIL import Image
import time

# ========== APP CONFIGURATION ==========
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
        background-color: #357abd;
        transform: scale(1.02);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
    }
    .stRadio>div {
        flex-direction: row;
        gap: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
    }
    .stProgress>div>div>div>div {
        background-color: #4a90e2;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ========== GEMINI AI SETUP ==========
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        ai_enabled = True
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Gemini: {str(e)}")
        ai_enabled = False
else:
    ai_enabled = False
    st.sidebar.warning("🔑 Gemini API key not found in secrets. AI features disabled.")

# ========== FILE UPLOAD SECTION ==========
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Choose CSV/Excel file", type=["csv", "xlsx"], 
                                   help="Upload your dataset for analysis")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
            
            # Show basic info in sidebar
            st.subheader("Dataset Preview")
            st.dataframe(df.head(3), height=150)
            
            # Calculate memory usage
            mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.caption(f"Memory usage: {mem_usage:.2f} MB")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.stop()

# ========== MAIN APP FUNCTIONALITY ==========
if 'df' not in st.session_state:
    st.info("👋 Welcome to DataInsight Pro! Please upload a dataset to begin.")
    st.image("https://cdn.pixabay.com/photo/2018/05/18/15/30/web-design-3411373_1280.jpg", 
            use_container_width=True)
    st.stop()

df = st.session_state.df

# ========== AUTO-EDA TAB ==========
tab1, tab2, tab3, tab4 = st.tabs(["📊 Auto Analysis", "📈 Visual Explorer", "🤖 AI Insights", "📤 Export"])

with tab1:
    st.header("Automated Exploratory Data Analysis")
    
    with st.expander("⚙️ Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            analysis_mode = st.radio("Analysis Depth", ["Quick", "Comprehensive"], 
                                   help="Quick mode for fast results, Comprehensive for detailed analysis")
        with col2:
            show_correlations = st.checkbox("Show Correlation Matrix", value=True)
    
    if st.button("🚀 Run Full Analysis", type="primary"):
        with st.spinner(f"Performing {analysis_mode.lower()} analysis..."):
            start_time = time.time()
            
            config = {
                "progress_bar": False,
                "correlations": {"auto": {"calculate": show_correlations}},
                "missing_diagrams": {"heatmap": True, "dendrogram": True},
                "interactions": {"continuous": True}
            }
            
            if analysis_mode == "Quick":
                config["minimal"] = True
            else:
                config["explorative"] = True
            
            profile = ProfileReport(df, **config)
            st.session_state.profile = profile
            st.session_state.profile_html = profile.to_html()
            
            end_time = time.time()
            st.success(f"Analysis completed in {end_time - start_time:.1f} seconds!")
        
        st.components.v1.html(st.session_state.profile_html, height=1000, scrolling=True)

# ========== VISUALIZATION TAB ==========
with tab2:
    st.header("Interactive Data Visualization")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line", "Histogram", "Box", "Violin"])
        x_axis = st.selectbox("X-Axis", df.columns)
        y_axis = st.selectbox("Y-Axis", [c for c in df.columns if c != x_axis])
        
        if chart_type in ["Scatter", "Line", "Bar"]:
            color_by = st.selectbox("Color By", ["None"] + [c for c in df.columns if c != x_axis and c != y_axis])
            hover_data = st.multiselect("Hover Data", [c for c in df.columns if c != x_axis and c != y_axis])
    
    with col2:
        try:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, 
                               color=None if color_by == "None" else color_by,
                               hover_data=hover_data)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, 
                            color=None if color_by == "None" else color_by,
                            hover_data=hover_data)
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, 
                             color=None if color_by == "None" else color_by,
                             hover_data=hover_data)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis)
            elif chart_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis)
            else:  # Violin
                fig = px.violin(df, x=x_axis, y=y_axis)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create chart: {str(e)}")

# ========== AI INSIGHTS TAB ==========
with tab3:
    st.header("AI-Powered Data Insights")
    
    if not ai_enabled:
        st.warning("🔒 AI features require a Gemini API key in Space secrets")
    else:
        with st.expander("💡 Ask AI About Your Data", expanded=True):
            question = st.text_area("Your question about the data:", 
                                  placeholder="E.g., 'What are the key trends in this data?', 'Suggest machine learning models for this dataset'",
                                  height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Get Insights", help="General data analysis"):
                    with st.spinner("Analyzing with AI..."):
                        try:
                            response = model.generate_content(
                                f"""Analyze this dataset:
                                
                                Columns: {list(df.columns)}
                                First 5 rows:
                                {df.head().to_string()}
                                
                                Question: {question}
                                
                                Provide:
                                1. Key findings (bullet points)
                                2. Recommended visualizations
                                3. Potential next steps for analysis
                                4. Any data quality concerns
                                """
                            )
                            st.subheader("AI Analysis Results")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"AI Error: {str(e)}")
            
            with col2:
                if st.button("💻 Generate SQL", help="Get SQL queries for analysis"):
                    with st.spinner("Generating SQL..."):
                        try:
                            response = model.generate_content(
                                f"""Generate SQL queries for this dataset:
                                
                                Columns: {list(df.columns)}
                                Sample data types: {df.dtypes.to_dict()}
                                
                                Request: {question}
                                
                                Provide:
                                1. A well-commented SQL query
                                2. Explanation of what it does
                                3. Expected output format
                                """
                            )
                            st.subheader("Generated SQL Query")
                            st.code(response.text, language='sql')
                        except Exception as e:
                            st.error(f"AI Error: {str(e)}")

# ========== EXPORT TAB ==========
with tab4:
    st.header("Export Analysis Results")
    
    if 'profile' not in st.session_state:
        st.warning("Please run the Auto Analysis first to generate reports")
    else:
        # PDF Report Generation
        st.subheader("📄 Comprehensive PDF Report")
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Creating professional PDF report..."):
                try:
                    # Create PDF with improved layout
                    pdf = FPDF(orientation='L')  # Landscape for more space
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    
                    # Add title page
                    pdf.set_font('Arial', 'B', 24)
                    pdf.cell(0, 20, "DataInsight Pro Analysis Report", 0, 1, 'C')
                    pdf.ln(10)
                    
                    pdf.set_font('Arial', '', 12)
                    pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
                    pdf.cell(0, 10, f"Source file: {uploaded_file.name}", 0, 1, 'C')
                    pdf.ln(15)
                    
                    # Dataset Overview
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, "1. Dataset Overview", 0, 1)
                    pdf.set_font('Arial', '', 10)
                    
                    # Create a table for basic stats
                    col_widths = [60, 40, 40, 60]
                    pdf.cell(col_widths[0], 10, "Statistic", border=1)
                    pdf.cell(col_widths[1], 10, "Value", border=1)
                    pdf.cell(col_widths[2], 10, "Statistic", border=1)
                    pdf.cell(col_widths[3], 10, "Value", border=1, ln=1)
                    
                    stats = [
                        ("Total Rows", len(df)),
                        ("Total Columns", len(df.columns)),
                        ("Missing Values", df.isnull().sum().sum()),
                        ("Duplicate Rows", df.duplicated().sum()),
                        ("Numeric Columns", len(df.select_dtypes(include=np.number).columns)),
                        ("Categorical Columns", len(df.select_dtypes(include='object').columns))
                    ]
                    
                    for i in range(0, len(stats), 2):
                        pdf.cell(col_widths[0], 10, stats[i][0], border=1)
                        pdf.cell(col_widths[1], 10, str(stats[i][1]), border=1)
                        if i+1 < len(stats):
                            pdf.cell(col_widths[2], 10, stats[i+1][0], border=1)
                            pdf.cell(col_widths[3], 10, str(stats[i+1][1]), border=1, ln=1)
                        else:
                            pdf.cell(col_widths[2] + col_widths[3], 10, "", border=1, ln=1)
                    
                    pdf.ln(10)
                    
                    # Add more sections (variables, correlations, missing values, etc.)
                    # ... (additional PDF content sections would go here)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf.output(tmp.name)
                        with open(tmp.name, "rb") as f:
                            pdf_bytes = f.read()
                    
                    # Create download button
                    st.download_button(
                        label="⬇️ Download Full PDF Report",
                        data=pdf_bytes,
                        file_name="data_insight_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("Professional PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to generate PDF: {str(e)}")
        
        # HTML Report Download
        st.subheader("🌐 Interactive HTML Report")
        st.download_button(
            label="⬇️ Download HTML Report",
            data=st.session_state.profile_html,
            file_name="data_insight_report.html",
            mime="text/html"
        )
        
        # Data Export
        st.subheader("💾 Export Data")
        export_format = st.radio("Format", ["CSV", "Excel", "JSON"])
        
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="dataset_analysis.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            excel = BytesIO()
            df.to_excel(excel, index=False)
            st.download_button(
                label="⬇️ Download Excel",
                data=excel.getvalue(),
                file_name="dataset_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name="dataset_analysis.json",
                mime="application/json"
            )
