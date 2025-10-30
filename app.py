import streamlit as st
import requests
import pandas as pd
import json
import os

# === !! CRITICAL UPDATE FOR DOCKER !! ===
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# ==========================================

# --- Page Config ---
st.set_page_config(
    page_title="A/B Testing Suite",
    page_icon="🧪",
    layout="wide"
)

# --- App State ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = None

# --- Helper Functions ---
def clear_results():
    """Clears the stored analysis results."""
    st.session_state.analysis_results = None
    st.session_state.executive_summary = None

# --- Main App ---
st.title("🧪 A/B Testing Suite")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs([
    "🚀 1. Experiment Designer", 
    "🧮 2. Power Calculator", 
    "📊 3. Results Analyzer"
])


# ==============================================================================
# --- TAB 1: EXPERIMENT DESIGNER ---
# ==============================================================================
with tab1:
    st.header("LLM Experiment Designer")
    st.write("Get a complete A/B experiment spec from a simple product idea.")
    
    with st.form("designer_form"):
        product_idea = st.text_area(
            "Enter your product idea:", 
            "Change the main 'Buy Now' button on the checkout page from blue to bright green."
        )
        submit_design = st.form_submit_button("Design Experiment")

    if submit_design:
        if not product_idea:
            st.error("Please enter a product idea.")
        else:
            with st.spinner("LLM agent is designing your experiment..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/design-experiment",
                        # --- THIS IS THE FIX ---
                        # The API expects "product_idea", not "idea"
                        json={"product_idea": product_idea}
                        # -------------------------
                    )
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    st.subheader("🤖 Generated Experiment Spec")
                    st.json(response.json())
                except requests.exceptions.RequestException as e:
                    # This is where the 500 or 422 error was being caught
                    st.error(f"Failed to connect to API: {e}") 
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ==============================================================================
# --- TAB 2: POWER CALCULATOR ---
# ==============================================================================
with tab2:
    st.header("Sample Size Calculator (Power Analysis)")
    st.write("Calculate the required sample size per variation for a **continuous metric**.")
    
    with st.form("power_calc_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            baseline_mean = st.number_input(
                "Baseline Metric Value", 
                min_value=0.0, 
                value=10.0, 
                help="The average value of your metric for the control group (e.g., $10.00)."
            )
        with col2:
            std_dev = st.number_input(
                "Metric Standard Deviation", 
                min_value=0.1, 
                value=5.0, 
                help="The standard deviation of your metric (e.g., 5.0)."
            )
        with col3:
            mde_absolute = st.number_input(
                "Minimum Detectable Effect (Absolute)", 
                min_value=0.01, 
                value=0.5, 
                help="The smallest absolute lift you want to detect (e.g., $0.50)."
            )
        
        submit_power = st.form_submit_button("Calculate Sample Size")

    if submit_power:
        with st.spinner("Calculating..."):
            try:
                payload = {
                    "baseline_mean": baseline_mean,
                    "mde_absolute": mde_absolute,
                    "std_dev": std_dev
                }
                response = requests.post(
                    f"{BACKEND_URL}/calculate-power",
                    json=payload
                )
                response.raise_for_status()
                
                size = response.json().get('required_sample_size_per_variation')
                if size:
                    st.metric(
                        "Required Sample Size (Per Variation)", 
                        f"{size:,}"
                    )
                    st.info(f"You need a total of **{size*2:,}** users in your experiment (assuming 2 variations).")
                else:
                    st.error("Invalid response from API.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")


# ==============================================================================
# --- TAB 3: RESULTS ANALYZER ---
# ==============================================================================
with tab3:
    st.header("Analyze Experiment Results")
    
    # --- Form for all inputs ---
    with st.form("analysis_form"): # clear_on_submit=True was removed, as we use submit button logic
        st.write("Select your analysis type and provide the column names from your CSV.")
        
        # --- Analysis Type Selection ---
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "Continuous (CUPED)", 
                "Proportion (Z-Test)", 
                "Sequential (Bonferroni)"
            ],
            help="""
            - **Continuous (CUPED):** For metrics like Revenue, Session Time. Requires a pre-experiment covariate.
            - **Proportion (Z-Test):** For conversion rates (e.g., Clicks / Sessions).
            - **Sequential (Bonferroni):** For continuous metrics where you "peeked" at results.
            """
        )
        
        st.subheader("Required Column Names")
        
        # --- Common Inputs ---
        col1, col2 = st.columns(2)
        with col1:
            user_col = st.text_input("User ID Column", "user_id")
        with col2:
            variation_col = st.text_input("Variation Column", "variation")

        # --- Conditional Inputs ---
        metric_col, covariate_col, numerator_col, denominator_col, total_peeks = (None,)*5

        if analysis_type == "Continuous (CUPED)":
            col1, col2 = st.columns(2)
            with col1:
                metric_col = st.text_input("Metric Column (e.g., 'revenue')", "revenue")
            with col2:
                covariate_col = st.text_input("Covariate Column (e.g., 'revenue_pre')", "revenue_pre")
        
        elif analysis_type == "Proportion (Z-Test)":
            col1, col2 = st.columns(2)
            with col1:
                numerator_col = st.text_input("Numerator Column (e.g., 'converted')", "converted")
            with col2:
                denominator_col = st.text_input("Denominator Unit (e.g., 'session_id')", "session_id")
            st.info("Note: The Z-Test aggregates your data. The 'Denominator Unit' column will be *counted* to get the total N for each user (e.g., total sessions). The 'Numerator Column' will be *summed*.")

        elif analysis_type == "Sequential (Bonferroni)":
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_col = st.text_input("Metric Column (e.g., 'revenue')", "revenue")
            with col2:
                covariate_col = st.text_input("Covariate Column (e.g., 'revenue_pre')", "revenue_pre")
            with col3:
                total_peeks = st.number_input("Total Planned Peeks", min_value=1, value=4, help="The total number of times you planned to check this test (e.g., 4 = 4 weekly checks).")

        # --- File Uploader ---
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader(
            "Upload your experiment results (CSV)", 
            type="csv"
            # on_change=clear_results was removed to fix form callback error
        )
        
        # --- Submit Button ---
        submit_analysis = st.form_submit_button("Analyze Results")

    # --- Analysis Logic (Outside Form) ---
    if submit_analysis:
        # When submit is clicked, we clear the old results *first*
        clear_results() 
        
        if not uploaded_file:
            st.error("Please upload a CSV file.")
        else:
            # --- 1. Call Analysis API ---
            with st.spinner("Analyzing data... (This may take a moment)"):
                try:
                    # Prepare the form data for the 'requests' library
                    # This is how you send multipart/form-data
                    form_data = {
                        "analysis_type": analysis_type,
                        "user_col": user_col,
                        "variation_col": variation_col,
                        "metric_col": metric_col,
                        "covariate_col": covariate_col,
                        "numerator_col": numerator_col,
                        "denominator_col": denominator_col,
                        "total_peeks": total_peeks
                    }
                    # Filter out None values
                    form_data = {k: v for k, v in form_data.items() if v is not None}
                    
                    # Prepare the file
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                    
                    response = requests.post(
                        f"{BACKEND_URL}/analyze-results",
                        data=form_data,
                        files=files
                    )
                    response.raise_for_status()
                    
                    st.session_state.analysis_results = response.json()
                    
                except requests.exceptions.RequestException as e:
                    # Handle API/connection errors
                    try:
                        # Try to get the error detail from FastAPI
                        detail = e.response.json().get('detail')
                        st.error(f"Error analyzing file: {detail}")
                    except:
                        st.error(f"Failed to connect to API: {e}")
                except Exception as e:
                    # Handle other errors (e.g., file processing)
                    st.error(f"An error occurred: {e}")

            # --- 2. Call Summary API (if analysis was successful) ---
            if st.session_state.analysis_results:
                with st.spinner("LLM Reporter Agent is writing summary..."):
                    try:
                        summary_response = requests.post(
                            f"{BACKEND_URL}/generate-summary",
                            json={"results": st.session_state.analysis_results}
                        )
                        summary_response.raise_for_status()
                        st.session_state.executive_summary = summary_response.json().get('summary')
                    except Exception as e:
                        st.session_state.executive_summary = "Failed to generate summary."


    # --- Display Results (Always runs, shows state) ---
    if st.session_state.analysis_results:
        st.divider()
        st.header("Analysis Results")
        
        # --- 1. Show Summary ---
        if st.session_state.executive_summary:
            st.subheader("🤖 Executive Summary")
            st.markdown(st.session_state.executive_summary)
        
        # --- 2. Show Key Metrics ---
        st.subheader("Key Metrics")
        results = st.session_state.analysis_results.get('results', {})
        cuped_info = st.session_state.analysis_results.get('cuped_adjustment', {})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Lift %",
            f"{results.get('lift_percent', 0):.2f}%"
        )
        col2.metric(
            "P-Value",
            f"{results.get('p_value', 0):.4f}"
        )
        col3.metric(
            "Stat-Sig?",
            "✅ Yes" if results.get('is_stat_sig') else "❌ No"
        )
        col4.metric(
            "CUPED Reduction",
            f"{cuped_info.get('variance_reduction_percent', 0):.2f}%"
        )
        
        # --- 3. Show Sequential Info (if applicable) ---
        if "sequential_test_info" in st.session_state.analysis_results:
            seq_info = st.session_state.analysis_results.get('sequential_test_info', {})
            st.info(f"""
            **Sequential Test (Bonferroni):** This p-value was judged against a corrected alpha of **{seq_info.get('adjusted_alpha'):.4f}** (based on {seq_info.get('total_peeks')} total planned peeks).
            """)
        
        # --- 4. Show Full JSON ---
        with st.expander("Show Full Statistical Results (JSON)"):
            st.json(st.session_state.analysis_results)