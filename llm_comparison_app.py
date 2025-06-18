import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback
import tempfile
import time
import copy

#This file is used to compare the outputs of 2 LLMs on a given RR using Thejan's API

# Import the API script functions
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    normalize_value,
    FIELDS_TO_COMPARE,
    API_BASE_URL,
    API_KEY
)

def compare_api_outputs_3way(api_output1, api_output2, api_output3):
    """Compares three API outputs and returns the differences with consensus analysis."""
    
    # Check if API outputs have the expected structure
    for i, api_output in enumerate([api_output1, api_output2, api_output3], 1):
        if not api_output or "df" not in api_output or not api_output["df"]:
            st.error(f"Error: API output {i} is empty or not in the expected format.")
            return None
    
    # Convert API outputs to DataFrames
    df_api1 = pd.DataFrame(api_output1["df"])
    df_api2 = pd.DataFrame(api_output2["df"])
    df_api3 = pd.DataFrame(api_output3["df"])
    
    # Normalize column names
    for df in [df_api1, df_api2, df_api3]:
        df.columns = df.columns.str.strip().str.lower()
    
    # Check for missing columns and add them
    for i, df in enumerate([df_api1, df_api2, df_api3], 1):
        missing_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df.columns)
        if missing_cols:
            st.warning(f"Warning: Missing expected columns in API output {i}: {missing_cols}")
            for col in missing_cols:
                df[col] = pd.NA
    
    # --- Data Type Conversion & Cleaning ---
    date_cols = ['move_in', 'lease_start', 'lease_end']
    numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
    
    for df in [df_api1, df_api2, df_api3]:
        # Convert date columns
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df[col] = pd.NA
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle boolean 'is_mtm'
        if 'is_mtm' in df.columns:
            df['is_mtm'] = df['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df['is_mtm'] = pd.to_numeric(df['is_mtm'], errors='coerce').fillna(0).astype(int)
    
    # --- Construct Match Key ---
    for df in [df_api1, df_api2, df_api3]:
        unit_col = df.get('unit_num', pd.Series(dtype=str))
        df['match_key'] = unit_col.fillna('').astype(str).str.strip().str.lower()
    
    # --- Three-way Merge ---
    # First merge API1 and API2
    merged_12 = df_api1.merge(df_api2, on="match_key", suffixes=("_api1", "_api2"), how="outer")
    # Then merge with API3
    merged_all = merged_12.merge(df_api3, on="match_key", suffixes=("", "_api3"), how="outer")
    
    # Handle suffix conflicts for API3 columns
    for col in FIELDS_TO_COMPARE:
        if col in merged_all.columns and f"{col}_api3" not in merged_all.columns:
            merged_all[f"{col}_api3"] = merged_all[col]
            if f"{col}_api1" in merged_all.columns:  # Only drop if we have the suffixed version
                merged_all = merged_all.drop(columns=[col])
    
    matched_rows = merged_all.dropna(subset=['match_key'])
    
    st.write(f"--- 3-Way Comparison Results ---")
    st.write(f"Total Rows Found: {len(matched_rows)}")
    
    # Initialize counters
    total_comparisons = 0
    perfect_agreement = 0  # All 3 match
    majority_agreement = 0  # 2 out of 3 match
    complete_disagreement = 0  # All 3 different
    
    # Track consensus patterns
    consensus_data = {}
    field_consensus_stats = {}
    api_outlier_count = {"api1": 0, "api2": 0, "api3": 0}
    
    for _, row in matched_rows.iterrows():
        for field in FIELDS_TO_COMPARE:
            field_api1 = f"{field}_api1"
            field_api2 = f"{field}_api2"
            field_api3 = f"{field}_api3"
            
            val_api1 = row.get(field_api1)
            val_api2 = row.get(field_api2)
            val_api3 = row.get(field_api3)
            
            # Normalize values for comparison
            norm_vals = []
            raw_vals = [val_api1, val_api2, val_api3]
            
            for val in raw_vals:
                if field == 'is_mtm':
                    norm_vals.append(int(val) if not pd.isna(val) else 0)
                elif field in numeric_cols:
                    if pd.isna(val):
                        norm_vals.append(0)  # Treat NaN as 0 for comparison
                    else:
                        norm_vals.append(float(val))
                else:
                    norm_vals.append(normalize_value(val))
            
            # Determine consensus pattern
            unique_vals = list(set(norm_vals))
            total_comparisons += 1
            
            if len(unique_vals) == 1:
                # Perfect agreement - all 3 match
                perfect_agreement += 1
                consensus_type = "perfect"
                outlier_api = None
            elif len(unique_vals) == 2:
                # Majority agreement - 2 match, 1 differs
                majority_agreement += 1
                consensus_type = "majority"
                # Find the outlier
                val_counts = {val: norm_vals.count(val) for val in unique_vals}
                outlier_val = min(val_counts.keys(), key=lambda x: val_counts[x])
                outlier_idx = norm_vals.index(outlier_val)
                outlier_api = f"api{outlier_idx + 1}"
                api_outlier_count[outlier_api] += 1
            else:
                # Complete disagreement - all 3 different
                complete_disagreement += 1
                consensus_type = "disagreement"
                outlier_api = None
            
            # Store detailed mismatch info
            if consensus_type != "perfect":
                if field not in consensus_data:
                    consensus_data[field] = []
                
                consensus_data[field].append({
                    "key": row["match_key"],
                    "api1_value": val_api1,
                    "api2_value": val_api2,
                    "api3_value": val_api3,
                    "consensus_type": consensus_type,
                    "outlier_api": outlier_api
                })
            
            # Track field-level consensus stats
            if field not in field_consensus_stats:
                field_consensus_stats[field] = {"perfect": 0, "majority": 0, "disagreement": 0, "total": 0}
            
            field_consensus_stats[field][consensus_type] += 1
            field_consensus_stats[field]["total"] += 1
    
    # Calculate accuracy metrics
    perfect_accuracy = round(100 * perfect_agreement / total_comparisons, 2) if total_comparisons > 0 else 0
    majority_accuracy = round(100 * (perfect_agreement + majority_agreement) / total_comparisons, 2) if total_comparisons > 0 else 0
    
    st.write(f"\n--- Consensus Analysis ---")
    st.write(f"Perfect Agreement (all 3 match): {perfect_accuracy}% ({perfect_agreement}/{total_comparisons} fields)")
    st.write(f"Majority Consensus (2+ match): {majority_accuracy}% ({perfect_agreement + majority_agreement}/{total_comparisons} fields)")
    st.write(f"Complete Disagreement: {round(100 * complete_disagreement / total_comparisons, 2)}% ({complete_disagreement}/{total_comparisons} fields)")
    
    st.write(f"\n--- Outlier Analysis ---")
    for api, count in api_outlier_count.items():
        percentage = round(100 * count / majority_agreement, 2) if majority_agreement > 0 else 0
        st.write(f"{api.upper()} was the outlier: {count} times ({percentage}% of majority cases)")
    
    return {
        "perfect_accuracy": perfect_accuracy,
        "majority_accuracy": majority_accuracy,
        "consensus_data": consensus_data,
        "field_consensus_stats": field_consensus_stats,
        "api_outlier_count": api_outlier_count,
        "merged_df": merged_all,
        "total_comparisons": total_comparisons,
        "perfect_agreement": perfect_agreement,
        "majority_agreement": majority_agreement,
        "complete_disagreement": complete_disagreement
    }

def compare_api_outputs(api_output1, api_output2):
    """Compares two API outputs and returns the differences."""
    
    # Check if API outputs have the expected structure
    if not api_output1 or "df" not in api_output1 or not api_output1["df"]:
        st.error("Error: First API output is empty or not in the expected format.")
        return None, None, None, None
    
    if not api_output2 or "df" not in api_output2 or not api_output2["df"]:
        st.error("Error: Second API output is empty or not in the expected format.")
        return None, None, None, None
    
    # Convert API outputs to DataFrames
    df_api1 = pd.DataFrame(api_output1["df"])
    df_api2 = pd.DataFrame(api_output2["df"])
    
    # Normalize column names
    df_api1.columns = df_api1.columns.str.strip().str.lower()
    df_api2.columns = df_api2.columns.str.strip().str.lower()
    
    # Check for missing columns
    missing_api1_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_api1.columns)
    missing_api2_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_api2.columns)
    
    if missing_api1_cols:
        st.warning(f"Warning: Missing expected columns in first API output: {missing_api1_cols}")
        # Add missing columns with NaN to allow merge/comparison
        for col in missing_api1_cols:
            df_api1[col] = pd.NA
    
    if missing_api2_cols:
        st.warning(f"Warning: Missing expected columns in second API output: {missing_api2_cols}")
        # Add missing columns with NaN to allow merge/comparison
        for col in missing_api2_cols:
            df_api2[col] = pd.NA
    
    # --- Data Type Conversion & Cleaning ---
    # Convert date columns safely, coercing errors to NaT
    date_cols = ['move_in', 'lease_start', 'lease_end']
    
    for col in date_cols:
        if col in df_api1.columns:
            # Handle 'nan' strings before conversion
            df_api1[col] = df_api1[col].replace('nan', pd.NA)
            df_api1[col] = pd.to_datetime(df_api1[col], errors='coerce').dt.strftime('%m/%d/%Y')
        else:
            df_api1[col] = pd.NA  # Ensure column exists if missing
        
        if col in df_api2.columns:
            # Handle 'nan' strings before conversion
            df_api2[col] = df_api2[col].replace('nan', pd.NA)
            df_api2[col] = pd.to_datetime(df_api2[col], errors='coerce').dt.strftime('%m/%d/%Y')
        else:
            df_api2[col] = pd.NA  # Ensure column exists if missing
    
    # Convert numeric columns safely, coercing errors to NaN
    numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
    for col in numeric_cols:
        if col in df_api1.columns:
            df_api1[col] = df_api1[col].replace('nan', pd.NA)
            df_api1[col] = pd.to_numeric(df_api1[col], errors='coerce')
        if col in df_api2.columns:
            df_api2[col] = df_api2[col].replace('nan', pd.NA)
            df_api2[col] = pd.to_numeric(df_api2[col], errors='coerce')
    
    # Handle boolean 'is_mtm' (normalize 0/1, True/False, 'true'/'false' to 0/1)
    if 'is_mtm' in df_api1.columns:
        df_api1['is_mtm'] = df_api1['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
        df_api1['is_mtm'] = pd.to_numeric(df_api1['is_mtm'], errors='coerce').fillna(0).astype(int)  # Default to 0 (False) if conversion fails or missing
    if 'is_mtm' in df_api2.columns:
        df_api2['is_mtm'] = df_api2['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
        df_api2['is_mtm'] = pd.to_numeric(df_api2['is_mtm'], errors='coerce').fillna(0).astype(int)
    
    # --- Construct Match Key ---
    # Use unit_num as the match key
    unit_col_1 = df_api1.get('unit_num', pd.Series(dtype=str))  # Get column or empty series
    df_api1['match_key'] = unit_col_1.fillna('').astype(str).str.strip().str.lower()
    
    unit_col_2 = df_api2.get('unit_num', pd.Series(dtype=str))
    df_api2['match_key'] = unit_col_2.fillna('').astype(str).str.strip().str.lower()
    
    # --- Merge and Compare ---
    merged = df_api1.merge(df_api2, on="match_key", suffixes=("_api1", "_api2"), how="outer", indicator=True)
    
    total_comparisons, correct_comparisons = 0, 0
    diffs = {}
    unmatched_api1 = merged[merged['_merge'] == 'left_only']
    unmatched_api2 = merged[merged['_merge'] == 'right_only']
    matched = merged[merged['_merge'] == 'both']
    
    st.write(f"--- Comparison Results ---")
    st.write(f"Matched Rows: {len(matched)}")
    st.write(f"Unmatched API1 Rows (in API1 output but not API2): {len(unmatched_api1)}")
    st.write(f"Unmatched API2 Rows (in API2 output but not API1): {len(unmatched_api2)}")
    
    for _, row in matched.iterrows():
        for field in FIELDS_TO_COMPARE:
            field_api1 = f"{field}_api1"
            field_api2 = f"{field}_api2"
            
            # Check if columns exist before trying to access
            val_api1 = row.get(field_api1)
            val_api2 = row.get(field_api2)
            
            # Normalize for comparison
            norm_api1 = normalize_value(val_api1)
            norm_api2 = normalize_value(val_api2)
            
            # Special handling for boolean 'is_mtm' (compare normalized 0/1)
            if field == 'is_mtm':
                # Already converted to 0/1 int above
                is_match = (int(val_api1) == int(val_api2))
            # Special handling for numeric fields (allow small tolerance?) - For now, exact match after normalization
            elif field in numeric_cols:
                # Handle potential NaN comparison carefully
                if pd.isna(val_api1) and pd.isna(val_api2):
                    is_match = True
                # Special case: treat 0 and nan as a match
                elif (pd.isna(val_api1) and val_api2 == 0) or (pd.isna(val_api2) and val_api1 == 0):
                    is_match = True
                elif pd.isna(val_api1) or pd.isna(val_api2):
                    is_match = False
                else:
                    # Could add tolerance here if needed: abs(val_api1 - val_api2) < tolerance
                    is_match = (float(val_api1) == float(val_api2))
            else:  # Default string comparison
                is_match = (norm_api1 == norm_api2)
            
            total_comparisons += 1
            if is_match:
                correct_comparisons += 1
            else:
                diffs.setdefault(field, []).append({
                    "key": row["match_key"],
                    "api1_value": val_api1,
                    "api2_value": val_api2
                })
    
    accuracy = round(100 * correct_comparisons / total_comparisons, 2) if total_comparisons > 0 else 0
    st.write(f"\nOverall Accuracy (on matched rows): {accuracy}% ({correct_comparisons}/{total_comparisons} fields)")
    
    # Calculate per-field accuracy
    field_stats = {}
    for field in FIELDS_TO_COMPARE:
        field_total = len(matched)  # Total possible comparisons for this field
        field_mismatches = len(diffs.get(field, []))
        field_correct = field_total - field_mismatches
        field_accuracy = round(100 * field_correct / field_total, 2) if field_total > 0 else 0
        field_stats[field] = {
            "accuracy": field_accuracy,
            "correct": field_correct,
            "total": field_total,
            "mismatches": field_mismatches
        }
    
    return accuracy, diffs, merged, field_stats

def run_streamlit_app():
    """Main Streamlit app function for LLM comparison"""
    st.set_page_config(layout="wide")
    st.title("Rent Roll LLM Comparison Tool")
    
    # Comparison Mode Selection
    st.subheader("Comparison Mode")
    comparison_mode = st.radio(
        "Select comparison mode:",
        ["2-Way Comparison", "3-Way Comparison"],
        help="Choose between comparing 2 or 3 different LLM configurations"
    )
    
    if comparison_mode == "2-Way Comparison":
        st.info("Upload one or more rent roll files (.xlsx or .pdf) to compare the outputs of two API calls with different LLM configurations.")
    else:
        st.info("Upload one or more rent roll files (.xlsx or .pdf) to compare the outputs of three API calls with different LLM configurations. This will show consensus analysis and identify outliers.")
    
    # API Configuration
    with st.expander("API Configuration"):
        # Import at the beginning of the function to get the current values
        import api_rent_roll_verifier
        api_base_url = st.text_input("API Base URL", value=api_rent_roll_verifier.API_BASE_URL)
        api_key = st.text_input("API Key (optional)", type="password", value=api_rent_roll_verifier.API_KEY)
        
        if st.button("Save API Configuration"):
            # Update the module variables directly
            api_rent_roll_verifier.API_BASE_URL = api_base_url
            api_rent_roll_verifier.API_KEY = api_key if api_key else None
            st.success("API configuration saved!")
    
    # Define supported models
    anthropic_models = [
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'claude-3-7-sonnet-latest',
        'claude-opus-4-20250514',
        'claude-sonnet-4-20250514'
    ]
    openai_models = [
        'gpt-4o-mini',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano'
    ]
    mistral_models = [
        'mistral-small-latest',
        'mistral-medium-2505',
        'magistral-medium-2506'
    ]
    google_models = [
        'gemini-2.0-flash',
        'gemini-2.5-flash-preview-05-20',
        'gemini-2.5-pro-preview-06-05'
    ]
    
    # Combine all models for selection
    all_models = anthropic_models + openai_models + mistral_models + google_models
    
    # Processing options
    with st.expander("LLM Configuration"):
        bypass_rb = st.checkbox("Bypass Rules-Based Approach", value=True, 
                               help="When checked, the API will always use LLMs instead of the rules-based approach")
        
        max_batch_rows = st.number_input("Max Batch Rows", 
                                        min_value=1, 
                                        max_value=200, 
                                        value=50,
                                        help="Maximum number of rows to process in each batch")
        
        st.info("Configure the LLM models for comparison. The first API call will use the first set of models, and the second API call will use the second set.")
        
        # First API call LLM configuration
        st.subheader("First API Call")
        col1, col2 = st.columns(2)
        with col1:
            llm1_master = st.selectbox("Primary LLM (First API Call)", 
                                      all_models, 
                                      index=all_models.index('claude-sonnet-4-20250514'),
                                      help="Primary LLM for processing")
        with col2:
            llm1_slave = st.selectbox("Secondary LLM (First API Call)", 
                                     all_models, 
                                     index=all_models.index('gemini-2.5-flash-preview-05-20'),
                                     help="Secondary LLM for verification")
        
        # Second API call LLM configuration
        st.subheader("Second API Call")
        col1, col2 = st.columns(2)
        with col1:
            llm2_master = st.selectbox("Primary LLM (Second API Call)", 
                                      all_models, 
                                      index=all_models.index('claude-3-7-sonnet-latest'),
                                      help="Primary LLM for processing")
        with col2:
            llm2_slave = st.selectbox("Secondary LLM (Second API Call)", 
                                     all_models, 
                                     index=all_models.index('gemini-2.0-flash'),
                                     help="Secondary LLM for verification")
        
        # Third API call LLM configuration (only show for 3-way comparison)
        if comparison_mode == "3-Way Comparison":
            st.subheader("Third API Call")
            col1, col2 = st.columns(2)
            with col1:
                llm3_master = st.selectbox("Primary LLM (Third API Call)", 
                                          all_models, 
                                          index=all_models.index('gpt-4o-mini'),
                                          help="Primary LLM for processing")
            with col2:
                llm3_slave = st.selectbox("Secondary LLM (Third API Call)", 
                                         all_models, 
                                         index=all_models.index('mistral-small-latest'),
                                         help="Secondary LLM for verification")
        
        # Display current selections
        st.write("**Current Configuration:**")
        st.write(f"First API Call: {llm1_master} (master) + {llm1_slave} (slave)")
        st.write(f"Second API Call: {llm2_master} (master) + {llm2_slave} (slave)")
        if comparison_mode == "3-Way Comparison":
            st.write(f"Third API Call: {llm3_master} (master) + {llm3_slave} (slave)")
    
    # Show model categories for reference (outside the LLM Configuration expander)
    with st.expander("Available Models by Provider"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Anthropic Models:**")
            for model in anthropic_models:
                st.write(f"• {model}")
        with col2:
            st.write("**OpenAI Models:**")
            for model in openai_models:
                st.write(f"• {model}")
        with col3:
            st.write("**Mistral Models:**")
            for model in mistral_models:
                st.write(f"• {model}")
        with col4:
            st.write("**Google Models:**")
            for model in google_models:
                st.write(f"• {model}")
    
    # File upload
    st.subheader("Upload Rent Roll Files")
    uploaded_files = st.file_uploader(
        "Upload Rent Roll Files (.xlsx or .pdf)", 
        type=["xlsx", "pdf"],
        accept_multiple_files=True
    )
    
    # Optional sheet name selection
    sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
    st.info("Note: Sheet name will be ignored for PDF files.")
    
    # Display file counts
    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_type = "PDF" if file_ext == '.pdf' else "Excel"
            st.write(f"- {file.name} ({file_type})")
        
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        
        # Save all uploaded files to the temp directory
        temp_paths = []
        
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(file_path)
        
        # Create output directory for results
        output_dir = os.path.join(temp_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        if st.button("Run Comparison"):
            # Process files with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a placeholder for results
            results_container = st.container()
            
            # Process each file sequentially
            results = []
            for i, file_path in enumerate(temp_paths):
                file_name = os.path.basename(file_path)
                status_text.text(f"Processing file {i+1}/{len(temp_paths)}: {file_name}")
                progress_bar.progress((i) / len(temp_paths))
                
                # Create output files
                output_api1_file = os.path.join(output_dir, f"api1_output_{file_name}.json")
                output_api2_file = os.path.join(output_dir, f"api2_output_{file_name}.json")
                output_diff_file = os.path.join(output_dir, f"comparison_diff_{file_name}.json")
                if comparison_mode == "3-Way Comparison":
                    output_api3_file = os.path.join(output_dir, f"api3_output_{file_name}.json")
                
                # Process the file
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # Step 1: Submit jobs to the API simultaneously
                        if comparison_mode == "2-Way Comparison":
                            st.write(f"Submitting {file_name} to API (Both Calls Simultaneously)...")
                        else:
                            st.write(f"Submitting {file_name} to API (All Three Calls Simultaneously)...")
                        
                        # Submit first job
                        job_id1 = submit_job(file_path, 
                                           token="", 
                                           doc_type="rent_roll", 
                                           sheet_name=sheet_name, 
                                           bypass_rb=bypass_rb,
                                           max_n_batch_rows=max_batch_rows,
                                           rr_master_llm=llm1_master,
                                           rr_slave_llm=llm1_slave)
                        
                        if not job_id1:
                            st.error(f"Failed to submit first job to API for {file_name}.")
                            continue
                        
                        # Submit second job immediately after first
                        job_id2 = submit_job(file_path, 
                                           token="", 
                                           doc_type="rent_roll", 
                                           sheet_name=sheet_name, 
                                           bypass_rb=bypass_rb,
                                           max_n_batch_rows=max_batch_rows,
                                           rr_master_llm=llm2_master,
                                           rr_slave_llm=llm2_slave)
                        
                        if not job_id2:
                            st.error(f"Failed to submit second job to API for {file_name}.")
                            continue
                        
                        # Submit third job if 3-way comparison
                        job_id3 = None
                        if comparison_mode == "3-Way Comparison":
                            job_id3 = submit_job(file_path, 
                                               token="", 
                                               doc_type="rent_roll", 
                                               sheet_name=sheet_name, 
                                               bypass_rb=bypass_rb,
                                               max_n_batch_rows=max_batch_rows,
                                               rr_master_llm=llm3_master,
                                               rr_slave_llm=llm3_slave)
                            
                            if not job_id3:
                                st.error(f"Failed to submit third job to API for {file_name}.")
                                continue
                        
                        if comparison_mode == "2-Way Comparison":
                            st.write(f"Both jobs submitted successfully. Job IDs: {job_id1}, {job_id2}")
                        else:
                            st.write(f"All three jobs submitted successfully. Job IDs: {job_id1}, {job_id2}, {job_id3}")
                        
                        # Step 2: Monitor jobs simultaneously
                        if comparison_mode == "2-Way Comparison":
                            st.write(f"Monitoring both API calls for {file_name}...")
                        else:
                            st.write(f"Monitoring all three API calls for {file_name}...")
                        
                        # Create progress tracking for jobs
                        if comparison_mode == "2-Way Comparison":
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**First API Call Progress:**")
                                file_progress_bar1 = st.progress(0)
                                file_status_text1 = st.empty()
                            with col2:
                                st.write("**Second API Call Progress:**")
                                file_progress_bar2 = st.progress(0)
                                file_status_text2 = st.empty()
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**First API Call Progress:**")
                                file_progress_bar1 = st.progress(0)
                                file_status_text1 = st.empty()
                            with col2:
                                st.write("**Second API Call Progress:**")
                                file_progress_bar2 = st.progress(0)
                                file_status_text2 = st.empty()
                            with col3:
                                st.write("**Third API Call Progress:**")
                                file_progress_bar3 = st.progress(0)
                                file_status_text3 = st.empty()
                        
                        # Set up retry parameters
                        max_retries = 50
                        retry_delay = 10
                        
                        # Track completion status
                        api_output1 = None
                        api_output2 = None
                        api_output3 = None
                        job1_complete = False
                        job2_complete = False
                        job3_complete = True if comparison_mode == "2-Way Comparison" else False
                        
                        # Monitor jobs simultaneously
                        for retry in range(max_retries):
                            # Check first job if not complete
                            if not job1_complete:
                                file_status_text1.text(f"Attempt {retry + 1}/{max_retries}: Checking first job status...")
                                file_progress_bar1.progress((retry + 1) / max_retries)
                                
                                temp_output1 = fetch_job_results(job_id1, max_retries=1, retry_delay=0)
                                
                                if temp_output1 is not None:
                                    if temp_output1.get('job', {}).get('status') != 'processing':
                                        # Job is complete
                                        api_output1 = temp_output1
                                        job1_complete = True
                                        file_status_text1.text(f"First job completed successfully!")
                                        file_progress_bar1.progress(1.0)
                                    else:
                                        file_status_text1.text(f"First job still processing...")
                                else:
                                    file_status_text1.text(f"First job status check failed, retrying...")
                            
                            # Check second job if not complete
                            if not job2_complete:
                                file_status_text2.text(f"Attempt {retry + 1}/{max_retries}: Checking second job status...")
                                file_progress_bar2.progress((retry + 1) / max_retries)
                                
                                temp_output2 = fetch_job_results(job_id2, max_retries=1, retry_delay=0)
                                
                                if temp_output2 is not None:
                                    if temp_output2.get('job', {}).get('status') != 'processing':
                                        # Job is complete
                                        api_output2 = temp_output2
                                        job2_complete = True
                                        file_status_text2.text(f"Second job completed successfully!")
                                        file_progress_bar2.progress(1.0)
                                    else:
                                        file_status_text2.text(f"Second job still processing...")
                                else:
                                    file_status_text2.text(f"Second job status check failed, retrying...")
                            
                            # Check third job if not complete (only for 3-way comparison)
                            if not job3_complete and comparison_mode == "3-Way Comparison":
                                file_status_text3.text(f"Attempt {retry + 1}/{max_retries}: Checking third job status...")
                                file_progress_bar3.progress((retry + 1) / max_retries)
                                
                                temp_output3 = fetch_job_results(job_id3, max_retries=1, retry_delay=0)
                                
                                if temp_output3 is not None:
                                    if temp_output3.get('job', {}).get('status') != 'processing':
                                        # Job is complete
                                        api_output3 = temp_output3
                                        job3_complete = True
                                        file_status_text3.text(f"Third job completed successfully!")
                                        file_progress_bar3.progress(1.0)
                                    else:
                                        file_status_text3.text(f"Third job still processing...")
                                else:
                                    file_status_text3.text(f"Third job status check failed, retrying...")
                            
                            # Check if all jobs are complete
                            if job1_complete and job2_complete and job3_complete:
                                if comparison_mode == "2-Way Comparison":
                                    st.success(f"Both jobs completed successfully for {file_name}!")
                                else:
                                    st.success(f"All three jobs completed successfully for {file_name}!")
                                break
                            
                            # Wait before next check
                            time.sleep(retry_delay)
                        else:
                            # Loop completed without all jobs finishing
                            if not job1_complete:
                                st.error(f"Maximum retries reached for first job on {file_name}. Job may still be processing.")
                            if not job2_complete:
                                st.error(f"Maximum retries reached for second job on {file_name}. Job may still be processing.")
                            if not job3_complete and comparison_mode == "3-Way Comparison":
                                st.error(f"Maximum retries reached for third job on {file_name}. Job may still be processing.")
                            
                            # Continue to next file if any job failed
                            if not (job1_complete and job2_complete and job3_complete):
                                continue
                        
                        # Verify we have all required outputs
                        if not api_output1:
                            st.error(f"Failed to fetch first job results from API for {file_name}.")
                            continue
                        
                        if not api_output2:
                            st.error(f"Failed to fetch second job results from API for {file_name}.")
                            continue
                        
                        if comparison_mode == "3-Way Comparison" and not api_output3:
                            st.error(f"Failed to fetch third job results from API for {file_name}.")
                            continue
                        
                        # Save API outputs for reference
                        with open(output_api1_file, 'w') as f:
                            json.dump(api_output1, f, indent=2)
                        
                        with open(output_api2_file, 'w') as f:
                            json.dump(api_output2, f, indent=2)
                        
                        if comparison_mode == "3-Way Comparison":
                            with open(output_api3_file, 'w') as f:
                                json.dump(api_output3, f, indent=2)
                        
                        # Step 5: Compare API outputs
                        st.write(f"Comparing API outputs for {file_name}...")
                        
                        # Debug: Check API output structures
                        st.write("**Debug Information:**")
                        for i, api_output in enumerate([api_output1, api_output2, api_output3 if comparison_mode == "3-Way Comparison" else None], 1):
                            if api_output is not None:
                                if isinstance(api_output, dict):
                                    has_df = "df" in api_output
                                    df_content = api_output.get("df", "No 'df' key")
                                    if has_df and api_output["df"]:
                                        df_len = len(api_output["df"]) if isinstance(api_output["df"], list) else "Not a list"
                                        st.write(f"API Output {i}: Has 'df' key: {has_df}, df length: {df_len}")
                                    else:
                                        st.write(f"API Output {i}: Has 'df' key: {has_df}, df content: {type(df_content)}")
                                        if has_df:
                                            st.write(f"API Output {i} df value: {df_content}")
                                else:
                                    st.write(f"API Output {i}: Not a dictionary, type: {type(api_output)}")
                                
                                # Show the keys in the API output
                                if isinstance(api_output, dict):
                                    st.write(f"API Output {i} keys: {list(api_output.keys())}")
                            else:
                                st.write(f"API Output {i}: None")
                        
                        if comparison_mode == "2-Way Comparison":
                            comparison_result = compare_api_outputs(api_output1, api_output2)
                            accuracy, diffs, merged_df, field_stats = comparison_result
                            
                            if accuracy is not None:
                                # Save detailed differences for 2-way
                                diff_output = {
                                    "file": file_path,
                                    "comparison_mode": "2-way",
                                    "accuracy_percent": accuracy,
                                    "field_mismatches": diffs,
                                    "llm_configuration": {
                                        "first_api_call": {
                                            "master_llm": llm1_master,
                                            "slave_llm": llm1_slave
                                        },
                                        "second_api_call": {
                                            "master_llm": llm2_master,
                                            "slave_llm": llm2_slave
                                        },
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                }
                                
                                results.append({
                                    "file": file_path,
                                    "comparison_mode": "2-way",
                                    "accuracy": accuracy,
                                    "diffs": diffs,
                                    "merged_df": merged_df,
                                    "field_stats": field_stats,
                                    "api_output1": api_output1,
                                    "api_output2": api_output2,
                                    "llm_config": {
                                        "first_api_call": {
                                            "master_llm": llm1_master,
                                            "slave_llm": llm1_slave
                                        },
                                        "second_api_call": {
                                            "master_llm": llm2_master,
                                            "slave_llm": llm2_slave
                                        },
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                })
                        else:
                            comparison_result = compare_api_outputs_3way(api_output1, api_output2, api_output3)
                            
                            if comparison_result is not None:
                                # Save detailed differences for 3-way
                                diff_output = {
                                    "file": file_path,
                                    "comparison_mode": "3-way",
                                    "perfect_accuracy_percent": comparison_result["perfect_accuracy"],
                                    "majority_accuracy_percent": comparison_result["majority_accuracy"],
                                    "consensus_data": comparison_result["consensus_data"],
                                    "field_consensus_stats": comparison_result["field_consensus_stats"],
                                    "api_outlier_count": comparison_result["api_outlier_count"],
                                    "llm_configuration": {
                                        "first_api_call": {
                                            "master_llm": llm1_master,
                                            "slave_llm": llm1_slave
                                        },
                                        "second_api_call": {
                                            "master_llm": llm2_master,
                                            "slave_llm": llm2_slave
                                        },
                                        "third_api_call": {
                                            "master_llm": llm3_master,
                                            "slave_llm": llm3_slave
                                        },
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                }
                                
                                results.append({
                                    "file": file_path,
                                    "comparison_mode": "3-way",
                                    "perfect_accuracy": comparison_result["perfect_accuracy"],
                                    "majority_accuracy": comparison_result["majority_accuracy"],
                                    "consensus_data": comparison_result["consensus_data"],
                                    "field_consensus_stats": comparison_result["field_consensus_stats"],
                                    "api_outlier_count": comparison_result["api_outlier_count"],
                                    "merged_df": comparison_result["merged_df"],
                                    "api_output1": api_output1,
                                    "api_output2": api_output2,
                                    "api_output3": api_output3,
                                    "llm_config": {
                                        "first_api_call": {
                                            "master_llm": llm1_master,
                                            "slave_llm": llm1_slave
                                        },
                                        "second_api_call": {
                                            "master_llm": llm2_master,
                                            "slave_llm": llm2_slave
                                        },
                                        "third_api_call": {
                                            "master_llm": llm3_master,
                                            "slave_llm": llm3_slave
                                        },
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                })
                            else:
                                st.error(f"3-way comparison failed for {file_name}. Check logs for details.")
                                continue
                        
                        # Save the diff output
                        with open(output_diff_file, 'w') as f:
                            json.dump(diff_output, f, indent=2, default=str)
                    
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                        st.text(traceback.format_exc())
                
                # Update progress
                progress_value = min(1.0, (i + 1) / len(temp_paths))
                progress_bar.progress(progress_value)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Display results
            if results:
                with results_container:
                    st.subheader("Comparison Results")
                    
                    # Calculate average accuracy based on comparison mode
                    if results[0]["comparison_mode"] == "2-way":
                        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
                        st.metric("Average Accuracy Across All Files", f"{avg_accuracy:.2f}%")
                    else:
                        avg_perfect = sum(r["perfect_accuracy"] for r in results) / len(results)
                        avg_majority = sum(r["majority_accuracy"] for r in results) / len(results)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Perfect Agreement", f"{avg_perfect:.2f}%")
                        with col2:
                            st.metric("Average Majority Consensus", f"{avg_majority:.2f}%")
                    
                    # Create tabs for each file
                    tabs = st.tabs([os.path.basename(r["file"]) for r in results])
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            if result["comparison_mode"] == "2-way":
                                st.metric("File Accuracy", f"{result['accuracy']:.2f}%")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Perfect Agreement", f"{result['perfect_accuracy']:.2f}%")
                                with col2:
                                    st.metric("Majority Consensus", f"{result['majority_accuracy']:.2f}%")
                            
                            # Display LLM configuration used for this comparison
                            with st.expander("LLM Configuration Used"):
                                st.write("**First API Call:**")
                                st.write(f"• Primary LLM: {result['llm_config']['first_api_call']['master_llm']}")
                                st.write(f"• Secondary LLM: {result['llm_config']['first_api_call']['slave_llm']}")
                                st.write("**Second API Call:**")
                                st.write(f"• Primary LLM: {result['llm_config']['second_api_call']['master_llm']}")
                                st.write(f"• Secondary LLM: {result['llm_config']['second_api_call']['slave_llm']}")
                                if result["comparison_mode"] == "3-way":
                                    st.write("**Third API Call:**")
                                    st.write(f"• Primary LLM: {result['llm_config']['third_api_call']['master_llm']}")
                                    st.write(f"• Secondary LLM: {result['llm_config']['third_api_call']['slave_llm']}")
                                st.write("**Other Settings:**")
                                st.write(f"• Max Batch Rows: {result['llm_config']['max_batch_rows']}")
                                st.write(f"• Bypass Rules-Based: {result['llm_config']['bypass_rb']}")
                            
                            if result["comparison_mode"] == "2-way":
                                # Display per-field accuracy for 2-way
                                st.subheader("Per-Field Accuracy")
                                field_df = pd.DataFrame({
                                    "Field": [field for field in result["field_stats"].keys()],
                                    "Accuracy (%)": [stats["accuracy"] for stats in result["field_stats"].values()],
                                    "Mismatches": [stats["mismatches"] for stats in result["field_stats"].values()],
                                    "Total": [stats["total"] for stats in result["field_stats"].values()]
                                })
                                field_df = field_df.sort_values("Accuracy (%)")
                                st.dataframe(field_df)
                                
                                # Display mismatch examples for 2-way
                                st.subheader("Field Mismatches")
                                all_mismatches = []
                                for field, mismatches in result["diffs"].items():
                                    for mismatch in mismatches:
                                        all_mismatches.append({
                                            "Field": field,
                                            "Unit": mismatch['key'],
                                            "API Call 1 Value": mismatch['api1_value'],
                                            "API Call 2 Value": mismatch['api2_value']
                                        })
                                
                                if all_mismatches:
                                    mismatches_df = pd.DataFrame(all_mismatches)
                                    st.dataframe(mismatches_df)
                                    
                                    # Text list of differences
                                    st.subheader("Text List of Differences")
                                    for mismatch in all_mismatches:
                                        st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, API Call 1: {mismatch['API Call 1 Value']}, API Call 2: {mismatch['API Call 2 Value']}")
                                else:
                                    st.write("No differences found between API calls.")
                            
                            else:
                                # Display 3-way consensus analysis
                                st.subheader("Consensus Analysis")
                                
                                # Outlier frequency
                                st.subheader("Outlier Frequency")
                                outlier_df = pd.DataFrame({
                                    "API": ["API 1", "API 2", "API 3"],
                                    "Times as Outlier": [
                                        result["api_outlier_count"]["api1"],
                                        result["api_outlier_count"]["api2"],
                                        result["api_outlier_count"]["api3"]
                                    ]
                                })
                                st.dataframe(outlier_df)
                                
                                # Per-field consensus stats
                                st.subheader("Per-Field Consensus")
                                field_consensus_df = pd.DataFrame({
                                    "Field": [field for field in result["field_consensus_stats"].keys()],
                                    "Perfect Agreement": [stats["perfect"] for stats in result["field_consensus_stats"].values()],
                                    "Majority Agreement": [stats["majority"] for stats in result["field_consensus_stats"].values()],
                                    "Complete Disagreement": [stats["disagreement"] for stats in result["field_consensus_stats"].values()],
                                    "Total": [stats["total"] for stats in result["field_consensus_stats"].values()]
                                })
                                st.dataframe(field_consensus_df)
                                
                                # Display consensus mismatches
                                st.subheader("Consensus Mismatches")
                                all_consensus_mismatches = []
                                for field, mismatches in result["consensus_data"].items():
                                    for mismatch in mismatches:
                                        consensus_type_display = {
                                            "majority": "Majority (2 agree, 1 differs)",
                                            "disagreement": "Complete Disagreement"
                                        }.get(mismatch['consensus_type'], mismatch['consensus_type'])
                                        
                                        mismatch_row = {
                                            "Field": field,
                                            "Unit": mismatch['key'],
                                            "API Call 1 Value": mismatch['api1_value'],
                                            "API Call 2 Value": mismatch['api2_value'],
                                            "API Call 3 Value": mismatch['api3_value'],
                                            "Consensus Type": consensus_type_display
                                        }
                                        
                                        if mismatch['outlier_api']:
                                            mismatch_row["Outlier"] = mismatch['outlier_api'].upper()
                                        else:
                                            mismatch_row["Outlier"] = "N/A"
                                        
                                        all_consensus_mismatches.append(mismatch_row)
                                
                                if all_consensus_mismatches:
                                    consensus_df = pd.DataFrame(all_consensus_mismatches)
                                    st.dataframe(consensus_df)
                                    
                                    # Text list of consensus differences
                                    st.subheader("Text List of Consensus Differences")
                                    for mismatch in all_consensus_mismatches:
                                        outlier_text = f" (Outlier: {mismatch['Outlier']})" if mismatch['Outlier'] != "N/A" else ""
                                        st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, API1: {mismatch['API Call 1 Value']}, API2: {mismatch['API Call 2 Value']}, API3: {mismatch['API Call 3 Value']} - {mismatch['Consensus Type']}{outlier_text}")
                                else:
                                    st.write("Perfect consensus across all API calls!")
                            
                            # Provide download buttons
                            if result["comparison_mode"] == "2-way":
                                col1, col2, col3 = st.columns(3)
                            else:
                                col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                # Download for diff details
                                st.download_button(
                                    label="Download Comparison Details (JSON)",
                                    data=json.dumps(diff_output, indent=2, default=str),
                                    file_name=f"comparison_diff_{os.path.basename(result['file'])}.json",
                                    mime="application/json",
                                    key=f"download_diff_{i}"
                                )
                            
                            with col2:
                                # Download for first API output
                                st.download_button(
                                    label="Download First API Output (JSON)",
                                    data=json.dumps(result["api_output1"], indent=2, default=str),
                                    file_name=f"api1_output_{os.path.basename(result['file'])}.json",
                                    mime="application/json",
                                    key=f"download_api1_{i}"
                                )
                            
                            with col3:
                                # Download for second API output
                                st.download_button(
                                    label="Download Second API Output (JSON)",
                                    data=json.dumps(result["api_output2"], indent=2, default=str),
                                    file_name=f"api2_output_{os.path.basename(result['file'])}.json",
                                    mime="application/json",
                                    key=f"download_api2_{i}"
                                )
                            
                            if result["comparison_mode"] == "3-way":
                                with col4:
                                    # Download for third API output
                                    st.download_button(
                                        label="Download Third API Output (JSON)",
                                        data=json.dumps(result["api_output3"], indent=2, default=str),
                                        file_name=f"api3_output_{os.path.basename(result['file'])}.json",
                                        mime="application/json",
                                        key=f"download_api3_{i}"
                                    )
                            
                            # Display side-by-side comparison (optional, can be large)
                            if result["merged_df"] is not None and st.checkbox("Show Detailed Comparison Table", key=f"show_table_{i}"):
                                st.dataframe(result["merged_df"])
            else:
                st.error("No successful comparisons were completed. Check the logs for errors.")
            
            # Clean up temp files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except OSError as e:
                st.warning(f"Could not remove temporary files: {e}")

if __name__ == "__main__":
    run_streamlit_app()
