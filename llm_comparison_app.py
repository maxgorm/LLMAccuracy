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
    
    st.info("Upload one or more rent roll files (.xlsx or .pdf) to compare the outputs of two API calls with different LLM configurations.")
    
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
        
        # Display current selections
        st.write("**Current Configuration:**")
        st.write(f"First API Call: {llm1_master} (master) + {llm1_slave} (slave)")
        st.write(f"Second API Call: {llm2_master} (master) + {llm2_slave} (slave)")
    
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
                
                # Process the file
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # Step 1: Submit both jobs to the API simultaneously
                        st.write(f"Submitting {file_name} to API (Both Calls Simultaneously)...")
                        
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
                        
                        st.write(f"Both jobs submitted successfully. Job IDs: {job_id1}, {job_id2}")
                        
                        # Step 2: Monitor both jobs simultaneously
                        st.write(f"Monitoring both API calls for {file_name}...")
                        
                        # Create progress tracking for both jobs
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**First API Call Progress:**")
                            file_progress_bar1 = st.progress(0)
                            file_status_text1 = st.empty()
                        with col2:
                            st.write("**Second API Call Progress:**")
                            file_progress_bar2 = st.progress(0)
                            file_status_text2 = st.empty()
                        
                        # Set up retry parameters
                        max_retries = 50
                        retry_delay = 5
                        
                        # Track completion status
                        api_output1 = None
                        api_output2 = None
                        job1_complete = False
                        job2_complete = False
                        
                        # Monitor both jobs simultaneously
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
                            
                            # If both jobs are complete, break out of the loop
                            if job1_complete and job2_complete:
                                st.success(f"Both jobs completed successfully for {file_name}!")
                                break
                            
                            # Wait before next check
                            time.sleep(retry_delay)
                        else:
                            # Loop completed without both jobs finishing
                            if not job1_complete:
                                st.error(f"Maximum retries reached for first job on {file_name}. Job may still be processing.")
                            if not job2_complete:
                                st.error(f"Maximum retries reached for second job on {file_name}. Job may still be processing.")
                            
                            # Continue to next file if either job failed
                            if not (job1_complete and job2_complete):
                                continue
                        
                        # Verify we have both outputs
                        if not api_output1:
                            st.error(f"Failed to fetch first job results from API for {file_name}.")
                            continue
                        
                        if not api_output2:
                            st.error(f"Failed to fetch second job results from API for {file_name}.")
                            continue
                        
                        # Save both API outputs for reference
                        with open(output_api1_file, 'w') as f:
                            json.dump(api_output1, f, indent=2)
                        
                        with open(output_api2_file, 'w') as f:
                            json.dump(api_output2, f, indent=2)
                        
                        # Step 5: Compare API outputs
                        st.write(f"Comparing API outputs for {file_name}...")
                        accuracy, diffs, merged_df, field_stats = compare_api_outputs(api_output1, api_output2)
                        
                        if accuracy is not None:
                            # Save detailed differences
                            diff_output = {
                                "file": file_path,
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
                            with open(output_diff_file, 'w') as f:
                                json.dump(diff_output, f, indent=2, default=str)
                            
                            results.append({
                                "file": file_path,
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
                            st.error(f"Comparison failed for {file_name}. Check logs for details.")
                    
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                        st.text(traceback.format_exc())
                
                # Update progress
                progress_bar.progress((i + 1) / len(temp_paths))
            
            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Display results
            if results:
                with results_container:
                    st.subheader("Comparison Results")
                    
                    # Calculate average accuracy
                    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
                    st.metric("Average Accuracy Across All Files", f"{avg_accuracy:.2f}%")
                    
                    # Create tabs for each file
                    tabs = st.tabs([os.path.basename(r["file"]) for r in results])
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            st.metric("File Accuracy", f"{result['accuracy']:.2f}%")
                            
                            # Display LLM configuration used for this comparison
                            with st.expander("LLM Configuration Used"):
                                st.write("**First API Call:**")
                                st.write(f"• Primary LLM: {result['llm_config']['first_api_call']['master_llm']}")
                                st.write(f"• Secondary LLM: {result['llm_config']['first_api_call']['slave_llm']}")
                                st.write("**Second API Call:**")
                                st.write(f"• Primary LLM: {result['llm_config']['second_api_call']['master_llm']}")
                                st.write(f"• Secondary LLM: {result['llm_config']['second_api_call']['slave_llm']}")
                                st.write("**Other Settings:**")
                                st.write(f"• Max Batch Rows: {result['llm_config']['max_batch_rows']}")
                                st.write(f"• Bypass Rules-Based: {result['llm_config']['bypass_rb']}")
                            
                            # Display per-field accuracy
                            st.subheader("Per-Field Accuracy")
                            field_df = pd.DataFrame({
                                "Field": [field for field in result["field_stats"].keys()],
                                "Accuracy (%)": [stats["accuracy"] for stats in result["field_stats"].values()],
                                "Mismatches": [stats["mismatches"] for stats in result["field_stats"].values()],
                                "Total": [stats["total"] for stats in result["field_stats"].values()]
                            })
                            field_df = field_df.sort_values("Accuracy (%)")
                            st.dataframe(field_df)
                            
                            # Display mismatch examples
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
                            
                            # Provide download buttons
                            col1, col2, col3 = st.columns(3)
                            
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
