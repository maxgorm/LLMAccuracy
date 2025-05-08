import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback
import tempfile
import time
import zipfile
import io
from typing import Dict, List, Any, Optional, Tuple

# For doing what Thejan asked:
# Comparing outputs between AI and Rules-Based API responses
# generating a detailed report of differences
# and providing a .zip file of .csv's of the JSON outputs

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'ai_output_path' not in st.session_state:
    st.session_state.ai_output_path = None
if 'rb_output_path' not in st.session_state:
    st.session_state.rb_output_path = None
if 'comparison_path' not in st.session_state:
    st.session_state.comparison_path = None
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None

# Import necessary functions from api_rent_roll_verifier
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    normalize_value,
    FIELDS_TO_COMPARE,
    API_BASE_URL,
    API_KEY
)

# Define folder names
AI_OUTPUT_FOLDER = "AI_OUTPUT"
RB_OUTPUT_FOLDER = "RB_OUTPUT"
COMPARISON_FOLDER = "COMPARISON"

def json_to_csv_string(json_data):
    """Converts the 'df' part of the API JSON output to a CSV string."""
    if not json_data or "df" not in json_data or not json_data["df"]:
        return None
    
    try:
        df = pd.DataFrame(json_data["df"])
        # Convert DataFrame to CSV string
        csv_string = df.to_csv(index=False)
        return csv_string
    except Exception as e:
        st.error(f"Error converting JSON to CSV: {e}")
        return None

def compare_csv_data(ai_csv_path, rb_csv_path):
    """Compares two CSV files and returns the differences and accuracy."""
    try:
        df_ai = pd.read_csv(ai_csv_path)
        df_rb = pd.read_csv(rb_csv_path)

        # Print the number of rows in each dataframe for debugging
        st.write(f"AI CSV rows: {len(df_ai)}, RB CSV rows: {len(df_rb)}")

        # Normalize column names
        df_ai.columns = df_ai.columns.str.strip().str.lower()
        df_rb.columns = df_rb.columns.str.strip().str.lower()

        # Ensure required columns exist, handling potential case differences
        required_cols = {f.lower() for f in FIELDS_TO_COMPARE}
        missing_ai_cols = required_cols - set(df_ai.columns)
        missing_rb_cols = required_cols - set(df_rb.columns)

        if missing_ai_cols:
            st.warning(f"Warning: Missing expected columns in AI output CSV: {missing_ai_cols}")
            for col in missing_ai_cols:
                df_ai[col] = pd.NA

        if missing_rb_cols:
            st.warning(f"Warning: Missing expected columns in RB output CSV: {missing_rb_cols}")
            for col in missing_rb_cols:
                df_rb[col] = pd.NA

        # --- Data Type Conversion & Cleaning ---
        # Convert date columns safely, coercing errors to NaT
        date_cols = ['move_in', 'lease_start', 'lease_end']

        for col in date_cols:
            if col in df_ai.columns:
                df_ai[col] = df_ai[col].replace('nan', pd.NA)
                df_ai[col] = pd.to_datetime(df_ai[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df_ai[col] = pd.NA

            if col in df_rb.columns:
                df_rb[col] = df_rb[col].replace('nan', pd.NA)
                df_rb[col] = pd.to_datetime(df_rb[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df_rb[col] = pd.NA

        # Convert numeric columns safely, coercing errors to NaN
        numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
        for col in numeric_cols:
            if col in df_ai.columns:
                df_ai[col] = df_ai[col].replace('nan', pd.NA)
                df_ai[col] = pd.to_numeric(df_ai[col], errors='coerce')
            if col in df_rb.columns:
                df_rb[col] = df_rb[col].replace('nan', pd.NA)
                df_rb[col] = pd.to_numeric(df_rb[col], errors='coerce')

        # Handle boolean 'is_mtm' (normalize 0/1, True/False, 'true'/'false' to 0/1)
        if 'is_mtm' in df_ai.columns:
            df_ai['is_mtm'] = df_ai['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_ai['is_mtm'] = pd.to_numeric(df_ai['is_mtm'], errors='coerce').fillna(0).astype(int)
        if 'is_mtm' in df_rb.columns:
            df_rb['is_mtm'] = df_rb['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_rb['is_mtm'] = pd.to_numeric(df_rb['is_mtm'], errors='coerce').fillna(0).astype(int)

        # --- Construct Match Key ---
        # Use unit_num as the match key, with additional cleaning
        unit_col_ai = df_ai.get('unit_num', pd.Series(dtype=str))
        # Clean unit numbers: remove spaces, dashes, and other non-alphanumeric characters
        df_ai['match_key'] = unit_col_ai.fillna('').astype(str).str.strip().str.lower()
        df_ai['match_key'] = df_ai['match_key'].str.replace(r'[^a-z0-9]', '', regex=True)

        unit_col_rb = df_rb.get('unit_num', pd.Series(dtype=str))
        # Apply the same cleaning to RB unit numbers
        df_rb['match_key'] = unit_col_rb.fillna('').astype(str).str.strip().str.lower()
        df_rb['match_key'] = df_rb['match_key'].str.replace(r'[^a-z0-9]', '', regex=True)

        # If there are no matches using the cleaned unit numbers, try using row position
        # --- Merge and Compare ---
        merged = df_ai.merge(df_rb, on="match_key", suffixes=("_ai", "_rb"), how="outer", indicator=True)
        matched = merged[merged['_merge'] == 'both']
        
        # If there are very few matches (less than 10% of the rows), try matching by row position instead
        if len(matched) < 0.1 * min(len(df_ai), len(df_rb)):
            st.warning("Very few matches found using unit numbers. Trying to match by row position instead.")
            
            # Add row number as match key
            df_ai['match_key'] = range(len(df_ai))
            df_rb['match_key'] = range(min(len(df_ai), len(df_rb)))  # Limit to the shorter dataframe
            
            # Trim the longer dataframe to match the shorter one
            if len(df_ai) > len(df_rb):
                df_ai = df_ai.iloc[:len(df_rb)].copy()
            elif len(df_rb) > len(df_ai):
                df_rb = df_rb.iloc[:len(df_ai)].copy()
            
            # Merge again using row position
            merged = df_ai.merge(df_rb, on="match_key", suffixes=("_ai", "_rb"), how="outer", indicator=True)

        # Calculate statistics
        total_comparisons, correct_comparisons = 0, 0
        diffs = {}
        unmatched_ai = merged[merged['_merge'] == 'left_only']
        unmatched_rb = merged[merged['_merge'] == 'right_only']
        matched = merged[merged['_merge'] == 'both']

        st.write(f"--- Comparison Results ---")
        st.write(f"Matched Rows: {len(matched)}")
        st.write(f"Unmatched AI Rows (in AI output but not RB): {len(unmatched_ai)}")
        st.write(f"Unmatched RB Rows (in RB output but not AI): {len(unmatched_rb)}")

        # If no matched rows, return 0% accuracy
        if len(matched) == 0:
            st.error("No matching rows found between AI and RB outputs. Cannot calculate accuracy.")
            return 0.0, {}, merged, {}

        for _, row in matched.iterrows():
            for field in FIELDS_TO_COMPARE:
                field_ai = f"{field}_ai"
                field_rb = f"{field}_rb"

                # Skip if the field doesn't exist in either dataframe
                if field_ai not in row or field_rb not in row:
                    continue

                val_ai = row.get(field_ai)
                val_rb = row.get(field_rb)

                # Skip comparison if both values are NaN/None
                if pd.isna(val_ai) and pd.isna(val_rb):
                    correct_comparisons += 1
                    total_comparisons += 1
                    continue

                norm_ai = normalize_value(val_ai)
                norm_rb = normalize_value(val_rb)

                # Special handling for boolean 'is_mtm' (compare normalized 0/1)
                if field == 'is_mtm':
                    try:
                        is_match = (int(val_ai) == int(val_rb))
                    except (ValueError, TypeError):
                        # If conversion fails, compare as strings
                        is_match = (norm_ai == norm_rb)
                # Special handling for numeric fields
                elif field in numeric_cols:
                    if pd.isna(val_ai) and pd.isna(val_rb):
                        is_match = True
                    elif (pd.isna(val_ai) and val_rb == 0) or (pd.isna(val_rb) and val_ai == 0):
                        is_match = True
                    elif pd.isna(val_ai) or pd.isna(val_rb):
                        is_match = False
                    else:
                        try:
                            # Try to compare as floats with a small tolerance
                            is_match = abs(float(val_ai) - float(val_rb)) < 0.01
                        except (ValueError, TypeError):
                            # If conversion fails, compare as strings
                            is_match = (norm_ai == norm_rb)
                else:
                    is_match = (norm_ai == norm_rb)

                total_comparisons += 1
                if is_match:
                    correct_comparisons += 1
                else:
                    diffs.setdefault(field, []).append({
                        "key": row.get("match_key", ""),
                        "ai_value": val_ai,
                        "rb_value": val_rb
                    })

        # Calculate accuracy
        accuracy = round(100 * correct_comparisons / total_comparisons, 2) if total_comparisons > 0 else 0
        st.write(f"\nOverall Accuracy (on matched rows): {accuracy}% ({correct_comparisons}/{total_comparisons} fields)")

        # Calculate per-field statistics
        field_stats = {}
        for field in FIELDS_TO_COMPARE:
            field_total = len(matched)
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

    except FileNotFoundError as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error during CSV comparison: {e}")
        st.text(traceback.format_exc())
        return None, None, None, None


def run_streamlit_app():
    """Main Streamlit app function for CSV-based comparison"""
    st.set_page_config(layout="wide")
    st.title("Rent Roll AI vs Rules-Based Comparison (CSV)")

    st.info("Upload raw rent roll files (.xlsx or .pdf) to compare AI (bypassRB=true) and Rules-Based (bypassRB=false) API outputs, saved as CSV.")

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

    # File upload
    st.subheader("Upload Raw Rent Roll Files")
    uploaded_files = st.file_uploader(
        "Upload Raw Rent Roll Files (.xlsx or .pdf)",
        type=["xlsx", "pdf"],
        accept_multiple_files=True
    )

    # Optional sheet name selection
    sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
    st.info("Note: Sheet name will be ignored for PDF files.")

    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_type = "PDF" if file_ext == '.pdf' else "Excel"
            st.write(f"- {file.name} ({file_type})")

        if st.button("Run Comparison") or st.session_state.results:
            # Create a temporary directory to store files and outputs if not already created
            if not st.session_state.temp_dir:
                st.session_state.temp_dir = tempfile.mkdtemp()
                
                # Create output directories within the temporary directory
                st.session_state.ai_output_path = os.path.join(st.session_state.temp_dir, AI_OUTPUT_FOLDER)
                st.session_state.rb_output_path = os.path.join(st.session_state.temp_dir, RB_OUTPUT_FOLDER)
                st.session_state.comparison_path = os.path.join(st.session_state.temp_dir, COMPARISON_FOLDER)
                
                os.makedirs(st.session_state.ai_output_path, exist_ok=True)
                os.makedirs(st.session_state.rb_output_path, exist_ok=True)
                os.makedirs(st.session_state.comparison_path, exist_ok=True)

            # Process files with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a placeholder for results
            results_container = st.container()

            # Use session state results if available, otherwise process files
            if st.session_state.results:
                results = st.session_state.results
            else:
                # Save uploaded files to the temp directory
                temp_paths = []
                for file in uploaded_files:
                    file_path = os.path.join(st.session_state.temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    temp_paths.append(file_path)

                # Process each file sequentially
                results = []
                for i, file_path in enumerate(temp_paths):
                    file_name = os.path.basename(file_path)
                    status_text.text(f"Processing file {i+1}/{len(temp_paths)}: {file_name}")
                    progress_bar.progress((i) / len(temp_paths))

                    try:
                        # --- Run with bypassRB=true (AI Output) ---
                        st.write(f"Submitting {file_name} to API (AI Output)...")
                        job_id_ai = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=True)

                        if not job_id_ai:
                            st.error(f"Failed to submit AI job to API for {file_name}.")
                            continue

                        st.write(f"AI job submitted successfully. Job ID: {job_id_ai}")

                        st.write(f"Fetching results for AI output on {file_name}...")
                        api_output_ai = fetch_job_results(job_id_ai, max_retries=50, retry_delay=10)

                        if not api_output_ai:
                            st.error(f"Failed to fetch AI job results from API for {file_name}.")
                            continue

                        # Convert AI output to CSV and save
                        ai_csv_string = json_to_csv_string(api_output_ai)
                        if ai_csv_string:
                            ai_csv_filename = os.path.join(st.session_state.ai_output_path, f"{os.path.splitext(file_name)[0]}_AI.csv")
                            with open(ai_csv_filename, 'w') as f:
                                f.write(ai_csv_string)
                            st.write(f"Saved AI output to {ai_csv_filename}")
                        else:
                            st.warning(f"No data found in AI output for {file_name} to save as CSV.")
                            ai_csv_filename = None # Ensure it's None if no data

                        # --- Run with bypassRB=false (Rules-Based Output) ---
                        st.write(f"Submitting {file_name} to API (Rules-Based Output)...")
                        job_id_rb = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=False)

                        if not job_id_rb:
                            st.error(f"Failed to submit Rules-Based job to API for {file_name}.")
                            continue

                        st.write(f"Rules-Based job submitted successfully. Job ID: {job_id_rb}")

                        st.write(f"Fetching results for Rules-Based output on {file_name}...")
                        api_output_rb = fetch_job_results(job_id_rb, max_retries=50, retry_delay=10)

                        if not api_output_rb:
                            st.error(f"Failed to fetch Rules-Based job results from API for {file_name}.")
                            continue

                        # Convert Rules-Based output to CSV and save
                        rb_csv_string = json_to_csv_string(api_output_rb)
                        if rb_csv_string:
                            rb_csv_filename = os.path.join(st.session_state.rb_output_path, f"{os.path.splitext(file_name)[0]}_RB.csv")
                            with open(rb_csv_filename, 'w') as f:
                                f.write(rb_csv_string)
                            st.write(f"Saved Rules-Based output to {rb_csv_filename}")
                        else:
                            st.warning(f"No data found in Rules-Based output for {file_name} to save as CSV.")
                            rb_csv_filename = None # Ensure it's None if no data

                        # --- Compare CSV Outputs and Generate Diff Report ---
                        if ai_csv_filename and rb_csv_filename:
                            st.write(f"Comparing AI and Rules-Based outputs for {file_name}...")
                            accuracy, diffs, merged_df, field_stats = compare_csv_data(ai_csv_filename, rb_csv_filename)

                            if accuracy is not None:
                                # Save detailed differences as JSON
                                diff_output = {
                                    "file": file_name,
                                    "accuracy_percent": accuracy,
                                    "field_mismatches": diffs
                                }
                                comparison_json_filename = os.path.join(st.session_state.comparison_path, f"comparison_diff_{os.path.splitext(file_name)[0]}.json")
                                with open(comparison_json_filename, 'w') as f:
                                    json.dump(diff_output, f, indent=2, default=str)
                                st.write(f"Saved comparison diff to {comparison_json_filename}")

                                results.append({
                                    "file": file_name,
                                    "accuracy": accuracy,
                                    "diffs": diffs,
                                    "merged_df": merged_df,
                                    "field_stats": field_stats,
                                    "ai_csv_path": ai_csv_filename,
                                    "rb_csv_path": rb_csv_filename,
                                    "comparison_json_path": comparison_json_filename
                                })
                            else:
                                st.error(f"Comparison failed for {file_name}. Check logs for details.")
                        else:
                            st.warning(f"Skipping comparison for {file_name} due to missing AI or Rules-Based CSV output.")

                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                        st.text(traceback.format_exc())

                    # Update progress
                    progress_bar.progress((i + 1) / len(temp_paths))

                # Save results to session state
                st.session_state.results = results

            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")

            # Display results and provide download options
            if results:
                with results_container:
                    st.subheader("Comparison Results")

                    # Calculate average accuracy
                    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
                    st.metric("Average Accuracy Across All Files", f"{avg_accuracy:.2f}%")

                    # Create a ZIP file with all results
                    if not st.session_state.zip_buffer:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add all files from the output folders
                            for folder_path in [st.session_state.ai_output_path, st.session_state.rb_output_path, st.session_state.comparison_path]:
                                for root, _, files in os.walk(folder_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        # Add file to zip, preserving folder structure relative to temp_dir
                                        arcname = os.path.relpath(file_path, st.session_state.temp_dir)
                                        zip_file.write(file_path, arcname)

                        # Reset buffer position to the beginning
                        zip_buffer.seek(0)
                        st.session_state.zip_buffer = zip_buffer

                    # Add a download button for the ZIP file
                    # Use a unique key based on a timestamp to prevent rerun issues
                    download_key = f"download_all_results_{int(time.time())}"
                    st.download_button(
                        label="Download All Results as ZIP",
                        data=st.session_state.zip_buffer,
                        file_name="rent_roll_comparison_results.zip",
                        mime="application/zip",
                        key=download_key
                    )

                    # Create tabs for each file
                    tabs = st.tabs([os.path.basename(r["file"]) for r in results])

                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            st.metric("File Accuracy", f"{result['accuracy']:.2f}%")

                            # Display per-field accuracy
                            if result["field_stats"]:
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
                            if result["diffs"]:
                                st.subheader("Field Mismatches")
                                all_mismatches = []
                                for field, mismatches in result["diffs"].items():
                                    for mismatch in mismatches:
                                        # Convert values to strings to avoid Arrow type errors
                                        ai_value = str(mismatch['ai_value']) if mismatch['ai_value'] is not None else ""
                                        rb_value = str(mismatch['rb_value']) if mismatch['rb_value'] is not None else ""
                                        
                                        all_mismatches.append({
                                            "Field": field,
                                            "Unit": mismatch['key'],
                                            "AI Value": ai_value,
                                            "Rules-Based Value": rb_value
                                        })

                                if all_mismatches:
                                    # Convert all values to strings to avoid Arrow type errors
                                    mismatches_df = pd.DataFrame(all_mismatches)
                                    st.dataframe(mismatches_df)

                                    # Text list of differences
                                    st.subheader("Text List of Differences")
                                    for mismatch in all_mismatches:
                                        st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, AI: {mismatch['AI Value']}, Rules-Based: {mismatch['Rules-Based Value']}")
                                else:
                                    st.write("No differences found between AI and Rules-Based outputs.")
                            else:
                                st.write("No differences found between AI and Rules-Based outputs.")

                            # Provide download buttons for individual files
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if result.get("ai_csv_path"):
                                    with open(result["ai_csv_path"], "rb") as f:
                                        ai_data = f.read()
                                    st.download_button(
                                        label="Download AI Output (CSV)",
                                        data=ai_data,
                                        file_name=os.path.basename(result["ai_csv_path"]),
                                        mime="text/csv",
                                        key=f"download_ai_{i}_{int(time.time())}"
                                    )
                            with col2:
                                if result.get("rb_csv_path"):
                                    with open(result["rb_csv_path"], "rb") as f:
                                        rb_data = f.read()
                                    st.download_button(
                                        label="Download Rules-Based Output (CSV)",
                                        data=rb_data,
                                        file_name=os.path.basename(result["rb_csv_path"]),
                                        mime="text/csv",
                                        key=f"download_rb_{i}_{int(time.time())}"
                                    )
                            with col3:
                                if result.get("comparison_json_path"):
                                    with open(result["comparison_json_path"], "rb") as f:
                                        comparison_data = f.read()
                                    st.download_button(
                                        label="Download Comparison Details (JSON)",
                                        data=comparison_data,
                                        file_name=os.path.basename(result["comparison_json_path"]),
                                        mime="application/json",
                                        key=f"download_diff_{i}_{int(time.time())}"
                                    )

                            # Display side-by-side comparison (optional, can be large)
                            if result["merged_df"] is not None and st.checkbox("Show Detailed Comparison Table", key=f"show_table_{i}"):
                                st.dataframe(result["merged_df"])
            else:
                st.error("No successful comparisons were completed. Check the logs for errors.")


if __name__ == "__main__":
    run_streamlit_app()
