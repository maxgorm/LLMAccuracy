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
import re
import concurrent.futures
import asyncio
from typing import Dict, List, Any, Optional, Tuple

# API Key: aEWEppF5Edt5Ffl3kMUOssW4VLrsIwnfGiPj3VDclMQN2DGeIPGXBWX4DKJbJ08CMO46CY6i5LmSg3K328o0AfXioytWYupCAOUIofEPSkDUjZwL3VQamLr4wBjilyWq

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
if 'download_clicked' not in st.session_state:
    st.session_state.download_clicked = False
if 'excluded_fields' not in st.session_state:
    st.session_state.excluded_fields = ['unit_type']  # Default to excluding unit_type to match current behavior

# Import necessary functions from api_rent_roll_verifier
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    async_submit_job,
    async_fetch_job_results,
    normalize_value,
    FIELDS_TO_COMPARE,
    API_BASE_URL,
    API_KEY
)

# Define folder names
AI_OUTPUT_FOLDER = "AI_OUTPUT"
RB_OUTPUT_FOLDER = "RB_OUTPUT"
COMPARISON_FOLDER = "COMPARISON"

def generate_summary_report(results, output_path):
    """
    Generate a summary report CSV file with file names, accuracy scores, and number of diffs.
    
    Args:
        results: List of comparison results
        output_path: Path to save the CSV file
    """
    try:
        # Create a list of dictionaries for the report
        report_data = []
        
        # Count files with 100% accuracy
        perfect_files = sum(1 for result in results if result["accuracy"] == 100.0)
        perfect_percentage = round((perfect_files / len(results) * 100), 2) if results else 0
        
        for result in results:
            # Count total number of diffs across all fields, excluding unit_type
            total_diffs = sum(len(diffs) for field, diffs in result.get("diffs", {}).items() if field != 'unit_type')
            
            report_data.append({
                "File Name": result["file"],
                "Accuracy Score (%)": result["accuracy"],
                "Number of Diffs": total_diffs,
                "Perfect Files (%)": perfect_percentage  # Add the perfect files percentage to each row
            })
        
        # Convert to DataFrame and save as CSV
        if report_data:
            df = pd.DataFrame(report_data)
            df.to_csv(output_path, index=False)
            st.success(f"Summary report generated with {len(report_data)} files")
            st.info(f"Perfect Files (100% accuracy): {perfect_files}/{len(results)} ({perfect_percentage}%)")
        else:
            st.warning("No data available to generate summary report")
            # Create an empty CSV with headers
            pd.DataFrame(columns=["File Name", "Accuracy Score (%)", "Number of Diffs", "Perfect Files (%)"]).to_csv(output_path, index=False)
    
    except Exception as e:
        st.error(f"Error generating summary report: {e}")
        st.text(traceback.format_exc())

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

def process_single_file(file_path, temp_dir, ai_output_path, rb_output_path, comparison_path, sheet_name=None):
    """Process a single rent roll file, generating both AI and Rules-Based outputs and comparing them."""
    file_name = os.path.basename(file_path)
    
    try:
        # --- Run with bypassRB=true (AI Output) ---
        job_id_ai = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=True)

        if not job_id_ai:
            st.error(f"Failed to submit AI job to API for {file_name}.")
            return None

        # Fetch AI results
        api_output_ai = fetch_job_results(job_id_ai, max_retries=50, retry_delay=10)

        if not api_output_ai:
            st.error(f"Failed to fetch AI job results from API for {file_name}.")
            return None

        # Convert AI output to CSV and save
        ai_csv_string = json_to_csv_string(api_output_ai)
        ai_csv_filename = None
        if ai_csv_string:
            ai_csv_filename = os.path.join(ai_output_path, f"{os.path.splitext(file_name)[0]}_AI.csv")
            with open(ai_csv_filename, 'w') as f:
                f.write(ai_csv_string)
        else:
            st.warning(f"No data found in AI output for {file_name} to save as CSV.")

        # --- Run with bypassRB=false (Rules-Based Output) ---
        job_id_rb = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=False)

        if not job_id_rb:
            st.error(f"Failed to submit Rules-Based job to API for {file_name}.")
            return None

        # Fetch Rules-Based results
        api_output_rb = fetch_job_results(job_id_rb, max_retries=50, retry_delay=10)

        if not api_output_rb:
            st.error(f"Failed to fetch Rules-Based job results from API for {file_name}.")
            return None

        # Convert Rules-Based output to CSV and save
        rb_csv_string = json_to_csv_string(api_output_rb)
        rb_csv_filename = None
        if rb_csv_string:
            rb_csv_filename = os.path.join(rb_output_path, f"{os.path.splitext(file_name)[0]}_RB.csv")
            with open(rb_csv_filename, 'w') as f:
                f.write(rb_csv_string)
        else:
            st.warning(f"No data found in Rules-Based output for {file_name} to save as CSV.")

        # --- Compare CSV Outputs and Generate Diff Report ---
        comparison_result = None
        if ai_csv_filename and rb_csv_filename:
            # Ensure excluded_fields exists in session state, use default if not
            excluded_fields = st.session_state.get('excluded_fields', ['unit_type'])
            accuracy, diffs, merged_df, field_stats = compare_csv_data(ai_csv_filename, rb_csv_filename, excluded_fields)

            if accuracy is not None:
                # Save detailed differences as JSON
                diff_output = {
                    "file": file_name,
                    "accuracy_percent": accuracy,
                    "field_mismatches": diffs
                }
                # Round accuracy to nearest whole number for filename
                accuracy_rounded = round(accuracy)
                comparison_json_filename = os.path.join(comparison_path, f"comparison_diff_{accuracy_rounded}_{os.path.splitext(file_name)[0]}.json")
                with open(comparison_json_filename, 'w') as f:
                    json.dump(diff_output, f, indent=2, default=str)

                comparison_result = {
                    "file": file_name,
                    "accuracy": accuracy,
                    "diffs": diffs,
                    "merged_df": merged_df,
                    "field_stats": field_stats,
                    "ai_csv_path": ai_csv_filename,
                    "rb_csv_path": rb_csv_filename,
                    "comparison_json_path": comparison_json_filename
                }
            else:
                st.error(f"Comparison failed for {file_name}. Check logs for details.")
        else:
            st.warning(f"Skipping comparison for {file_name} due to missing AI or Rules-Based CSV output.")

        return comparison_result

    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        st.text(traceback.format_exc())
        return None

async def async_process_single_file(file_path, temp_dir, ai_output_path, rb_output_path, comparison_path, sheet_name=None):
    """Async version to process a single rent roll file, generating both AI and Rules-Based outputs and comparing them."""
    file_name = os.path.basename(file_path)
    
    try:
        # --- Run with bypassRB=true (AI Output) ---
        job_id_ai = await async_submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=True)

        if not job_id_ai:
            st.error(f"Failed to submit AI job to API for {file_name}.")
            return None

        # Fetch AI results
        api_output_ai = await async_fetch_job_results(job_id_ai, max_retries=50, retry_delay=10)

        if not api_output_ai:
            st.error(f"Failed to fetch AI job results from API for {file_name}.")
            return None

        # Convert AI output to CSV and save
        ai_csv_string = json_to_csv_string(api_output_ai)
        ai_csv_filename = None
        if ai_csv_string:
            ai_csv_filename = os.path.join(ai_output_path, f"{os.path.splitext(file_name)[0]}_AI.csv")
            # Write file using executor to avoid blocking
            def write_ai_csv_file():
                with open(ai_csv_filename, 'w') as f:
                    f.write(ai_csv_string)
            await asyncio.get_event_loop().run_in_executor(None, write_ai_csv_file)
        else:
            st.warning(f"No data found in AI output for {file_name} to save as CSV.")

        # --- Run with bypassRB=false (Rules-Based Output) ---
        job_id_rb = await async_submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=False)

        if not job_id_rb:
            st.error(f"Failed to submit Rules-Based job to API for {file_name}.")
            return None

        # Fetch Rules-Based results
        api_output_rb = await async_fetch_job_results(job_id_rb, max_retries=50, retry_delay=10)

        if not api_output_rb:
            st.error(f"Failed to fetch Rules-Based job results from API for {file_name}.")
            return None

        # Convert Rules-Based output to CSV and save
        rb_csv_string = json_to_csv_string(api_output_rb)
        rb_csv_filename = None
        if rb_csv_string:
            rb_csv_filename = os.path.join(rb_output_path, f"{os.path.splitext(file_name)[0]}_RB.csv")
            # Write file using executor to avoid blocking
            def write_rb_csv_file():
                with open(rb_csv_filename, 'w') as f:
                    f.write(rb_csv_string)
            await asyncio.get_event_loop().run_in_executor(None, write_rb_csv_file)
        else:
            st.warning(f"No data found in Rules-Based output for {file_name} to save as CSV.")

        # --- Compare CSV Outputs and Generate Diff Report ---
        comparison_result = None
        if ai_csv_filename and rb_csv_filename:
            # Run compare_csv_data in an executor since it's CPU-bound and not async
            def run_comparison():
                # Ensure excluded_fields exists in session state, use default if not
                excluded_fields = st.session_state.get('excluded_fields', ['unit_type'])
                return compare_csv_data(ai_csv_filename, rb_csv_filename, excluded_fields)
            accuracy, diffs, merged_df, field_stats = await asyncio.get_event_loop().run_in_executor(None, run_comparison)

            if accuracy is not None:
                # Save detailed differences as JSON
                diff_output = {
                    "file": file_name,
                    "accuracy_percent": accuracy,
                    "field_mismatches": diffs
                }
                # Round accuracy to nearest whole number for filename
                accuracy_rounded = round(accuracy)
                comparison_json_filename = os.path.join(comparison_path, f"comparison_diff_{accuracy_rounded}_{os.path.splitext(file_name)[0]}.json")
                # Write file using executor to avoid blocking
                def write_comparison_json_file():
                    with open(comparison_json_filename, 'w') as f:
                        json.dump(diff_output, f, indent=2, default=str)
                await asyncio.get_event_loop().run_in_executor(None, write_comparison_json_file)

                comparison_result = {
                    "file": file_name,
                    "accuracy": accuracy,
                    "diffs": diffs,
                    "merged_df": merged_df,
                    "field_stats": field_stats,
                    "ai_csv_path": ai_csv_filename,
                    "rb_csv_path": rb_csv_filename,
                    "comparison_json_path": comparison_json_filename
                }
            else:
                st.error(f"Comparison failed for {file_name}. Check logs for details.")
        else:
            st.warning(f"Skipping comparison for {file_name} due to missing AI or Rules-Based CSV output.")

        return comparison_result

    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        st.text(traceback.format_exc())
        return None

def compare_csv_data(ai_csv_path, rb_csv_path, excluded_fields=None):
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
        # Note: lease_start is excluded from calculations as requested - will be added back later
        date_cols = ['move_in', 'lease_end']

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
        unit_type_mismatches = 0  # Counter for unit_type mismatches
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

                # Special handling for unit_num field - ignore formatting differences (dashes vs spaces)
                if field == 'unit_num':
                    # Remove all non-alphanumeric characters for comparison
                    clean_ai = re.sub(r'[^a-z0-9]', '', norm_ai)
                    clean_rb = re.sub(r'[^a-z0-9]', '', norm_rb)
                    is_match = (clean_ai == clean_rb)
                # Special handling for tenant field - consider "None" equivalent to vacant indicators
                elif field == 'tenant':
                    # Check if AI value is None/nan and RB is a vacant indicator
                    if (pd.isna(val_ai) or norm_ai in ['none', '', 'nan']) and ('vacant' in norm_rb or '<<<vacant' in norm_rb):
                        is_match = True
                    # Or if RB is None/nan and AI is a vacant indicator
                    elif (pd.isna(val_rb) or norm_rb in ['none', '', 'nan']) and ('vacant' in norm_ai or '<<<vacant' in norm_ai):
                        is_match = True
                    else:
                        is_match = (norm_ai == norm_rb)
                # Special handling for boolean 'is_mtm' (compare normalized 0/1)
                elif field == 'is_mtm':
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
                    # Record all differences in the diffs dictionary
                    diffs.setdefault(field, []).append({
                        "key": row.get("match_key", ""),
                        "ai_value": val_ai,
                        "rb_value": val_rb
                    })
                    
                    # Count unit_type mismatches separately
                    if field == 'unit_type':
                        unit_type_mismatches += 1

        # Calculate the total number of comparisons for each field
        field_comparisons = {}
        for field in FIELDS_TO_COMPARE:
            field_comparisons[field] = len(matched)  # Each field is compared once per matched row
        
        # Calculate the number of mismatches for each field
        field_mismatches = {}
        for field in FIELDS_TO_COMPARE:
            field_mismatches[field] = len(diffs.get(field, []))
        
        # If excluded_fields is None, initialize it as an empty list
        if excluded_fields is None:
            excluded_fields = []
            
        # Calculate the total number of comparisons and correct comparisons, excluding specified fields
        included_fields_total = sum(field_comparisons[field] for field in FIELDS_TO_COMPARE if field not in excluded_fields)
        included_fields_mismatches = sum(field_mismatches[field] for field in FIELDS_TO_COMPARE if field not in excluded_fields)
        included_fields_correct = included_fields_total - included_fields_mismatches
        
        # Calculate accuracy based on included fields only
        if included_fields_total > 0:
            accuracy = round(100 * included_fields_correct / included_fields_total, 2)
        else:
            accuracy = 100.0  # Default to 100% if there are no included fields
        
        excluded_fields_str = ", ".join(excluded_fields) if excluded_fields else "None"
        st.write(f"\nOverall Accuracy (excluding {excluded_fields_str}): {accuracy}% ({included_fields_correct}/{included_fields_total} included fields)")
        
        # Display information about excluded fields
        for field in excluded_fields:
            if field in field_mismatches:
                st.write(f"{field} differences: {field_mismatches.get(field, 0)} (not factored into accuracy)")

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

    st.info("Upload raw rent roll files (.xlsx, .xls, .xlsm, .csv, or .pdf) to compare AI (bypassRB=true) and Rules-Based (bypassRB=false) API outputs, saved as CSV.")

    # API Configuration
    with st.expander("API Configuration"):
        # Import at the beginning of the function to get the current values
        import api_rent_roll_verifier
        api_base_url = st.text_input("API Base URL", value=api_rent_roll_verifier.API_BASE_URL)
        api_key = st.text_input("API Key", type="password", placeholder="Enter your API key here")

        if st.button("Save API Configuration"):
            # Update the module variables directly
            api_rent_roll_verifier.API_BASE_URL = api_base_url
            api_rent_roll_verifier.API_KEY = api_key if api_key else None
            st.success("API configuration saved!")

    # File upload
    st.subheader("Upload Raw Rent Roll Files")
    uploaded_files = st.file_uploader(
        "Upload Raw Rent Roll Files (.xlsx, .xls, .xlsm, .csv, or .pdf)",
        type=["xlsx", "xls", "xlsm", "csv", "pdf"],
        accept_multiple_files=True
    )

    # Optional sheet name selection
    sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
    st.info("Note: Sheet name will be ignored for PDF files.")
    
    # Field selection for accuracy calculation
    st.subheader("Fields to Include in Accuracy Calculation")
    st.write("Select which fields to include when calculating the accuracy percentage:")
    
    # Create a checkbox for each field in FIELDS_TO_COMPARE
    excluded_fields = []
    cols = st.columns(3)  # Display checkboxes in 3 columns for better layout
    
    for i, field in enumerate(FIELDS_TO_COMPARE):
        col_idx = i % 3
        with cols[col_idx]:
            # Default to checked for all fields except unit_type (to match current behavior)
            default_checked = field != 'unit_type'
            if st.checkbox(field, value=default_checked, key=f"field_{field}"):
                # If checked, include in accuracy calculation (not excluded)
                pass
            else:
                # If unchecked, exclude from accuracy calculation
                excluded_fields.append(field)
    
    # Update session state with the current excluded fields
    st.session_state.excluded_fields = excluded_fields

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

                # Process files in parallel using asyncio
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Set the maximum number of concurrent tasks
                max_concurrent = 5
                
                st.write(f"Processing {len(temp_paths)} files in parallel (max {max_concurrent} at a time)...")
                
                # Define async function to process all files
                async def process_all_files():
                    # Create semaphore to limit concurrency
                    semaphore = asyncio.Semaphore(max_concurrent)
                    completed = 0
                    
                    # Define async function to process a single file with semaphore
                    async def process_with_semaphore(file_path):
                        nonlocal completed
                        file_name = os.path.basename(file_path)
                        
                        async with semaphore:
                            try:
                                result = await async_process_single_file(
                                    file_path, 
                                    st.session_state.temp_dir,
                                    st.session_state.ai_output_path,
                                    st.session_state.rb_output_path,
                                    st.session_state.comparison_path,
                                    sheet_name
                                )
                                
                                if result:
                                    results.append(result)
                                    status_text.text(f"Successfully processed: {file_name}")
                                else:
                                    status_text.text(f"Failed to process: {file_name}")
                            except Exception as e:
                                status_text.text(f"Error processing {file_name}: {e}")
                                st.text(traceback.format_exc())
                            
                            # Update progress
                            completed += 1
                            progress_bar.progress(completed / len(temp_paths))
                    
                    # Create tasks for all files
                    tasks = [process_with_semaphore(file_path) for file_path in temp_paths]
                    
                    # Wait for all tasks to complete
                    await asyncio.gather(*tasks)
                
                # Run the async function
                asyncio.run(process_all_files())
                
                # Save results to session state
                st.session_state.results = results
                
                # Print summary
                st.subheader("Processing Summary")
                st.write(f"Total files processed: {len(temp_paths)}")
                st.write(f"Successful: {len(results)}")
                st.write(f"Failed: {len(temp_paths) - len(results)}")

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
                            
                            # Generate and add the summary report CSV
                            summary_report_path = os.path.join(st.session_state.temp_dir, "summary_report.csv")
                            generate_summary_report(results, summary_report_path)
                            # Add the summary report at the root level of the ZIP
                            zip_file.write(summary_report_path, "summary_report.csv")

                        # Reset buffer position to the beginning
                        zip_buffer.seek(0)
                        st.session_state.zip_buffer = zip_buffer

                    # Add a download button for the ZIP file with a stable key
                    # Use a callback to track when download is clicked
                    def on_download_click():
                        st.session_state.download_clicked = True
                    
                    st.download_button(
                        label="Download All Results as ZIP",
                        data=st.session_state.zip_buffer,
                        file_name="rent_roll_comparison_results.zip",
                        mime="application/zip",
                        key="download_all_results",
                        on_click=on_download_click
                    )
                    
                    # Display a message if download was clicked
                    if st.session_state.download_clicked:
                        st.success("Download initiated! If your download doesn't start automatically, please click the button again.")

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
                                        key=f"download_ai_{i}"
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
                                        key=f"download_rb_{i}"
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
                                        key=f"download_diff_{i}"
                                    )

                            # Display side-by-side comparison (optional, can be large)
                            if result["merged_df"] is not None and st.checkbox("Show Detailed Comparison Table", key=f"show_table_{i}"):
                                st.dataframe(result["merged_df"])
            else:
                st.error("No successful comparisons were completed. Check the logs for errors.")


if __name__ == "__main__":
    run_streamlit_app()
