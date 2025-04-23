import pandas as pd
import json
import requests
import openpyxl  # Needed for pandas to read .xlsx
from datetime import datetime
import re  # For potential string cleaning
import os
import time
import pdfplumber
import tabula

# --- Configuration ---
API_BASE_URL = "https://demo.quickdata.ai/api/v1"
API_KEY = "aEWEppF5Edt5Ffl3kMUOssW4VLrsIwnfGiPj3VDclMQN2DGeIPGXBWX4DKJbJ08CMO46CY6i5LmSg3K328o0AfXioytWYupCAOUIofEPSkDUjZwL3VQamLr4wBjilyWq"

# Fields to compare (same as in rent_roll_verifier.py)
FIELDS_TO_COMPARE = [
    "unit_num", "unit_type", "sqft", "br", "bath", "tenant", "move_in",
    "lease_start", "lease_end", "rent_charge", "rent_gov_subsidy",
    "is_mtm", "mtm_charge", "rent_market"
]

# --- API Functions ---

def check_api_key():
    """Check if API key is set, prompt user if not."""
    global API_KEY
    if API_KEY is None:
        print("API key not set. You may need an API key to access the service.")
        key = input("Enter your API key (press Enter to skip): ").strip()
        if key:
            API_KEY = key
            print("API key set.")
        else:
            print("No API key provided. Proceeding without authentication.")
            print("Note: The API may reject requests without proper authentication.")

def submit_job(file_path, doc_type="rent_roll", sheet_name=""):
    """Submit a new job to the API."""
    check_api_key()
    
    url = f"{API_BASE_URL}/jobs"
    
    # Prepare headers with API key if available
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    # Determine file type and set appropriate content type
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        content_type = 'application/pdf'
        print(f"Submitting PDF file: {file_path}")
    else:  # Default to Excel for .xlsx and other formats
        content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        print(f"Submitting Excel file: {file_path}")
    
    # Prepare multipart form data
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), content_type)
    }
    
    data = {
        'doc_type': doc_type,
        'sheet_name': sheet_name if file_ext != '.pdf' else ''  # Sheet name only applies to Excel files
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        result = response.json()
        print(f"Job submitted successfully. Job ID: {result.get('job_id')}")
        return result.get('job_id')
    
    except requests.exceptions.RequestException as e:
        print(f"Error submitting job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    finally:
        # Make sure to close the file
        files['file'][1].close()

def fetch_job_results(job_id, max_retries=10, retry_delay=5):
    """Fetch job results, with retry logic for jobs that are still processing."""
    check_api_key()
    
    url = f"{API_BASE_URL}/jobs/{job_id}"
    
    # Set a large page size to avoid pagination
    params = {
        'pg_size': 10000,  # As mentioned in the requirements
        'pg_idx': 1
    }
    
    # Prepare headers with API key if available
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if job is still processing
            if result.get('job', {}).get('status') == 'processing':
                print(f"Job is still processing. Retrying in {retry_delay} seconds... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay)
                retries += 1
                continue
            
            # Job is complete, return the results
            return result
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching job results: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
    
    print(f"Max retries ({max_retries}) reached. Job may still be processing.")
    return None

# --- Helper Functions ---

def normalize_value(value):
    """Normalizes values for comparison (string, lowercase, strip whitespace)."""
    if pd.isna(value):
        return ""  # Treat NaN/None as empty string for comparison
    return str(value).strip().lower()

def compare_data(api_output, verified_file_path):
    """Compares API JSON output with the verified Excel file."""
    try:
        # Load verified Excel directly, assuming header is on row 3 (index 2)
        df_verified = pd.read_excel(verified_file_path, engine='openpyxl', header=2)
        print(f"Read verified file '{verified_file_path}' assuming header is on row 3.")
        
        # Normalize verified column names
        df_verified.columns = df_verified.columns.str.strip().str.lower()
        
        # Ensure required columns exist, handling potential case differences
        required_verified_cols = {'tenant', 'move_in'} | {f.lower() for f in FIELDS_TO_COMPARE}
        missing_verified_cols = required_verified_cols - set(df_verified.columns)
        if missing_verified_cols:
            print(f"Warning: Missing required columns in verified file: {missing_verified_cols}")
            # Decide how to handle: error out, or proceed with available columns?
            # For now, let's try to proceed but comparisons for missing cols will fail.

        # Check if API output has the expected structure
        if not api_output or "df" not in api_output or not api_output["df"]:
            print("Error: API output is empty or not in the expected format.")
            return None, None, None

        # Convert API output to DataFrame
        df_api = pd.DataFrame(api_output["df"])
        
        # Normalize API column names
        df_api.columns = df_api.columns.str.strip().str.lower()
        missing_api_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_api.columns)
        if missing_api_cols:
            print(f"Warning: Missing expected columns in API output: {missing_api_cols}")
            # Add missing columns with NaN to allow merge/comparison
            for col in missing_api_cols:
                df_api[col] = pd.NA

        # --- Data Type Conversion & Cleaning ---
        # Convert date columns safely, coercing errors to NaT
        date_cols_api = ['move_in', 'lease_start', 'lease_end']
        date_cols_verified = ['move_in', 'lease_start', 'lease_end']  # Assuming same names after lowercasing

        for col in date_cols_api:
            if col in df_api.columns:
                # Handle 'nan' strings before conversion
                df_api[col] = df_api[col].replace('nan', pd.NA)
                df_api[col] = pd.to_datetime(df_api[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df_api[col] = pd.NA  # Ensure column exists if missing

        for col in date_cols_verified:
            if col in df_verified.columns:
                df_verified[col] = pd.to_datetime(df_verified[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df_verified[col] = pd.NA  # Ensure column exists if missing

        # Convert numeric columns safely, coercing errors to NaN
        numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
        for col in numeric_cols:
            if col in df_api.columns:
                df_api[col] = df_api[col].replace('nan', pd.NA)
                df_api[col] = pd.to_numeric(df_api[col], errors='coerce')
            if col in df_verified.columns:
                df_verified[col] = pd.to_numeric(df_verified[col], errors='coerce')

        # Handle boolean 'is_mtm' (normalize 0/1, True/False, 'true'/'false' to 0/1)
        if 'is_mtm' in df_api.columns:
            df_api['is_mtm'] = df_api['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_api['is_mtm'] = pd.to_numeric(df_api['is_mtm'], errors='coerce').fillna(0).astype(int)  # Default to 0 (False) if conversion fails or missing
        if 'is_mtm' in df_verified.columns:
            df_verified['is_mtm'] = df_verified['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_verified['is_mtm'] = pd.to_numeric(df_verified['is_mtm'], errors='coerce').fillna(0).astype(int)

        # --- Construct Match Key ---
        # Check if move_in dates are available in API output
        has_move_in_dates = not df_api['move_in'].isna().all() and not (df_api['move_in'] == 'nan').all()
        
        if has_move_in_dates:
            # Use normalized tenant name and move-in date using vectorized operations
            print("Using tenant name and move-in date for matching...")
            tenant_col_v = df_verified.get('tenant', pd.Series(dtype=str))  # Get column or empty series
            move_in_col_v = df_verified.get('move_in', pd.Series(dtype=str))
            df_verified['match_key'] = tenant_col_v.fillna('').astype(str).str.strip().str.lower() + "|" + \
                                      move_in_col_v.fillna('').astype(str).str.strip().str.lower()

            tenant_col_l = df_api.get('tenant', pd.Series(dtype=str))
            move_in_col_l = df_api.get('move_in', pd.Series(dtype=str))
            df_api['match_key'] = tenant_col_l.fillna('').astype(str).str.strip().str.lower() + "|" + \
                                 move_in_col_l.fillna('').astype(str).str.strip().str.lower()
        else:
            # Use only tenant name for matching if move_in dates are not available
            print("Move-in dates not available in API output. Using only tenant name for matching...")
            tenant_col_v = df_verified.get('tenant', pd.Series(dtype=str))
            df_verified['match_key'] = tenant_col_v.fillna('').astype(str).str.strip().str.lower()
            
            tenant_col_l = df_api.get('tenant', pd.Series(dtype=str))
            df_api['match_key'] = tenant_col_l.fillna('').astype(str).str.strip().str.lower()
            
            # Remove commas from tenant names for better matching
            df_verified['match_key'] = df_verified['match_key'].str.replace(',', '')
            df_api['match_key'] = df_api['match_key'].str.replace(',', '')

        # --- Merge and Compare ---
        merged = df_api.merge(df_verified, on="match_key", suffixes=("_api", "_verified"), how="outer", indicator=True)

        total_comparisons, correct_comparisons = 0, 0
        diffs = {}
        unmatched_api = merged[merged['_merge'] == 'left_only']
        unmatched_verified = merged[merged['_merge'] == 'right_only']
        matched = merged[merged['_merge'] == 'both']

        print(f"\n--- Comparison Results ---")
        print(f"Matched Rows: {len(matched)}")
        print(f"Unmatched API Rows (in API output but not Verified): {len(unmatched_api)}")
        print(f"Unmatched Verified Rows (in Verified but not API output): {len(unmatched_verified)}")

        for _, row in matched.iterrows():
            for field in FIELDS_TO_COMPARE:
                field_api = f"{field}_api"
                field_verified = f"{field}_verified"

                # Check if columns exist before trying to access
                val_api = row.get(field_api)
                val_verified = row.get(field_verified)

                # Normalize for comparison
                norm_api = normalize_value(val_api)
                norm_verified = normalize_value(val_verified)

                # Special handling for boolean 'is_mtm' (compare normalized 0/1)
                if field == 'is_mtm':
                    # Already converted to 0/1 int above
                    is_match = (int(val_api) == int(val_verified))
                # Special handling for numeric fields (allow small tolerance?) - For now, exact match after normalization
                elif field in numeric_cols:
                    # Handle potential NaN comparison carefully
                    if pd.isna(val_api) and pd.isna(val_verified):
                        is_match = True
                    # Special case: treat 0 and nan as a match
                    elif (pd.isna(val_api) and val_verified == 0) or (pd.isna(val_verified) and val_api == 0):
                        is_match = True
                    elif pd.isna(val_api) or pd.isna(val_verified):
                        is_match = False
                    else:
                        # Could add tolerance here if needed: abs(val_api - val_verified) < tolerance
                        is_match = (float(val_api) == float(val_verified))
                else:  # Default string comparison
                    is_match = (norm_api == norm_verified)

                total_comparisons += 1
                if is_match:
                    correct_comparisons += 1
                else:
                    diffs.setdefault(field, []).append({
                        "key": row["match_key"],
                        "api_value": val_api,
                        "verified_value": val_verified
                    })

        accuracy = round(100 * correct_comparisons / total_comparisons, 2) if total_comparisons > 0 else 0
        print(f"\nOverall Accuracy (on matched rows): {accuracy}% ({correct_comparisons}/{total_comparisons} fields)")

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
        
        # Report per-field accuracy
        print("\n--- Per-Field Accuracy ---")
        for field, stats in sorted(field_stats.items(), key=lambda x: x[1]['accuracy']):
            print(f"  Field '{field}': {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
        
        # Report per-field mismatches with examples
        if diffs:
            print("\n--- Field Mismatches (Examples) ---")
            for field, mismatches in diffs.items():
                print(f"  Field '{field}': {len(mismatches)} mismatches")
                # Print details for a few mismatches
                for i, mismatch in enumerate(mismatches[:3]):
                    print(f"    - Key: {mismatch['key']}")
                    print(f"      API: '{mismatch['api_value']}', Verified: '{mismatch['verified_value']}'")
                if len(mismatches) > 3:
                    print(f"    - ... ({len(mismatches) - 3} more mismatches)")

        # Report unmatched rows
        if not unmatched_api.empty:
            print("\n--- Unmatched API Rows (Examples) ---")
            print(unmatched_api[['match_key'] + [f for f in FIELDS_TO_COMPARE if f in unmatched_api.columns]].head())

        if not unmatched_verified.empty:
            print("\n--- Unmatched Verified Rows (Examples) ---")
            print(unmatched_verified[['match_key'] + [f for f in FIELDS_TO_COMPARE if f in unmatched_verified.columns]].head())

        return accuracy, diffs, merged  # Return merged for potential further analysis/export

    except FileNotFoundError:
        print(f"Error: Verified file not found at {verified_file_path}")
        return None, None, None
    except KeyError as e:
        print(f"Error: Missing expected key during processing: {e}")
        print("This might be due to unexpected API output format or missing columns in input files.")
        return None, None, None
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return None, None, None

# --- Main Execution Logic ---

def main(raw_file, verified_file, output_diff_file=None):
    """Main function to run the verification process."""

    print(f"Processing raw file: {raw_file}")
    print(f"Comparing against verified file: {verified_file}")

    # Step 1: Submit the job to the API
    job_id = submit_job(raw_file)
    if not job_id:
        print("Failed to submit job to API. Exiting.")
        return

    # Step 2: Fetch the job results
    print(f"\nFetching results for job ID: {job_id}...")
    api_output = fetch_job_results(job_id)
    
    if not api_output:
        print("Failed to fetch job results from API. Exiting.")
        return

    # Save intermediate API output for inspection
    api_output_filename = "api_output.json"
    try:
        with open(api_output_filename, 'w') as f:
            json.dump(api_output, f, indent=2)
        print(f"\nSaved API output to {api_output_filename}")
    except Exception as e:
        print(f"Warning: Could not save API output JSON: {e}")

    # Step 3: Compare API output with Verified Data
    print("\nComparing API output with verified data...")
    accuracy, diffs, merged_df = compare_data(api_output, verified_file)

    if accuracy is not None and diffs is not None and output_diff_file:
        print(f"\nSaving differences to {output_diff_file}...")
        try:
            # Save detailed differences
            diff_output = {
                "accuracy_percent": accuracy,
                "field_mismatches": diffs,
                # Optionally include unmatched rows if needed
                # "unmatched_api_rows": merged_df[merged_df['_merge'] == 'left_only'].to_dict('records'),
                # "unmatched_verified_rows": merged_df[merged_df['_merge'] == 'right_only'].to_dict('records')
            }
            with open(output_diff_file, 'w') as f:
                json.dump(diff_output, f, indent=2, default=str)  # Use default=str for non-serializable types like NaT
            print("Differences saved.")
        except Exception as e:
            print(f"Error saving differences JSON: {e}")

    print("\nVerification process complete.")

# --- Streamlit App ---
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(layout="wide")
    st.title("Rent Roll API Verifier")

    st.info("Upload the raw rent roll (.xlsx) and the verified version (.xlsx) to compare API parsing accuracy.")

    # API Configuration
    with st.expander("API Configuration"):
        # Get current values
        api_base_url = st.text_input("API Base URL", value=API_BASE_URL)
        api_key = st.text_input("API Key (optional)", type="password")
        
        if st.button("Save API Configuration"):
            # Update module variables directly
            import sys
            current_module = sys.modules[__name__]
            setattr(current_module, 'API_BASE_URL', api_base_url)
            setattr(current_module, 'API_KEY', api_key if api_key else None)
            st.success("API configuration saved!")

    uploaded_raw_file = st.file_uploader("1. Upload Raw Rent Roll (.xlsx)", type="xlsx")
    uploaded_verified_file = st.file_uploader("2. Upload Verified Rent Roll (.xlsx)", type="xlsx")

    if uploaded_raw_file and uploaded_verified_file:
        st.success("Files uploaded successfully!")

        # Save temporary files to run the main logic
        raw_temp_path = f"temp_{uploaded_raw_file.name}"
        verified_temp_path = f"temp_{uploaded_verified_file.name}"
        diff_temp_path = "temp_diff.json"

        with open(raw_temp_path, "wb") as f:
            f.write(uploaded_raw_file.getbuffer())
        with open(verified_temp_path, "wb") as f:
            f.write(uploaded_verified_file.getbuffer())

        if st.button("Run Verification"):
            with st.spinner("Processing files and querying API... This may take a minute."):
                # Submit job to API
                st.write("Submitting job to API...")
                job_id = submit_job(raw_temp_path)
                
                if not job_id:
                    st.error("Failed to submit job to API.")
                else:
                    st.write(f"Job submitted successfully. Job ID: {job_id}")
                    
                    # Fetch job results
                    st.write("Fetching job results...")
                    api_output = fetch_job_results(job_id)
                    
                    if not api_output:
                        st.error("Failed to fetch job results from API.")
                    else:
                        st.write("Job results fetched successfully.")
                        
                        # Save API output for reference
                        api_output_filename = f"temp_api_output_{uploaded_raw_file.name}.json"
                        with open(api_output_filename, 'w') as f:
                            json.dump(api_output, f, indent=2)
                        
                        # Compare
                        st.write("Comparing API output with verified data...")
                        accuracy, diffs, merged_df = compare_data(api_output, verified_temp_path)

                        if accuracy is not None:
                            st.subheader("Verification Results")
                            st.metric("Overall Accuracy", f"{accuracy:.2f}%")

                            # Calculate per-field accuracy for display
                            if diffs:
                                field_stats = {}
                                for field in FIELDS_TO_COMPARE:
                                    field_total = len(merged_df[merged_df['_merge'] == 'both'])
                                    field_mismatches = len(diffs.get(field, []))
                                    field_correct = field_total - field_mismatches
                                    field_accuracy = round(100 * field_correct / field_total, 2) if field_total > 0 else 0
                                    field_stats[field] = {
                                        "accuracy": field_accuracy,
                                        "mismatches": field_mismatches,
                                        "total": field_total
                                    }
                                
                                # Display per-field accuracy
                                st.subheader("Per-Field Accuracy")
                                field_df = pd.DataFrame({
                                    "Field": [field for field in field_stats.keys()],
                                    "Accuracy (%)": [stats["accuracy"] for stats in field_stats.values()],
                                    "Mismatches": [stats["mismatches"] for stats in field_stats.values()],
                                    "Total": [stats["total"] for stats in field_stats.values()]
                                })
                                field_df = field_df.sort_values("Accuracy (%)")
                                st.dataframe(field_df)
                                
                                # Display mismatch examples
                                st.subheader("Field Mismatches (Examples)")
                                for field, mismatches in diffs.items():
                                    if mismatches:
                                        with st.expander(f"{field}: {len(mismatches)} mismatches"):
                                            for i, mismatch in enumerate(mismatches[:5]):
                                                st.write(f"**Key:** {mismatch['key']}")
                                                st.write(f"**API:** '{mismatch['api_value']}', **Verified:** '{mismatch['verified_value']}'")
                                            if len(mismatches) > 5:
                                                st.write(f"... and {len(mismatches) - 5} more mismatches")
                            
                            # Provide download for full diff details
                            diff_output = {
                                "accuracy_percent": accuracy,
                                "field_mismatches": diffs
                            }
                            st.download_button(
                                label="Download Mismatch Details (JSON)",
                                data=json.dumps(diff_output, indent=2, default=str),
                                file_name="api_comparison_diffs.json",
                                mime="application/json",
                            )

                            # Display side-by-side comparison (optional, can be large)
                            if merged_df is not None and st.checkbox("Show Detailed Comparison Table"):
                                st.dataframe(merged_df)

                        else:
                            st.error("Comparison failed. Check logs or input files.")

            # Clean up temp files
            import os
            try:
                os.remove(raw_temp_path)
                os.remove(verified_temp_path)
                if os.path.exists(api_output_filename):
                    os.remove(api_output_filename)
            except OSError as e:
                st.warning(f"Could not remove temporary files: {e}")

if __name__ == "__main__":
    import sys
    
    # If 'streamlit' is in sys.modules, we're being run by Streamlit
    if 'streamlit' in sys.modules:
        # We're running in Streamlit mode
        try:
            import streamlit as st
            # Run the Streamlit app
            run_streamlit_app()
        except ImportError:
            print("Streamlit is not installed. Please install it with: pip install streamlit")
    else:
        # We're running as a regular Python script
        # Use the provided sample files
        RAW_RENT_ROLL_FILE = "Saddlebrook I RR 02-23-24.xlsx"
        VERIFIED_RENT_ROLL_FILE = "Saddlebrook I RR 02-23-24.xlsx Verified.xlsx"
        OUTPUT_DIFF_JSON = "api_comparison_diff.json"  # Optional: Set to None to disable saving diffs
        
        main(RAW_RENT_ROLL_FILE, VERIFIED_RENT_ROLL_FILE, OUTPUT_DIFF_JSON)

# End of script
