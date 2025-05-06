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

def submit_job(file_path, doc_type="rent_roll", sheet_name="", bypass_rb=False):
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
        'sheet_name': sheet_name if file_ext != '.pdf' else '',  # Sheet name only applies to Excel files
        'bypass_rb': bypass_rb  # Add bypass_rb parameter to bypass rules-based approach and always use LLMs
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

        # --- Use Row Position for Matching ---
        print("Using row position for matching instead of unit numbers...")
        
        # Add row number as match key
        df_verified['match_key'] = range(len(df_verified))
        df_api['match_key'] = range(len(df_api))
        
        # Check if the number of rows match
        if len(df_api) != len(df_verified):
            print(f"Warning: Number of rows in API output ({len(df_api)}) does not match verified file ({len(df_verified)})")
            print("Will match rows by position up to the minimum length of both datasets")
            
            # Trim to the shorter length to ensure 1:1 matching
            min_rows = min(len(df_api), len(df_verified))
            df_api = df_api.iloc[:min_rows].copy()
            df_verified = df_verified.iloc[:min_rows].copy()
            
            # Reset match keys to ensure they match
            df_verified['match_key'] = range(len(df_verified))
            df_api['match_key'] = range(len(df_api))
        
        # Print row counts for debugging
        print(f"\nAPI rows: {len(df_api)}")
        print(f"Verified rows: {len(df_verified)}")

        # --- Merge and Compare ---
        # Since we're using row position, all rows should match
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

                # Special handling for unit_num field - ignore formatting differences (dashes vs spaces)
                if field == 'unit_num':
                    # Remove all non-alphanumeric characters for comparison
                    clean_api = re.sub(r'[^a-z0-9]', '', norm_api)
                    clean_verified = re.sub(r'[^a-z0-9]', '', norm_verified)
                    is_match = (clean_api == clean_verified)
                    
                # Special handling for tenant field - consider "None" equivalent to vacant indicators
                elif field == 'tenant':
                    # Check if API value is None/nan and verified is a vacant indicator
                    if (norm_api in ['none', ''] and 
                        ('vacant' in norm_verified or norm_verified in ['', 'nan'])):
                        is_match = True
                    # Or if verified is None/nan and API is a vacant indicator
                    elif (norm_verified in ['none', ''] and 
                          ('vacant' in norm_api or norm_api in ['', 'nan'])):
                        is_match = True
                    else:
                        is_match = (norm_api == norm_verified)
                
                # Special handling for boolean 'is_mtm' (compare normalized 0/1)
                elif field == 'is_mtm':
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

def process_single_file(raw_file, verified_file, output_diff_file=None, bypass_rb=False):
    """Process a single rent roll file and its verified counterpart using the API."""
    print(f"Processing raw file: {raw_file}")
    print(f"Comparing against verified file: {verified_file}")

    # Step 1: Submit the job to the API
    job_id = submit_job(raw_file, bypass_rb=bypass_rb)
    if not job_id:
        print(f"Failed to submit job to API for {raw_file}. Exiting.")
        return None, None, None

    # Step 2: Fetch the job results
    print(f"\nFetching results for job ID: {job_id}...")
    api_output = fetch_job_results(job_id)
    
    if not api_output:
        print(f"Failed to fetch job results from API for {raw_file}. Exiting.")
        return None, None, None

    # Save intermediate API output for inspection
    api_output_filename = f"api_output_{os.path.basename(raw_file)}.json"
    try:
        with open(api_output_filename, 'w') as f:
            json.dump(api_output, f, indent=2)
        print(f"\nSaved API output to {api_output_filename}")
    except Exception as e:
        print(f"Warning: Could not save API output JSON for {raw_file}: {e}")

    # Step 3: Compare API output with Verified Data
    print(f"\nComparing API output with verified data for {raw_file}...")
    accuracy, diffs, merged_df = compare_data(api_output, verified_file)

    if accuracy is not None and diffs is not None and output_diff_file:
        print(f"\nSaving differences to {output_diff_file}...")
        try:
            # Save detailed differences
            diff_output = {
                "raw_file": raw_file,
                "verified_file": verified_file,
                "accuracy_percent": accuracy,
                "field_mismatches": diffs,
            }
            with open(output_diff_file, 'w') as f:
                json.dump(diff_output, f, indent=2, default=str)  # Use default=str for non-serializable types like NaT
            print(f"Differences saved for {raw_file}.")
        except Exception as e:
            print(f"Error saving differences JSON for {raw_file}: {e}")

    print(f"\nVerification process complete for {raw_file}.")
    return accuracy, diffs, merged_df

def match_files(raw_files, verified_files):
    """Match raw rent roll files with their corresponding verified files based on filename similarity."""
    if len(raw_files) != len(verified_files):
        print(f"Error: Number of raw files ({len(raw_files)}) does not match number of verified files ({len(verified_files)})")
        return None
    
    matched_pairs = []
    
    # Extract base names without extensions for matching
    raw_base_names = [os.path.splitext(os.path.basename(f))[0] for f in raw_files]
    verified_base_names = [os.path.splitext(os.path.basename(f))[0].replace(" Verified", "") for f in verified_files]
    
    # For each raw file, find the best matching verified file
    for i, raw_file in enumerate(raw_files):
        raw_base = raw_base_names[i]
        
        # Find the verified file with the most similar name
        best_match_idx = -1
        best_match_score = -1
        
        for j, verified_base in enumerate(verified_base_names):
            # Simple similarity score: length of common prefix
            # This works well for files that differ only by a suffix like "(1)" or numbers at the end
            common_prefix_len = 0
            for a, b in zip(raw_base, verified_base):
                if a == b:
                    common_prefix_len += 1
                else:
                    break
            
            # If this is a better match than what we've seen so far
            if common_prefix_len > best_match_score:
                best_match_score = common_prefix_len
                best_match_idx = j
        
        if best_match_idx >= 0:
            matched_pairs.append((raw_file, verified_files[best_match_idx]))
            # Remove the matched verified file to prevent duplicate matches
            verified_files.pop(best_match_idx)
            verified_base_names.pop(best_match_idx)
    
    return matched_pairs

def main(raw_files, verified_files, output_diff_dir=None, bypass_rb=False):
    """Main function to run the verification process for multiple files."""
    
    # Validate inputs
    if not raw_files or not verified_files:
        print("Error: No files provided")
        return
    
    # Convert to lists if single strings were provided
    if isinstance(raw_files, str):
        raw_files = [raw_files]
    if isinstance(verified_files, str):
        verified_files = [verified_files]
    
    # Check if the number of files match
    if len(raw_files) != len(verified_files):
        print(f"Error: Number of raw files ({len(raw_files)}) does not match number of verified files ({len(verified_files)})")
        print("Please ensure there is one verified file for each raw file.")
        return
    
    # Match raw files with their corresponding verified files
    file_pairs = match_files(raw_files, verified_files.copy())
    if not file_pairs:
        print("Error: Failed to match files")
        return
    
    # Print matched pairs for verification
    print("\nMatched file pairs:")
    for raw, verified in file_pairs:
        print(f"  {os.path.basename(raw)} -> {os.path.basename(verified)}")
    
    # Create output directory if needed
    if output_diff_dir:
        os.makedirs(output_diff_dir, exist_ok=True)
    
    # Process each file pair
    results = []
    for i, (raw_file, verified_file) in enumerate(file_pairs):
        print(f"\n[{i+1}/{len(file_pairs)}] Processing file pair:")
        
        # Create output diff file path if directory was provided
        output_diff_file = None
        if output_diff_dir:
            output_diff_file = os.path.join(output_diff_dir, f"api_comparison_diff_{os.path.basename(raw_file)}.json")
        
        # Process the file pair
        accuracy, diffs, merged_df = process_single_file(
            raw_file, 
            verified_file, 
            output_diff_file,
            bypass_rb
        )
        
        if accuracy is not None:
            results.append({
                "raw_file": raw_file,
                "verified_file": verified_file,
                "accuracy": accuracy,
                "has_diffs": bool(diffs)
            })
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total file pairs processed: {len(file_pairs)}")
    print(f"Successful comparisons: {len(results)}")
    
    if results:
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        print(f"Average accuracy across all files: {avg_accuracy:.2f}%")
        
        # Print individual file results
        print("\nIndividual file results:")
        for r in results:
            print(f"  {os.path.basename(r['raw_file'])}: {r['accuracy']:.2f}% accuracy")
    
    print("\nVerification process complete for all files.")

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
    
    # Processing options
    with st.expander("Processing Options"):
        bypass_rb = st.checkbox("Bypass Rules-Based Approach", value=False, 
                               help="When checked, the API will always use LLMs instead of the rules-based approach")
        st.info("Bypassing the rules-based approach will use LLMs for all processing, which may be more accurate but slower.")

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
                job_id = submit_job(raw_temp_path, bypass_rb=bypass_rb)
                
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
        # Check if command-line arguments are provided
        if len(sys.argv) > 1:
            # Use command-line arguments
            raw_files = sys.argv[1].split(',')
            verified_files = sys.argv[2].split(',') if len(sys.argv) > 2 else ["Saddlebrook I RR 02-23-24.xlsx Verified.xlsx"]
            output_diff_dir = sys.argv[3] if len(sys.argv) > 3 else "api_comparison_diffs"
            bypass_rb = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
            
            print(f"Using command-line arguments:")
            print(f"Raw files: {raw_files}")
            print(f"Verified files: {verified_files}")
            print(f"Output diff directory: {output_diff_dir}")
            print(f"Bypass rules-based approach: {bypass_rb}")
            
            main(raw_files, verified_files, output_diff_dir, bypass_rb)
        else:
            # Use the provided sample files
            RAW_RENT_ROLL_FILES = ["Saddlebrook I RR 02-23-24.xlsx"]
            VERIFIED_RENT_ROLL_FILES = ["Saddlebrook I RR 02-23-24.xlsx Verified.xlsx"]
            OUTPUT_DIFF_DIR = "api_comparison_diffs"  # Optional: Set to None to disable saving diffs
            BYPASS_RB = False  # Default to using rules-based approach
            
            main(RAW_RENT_ROLL_FILES, VERIFIED_RENT_ROLL_FILES, OUTPUT_DIFF_DIR, BYPASS_RB)

# End of script
