import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback

# Import the API script functions
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    compare_data,
    FIELDS_TO_COMPARE,
    API_BASE_URL,
    API_KEY
)

def run_streamlit_app():
    """Main Streamlit app function for API-based rent roll verification"""
    st.set_page_config(layout="wide")
    st.title("Rent Roll API Verifier")

    st.info("Upload the raw rent roll (.xlsx) and the verified version (.xlsx) to compare API parsing accuracy.")

    # API Configuration
    with st.expander("API Configuration"):
        # Import at the beginning of the function to get the current values
        import api_rent_roll_verifier
        api_base_url = st.text_input("API Base URL", value=api_rent_roll_verifier.API_BASE_URL)
        api_key = st.text_input("API Key (optional)", type="password")
        
        if st.button("Save API Configuration"):
            # Update the module variables directly
            api_rent_roll_verifier.API_BASE_URL = api_base_url
            api_rent_roll_verifier.API_KEY = api_key if api_key else None
            st.success("API configuration saved!")

    uploaded_raw_file = st.file_uploader("1. Upload Raw Rent Roll (.xlsx or .pdf)", type=["xlsx", "pdf"])
    uploaded_verified_file = st.file_uploader("2. Upload Verified Rent Roll (.xlsx)", type="xlsx")

    # Display file type information
    if uploaded_raw_file is not None:
        file_ext = os.path.splitext(uploaded_raw_file.name)[1].lower()
        if file_ext == '.pdf':
            st.info("PDF file detected. Sheet name will be ignored for PDF files.")

    if uploaded_raw_file and uploaded_verified_file:
        st.success("Files uploaded successfully!")

        # Save temporary files to run the main logic
        raw_temp_path = f"temp_{uploaded_raw_file.name}"
        verified_temp_path = f"temp_{uploaded_verified_file.name}"

        with open(raw_temp_path, "wb") as f:
            f.write(uploaded_raw_file.getbuffer())
        with open(verified_temp_path, "wb") as f:
            f.write(uploaded_verified_file.getbuffer())

        # Optional sheet name selection
        sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")

        if st.button("Run Verification"):
            with st.spinner("Processing files and querying API... This may take a minute."):
                # Submit job to API
                st.write("Submitting job to API...")
                job_id = submit_job(raw_temp_path, doc_type="rent_roll", sheet_name=sheet_name)
                
                if not job_id:
                    st.error("Failed to submit job to API.")
                else:
                    st.write(f"Job submitted successfully. Job ID: {job_id}")
                    
                    # Fetch job results with progress bar
                    st.write("Fetching job results...")
                    progress_bar = st.progress(0)
                    
                    # Create a placeholder for status updates
                    status_text = st.empty()
                    
                    # Set up retry parameters
                    max_retries = 10
                    retry_delay = 5
                    
                    # Implement our own retry logic with progress updates
                    for retry in range(max_retries):
                        status_text.text(f"Attempt {retry + 1}/{max_retries}: Checking job status...")
                        progress_bar.progress((retry + 1) / max_retries)
                        
                        # Call the API to check job status
                        api_output = fetch_job_results(job_id, max_retries=1, retry_delay=0)
                        
                        if api_output is None:
                            status_text.text(f"Attempt {retry + 1}/{max_retries}: API request failed. Retrying...")
                            st.sleep(retry_delay)
                            continue
                            
                        # Check if job is still processing
                        if api_output.get('job', {}).get('status') == 'processing':
                            status_text.text(f"Attempt {retry + 1}/{max_retries}: Job is still processing. Waiting...")
                            st.sleep(retry_delay)
                            continue
                            
                        # Job is complete
                        status_text.text("Job completed successfully!")
                        progress_bar.progress(1.0)
                        break
                    else:
                        # Loop completed without breaking - max retries reached
                        st.error("Maximum retries reached. Job may still be processing.")
                        api_output = None
                    
                    if api_output:
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

                            # Provide download for API output
                            st.download_button(
                                label="Download Raw API Output (JSON)",
                                data=json.dumps(api_output, indent=2, default=str),
                                file_name="api_output.json",
                                mime="application/json",
                            )

                        else:
                            st.error("Comparison failed. Check logs or input files.")

            # Clean up temp files
            try:
                os.remove(raw_temp_path)
                os.remove(verified_temp_path)
                if 'api_output_filename' in locals() and os.path.exists(api_output_filename):
                    os.remove(api_output_filename)
            except OSError as e:
                st.warning(f"Could not remove temporary files: {e}")

if __name__ == "__main__":
    run_streamlit_app()
