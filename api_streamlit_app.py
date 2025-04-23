import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback
import tempfile
import time

# Import the API script functions
from api_rent_roll_verifier import (
    process_single_file,
    match_files,
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

    st.info("Upload raw rent roll files (.xlsx or .pdf) and their corresponding verified versions (.xlsx) to compare API parsing accuracy.")

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

    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Rent Roll Files")
        uploaded_raw_files = st.file_uploader(
            "Upload Raw Rent Roll Files (.xlsx or .pdf)", 
            type=["xlsx", "pdf"],
            accept_multiple_files=True
        )
    
    with col2:
        st.subheader("Verified Rent Roll Files")
        uploaded_verified_files = st.file_uploader(
            "Upload Verified Rent Roll Files (.xlsx)", 
            type=["xlsx"],
            accept_multiple_files=True
        )
    
    # Optional sheet name selection
    sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
    st.info("Note: Sheet name will be ignored for PDF files.")
    
    # Display file counts
    if uploaded_raw_files and uploaded_verified_files:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Raw files uploaded: {len(uploaded_raw_files)}")
            for file in uploaded_raw_files:
                file_ext = os.path.splitext(file.name)[1].lower()
                file_type = "PDF" if file_ext == '.pdf' else "Excel"
                st.write(f"- {file.name} ({file_type})")
        
        with col2:
            st.write(f"Verified files uploaded: {len(uploaded_verified_files)}")
            for file in uploaded_verified_files:
                st.write(f"- {file.name}")
        
        # Check if counts match
        if len(uploaded_raw_files) != len(uploaded_verified_files):
            st.error(f"Error: Number of raw files ({len(uploaded_raw_files)}) does not match number of verified files ({len(uploaded_verified_files)})")
            st.warning("Please ensure there is one verified file for each raw file.")
        else:
            st.success(f"Files uploaded successfully! {len(uploaded_raw_files)} file pairs detected.")
            
            # Create a temporary directory to store files
            temp_dir = tempfile.mkdtemp()
            
            # Save all uploaded files to the temp directory
            raw_temp_paths = []
            verified_temp_paths = []
            
            for raw_file in uploaded_raw_files:
                file_path = os.path.join(temp_dir, raw_file.name)
                with open(file_path, "wb") as f:
                    f.write(raw_file.getbuffer())
                raw_temp_paths.append(file_path)
            
            for verified_file in uploaded_verified_files:
                file_path = os.path.join(temp_dir, verified_file.name)
                with open(file_path, "wb") as f:
                    f.write(verified_file.getbuffer())
                verified_temp_paths.append(file_path)
            
            # Create output directory for results
            output_dir = os.path.join(temp_dir, "results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Match files based on name similarity
            if st.button("Run Verification"):
                # Match files
                file_pairs = match_files(raw_temp_paths, verified_temp_paths.copy())
                
                if not file_pairs:
                    st.error("Failed to match files. Please check file names.")
                else:
                    # Display matched pairs
                    st.subheader("Matched File Pairs")
                    for raw, verified in file_pairs:
                        st.write(f"- {os.path.basename(raw)} → {os.path.basename(verified)}")
                    
                    # Process files with progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a placeholder for results
                    results_container = st.container()
                    
                    # Process each file pair sequentially
                    results = []
                    for i, (raw_file, verified_file) in enumerate(file_pairs):
                        file_name = os.path.basename(raw_file)
                        status_text.text(f"Processing file {i+1}/{len(file_pairs)}: {file_name}")
                        progress_bar.progress((i) / len(file_pairs))
                        
                        # Create output diff file path
                        output_diff_file = os.path.join(output_dir, f"api_comparison_diff_{file_name}.json")
                        
                        # Process the file pair
                        with st.spinner(f"Processing {file_name}..."):
                            try:
                                # Step 1: Submit the job to the API
                                st.write(f"Submitting {file_name} to API...")
                                job_id = submit_job(raw_file, doc_type="rent_roll", sheet_name=sheet_name)
                                
                                if not job_id:
                                    st.error(f"Failed to submit job to API for {file_name}.")
                                    continue
                                
                                st.write(f"Job submitted successfully. Job ID: {job_id}")
                                
                                # Step 2: Fetch job results with progress updates
                                st.write(f"Fetching results for {file_name}...")
                                file_progress_bar = st.progress(0)
                                file_status_text = st.empty()
                                
                                # Set up retry parameters
                                max_retries = 10
                                retry_delay = 5
                                
                                # Implement retry logic with progress updates
                                api_output = None
                                for retry in range(max_retries):
                                    file_status_text.text(f"Attempt {retry + 1}/{max_retries}: Checking job status...")
                                    file_progress_bar.progress((retry + 1) / max_retries)
                                    
                                    # Call the API to check job status
                                    api_output = fetch_job_results(job_id, max_retries=1, retry_delay=0)
                                    
                                    if api_output is None:
                                        file_status_text.text(f"Attempt {retry + 1}/{max_retries}: API request failed. Retrying...")
                                        time.sleep(retry_delay)
                                        continue
                                    
                                    # Check if job is still processing
                                    if api_output.get('job', {}).get('status') == 'processing':
                                        file_status_text.text(f"Attempt {retry + 1}/{max_retries}: Job is still processing. Waiting...")
                                        time.sleep(retry_delay)
                                        continue
                                    
                                    # Job is complete
                                    file_status_text.text(f"Job completed successfully for {file_name}!")
                                    file_progress_bar.progress(1.0)
                                    break
                                else:
                                    # Loop completed without breaking - max retries reached
                                    st.error(f"Maximum retries reached for {file_name}. Job may still be processing.")
                                    continue
                                
                                if not api_output:
                                    st.error(f"Failed to fetch job results from API for {file_name}.")
                                    continue
                                
                                # Save API output for reference
                                api_output_filename = os.path.join(output_dir, f"api_output_{file_name}.json")
                                with open(api_output_filename, 'w') as f:
                                    json.dump(api_output, f, indent=2)
                                
                                # Step 3: Compare API output with Verified Data
                                st.write(f"Comparing API output with verified data for {file_name}...")
                                accuracy, diffs, merged_df = compare_data(api_output, verified_file)
                                
                                if accuracy is not None:
                                    # Save detailed differences
                                    diff_output = {
                                        "raw_file": raw_file,
                                        "verified_file": verified_file,
                                        "accuracy_percent": accuracy,
                                        "field_mismatches": diffs
                                    }
                                    with open(output_diff_file, 'w') as f:
                                        json.dump(diff_output, f, indent=2, default=str)
                                    
                                    results.append({
                                        "raw_file": raw_file,
                                        "verified_file": verified_file,
                                        "accuracy": accuracy,
                                        "diffs": diffs,
                                        "merged_df": merged_df,
                                        "api_output": api_output
                                    })
                                else:
                                    st.error(f"Comparison failed for {file_name}. Check logs for details.")
                            
                            except Exception as e:
                                st.error(f"Error processing {file_name}: {e}")
                                st.text(traceback.format_exc())
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(file_pairs))
                    
                    # Complete progress bar
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Display results
                    if results:
                        with results_container:
                            st.subheader("Verification Results")
                            
                            # Calculate average accuracy
                            avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
                            st.metric("Average Accuracy Across All Files", f"{avg_accuracy:.2f}%")
                            
                            # Create tabs for each file
                            tabs = st.tabs([os.path.basename(r["raw_file"]) for r in results])
                            
                            for i, (tab, result) in enumerate(zip(tabs, results)):
                                with tab:
                                    st.metric("File Accuracy", f"{result['accuracy']:.2f}%")
                                    
                                    # Calculate per-field accuracy for this file
                                    if result["diffs"]:
                                        field_stats = {}
                                        for field in FIELDS_TO_COMPARE:
                                            field_total = len(result["merged_df"][result["merged_df"]['_merge'] == 'both'])
                                            field_mismatches = len(result["diffs"].get(field, []))
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
                                        for field, mismatches in result["diffs"].items():
                                            if mismatches:
                                                with st.expander(f"{field}: {len(mismatches)} mismatches"):
                                                    for j, mismatch in enumerate(mismatches[:5]):
                                                        st.write(f"**Key:** {mismatch['key']}")
                                                        st.write(f"**API:** '{mismatch['api_value']}', **Verified:** '{mismatch['verified_value']}'")
                                                    if len(mismatches) > 5:
                                                        st.write(f"... and {len(mismatches) - 5} more mismatches")
                                    
                                    # Provide download buttons
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Download for diff details
                                        diff_output = {
                                            "raw_file": os.path.basename(result["raw_file"]),
                                            "verified_file": os.path.basename(result["verified_file"]),
                                            "accuracy_percent": result["accuracy"],
                                            "field_mismatches": result["diffs"]
                                        }
                                        st.download_button(
                                            label="Download Mismatch Details (JSON)",
                                            data=json.dumps(diff_output, indent=2, default=str),
                                            file_name=f"api_comparison_diff_{os.path.basename(result['raw_file'])}.json",
                                            mime="application/json",
                                            key=f"download_diff_{i}"
                                        )
                                    
                                    with col2:
                                        # Download for API output
                                        st.download_button(
                                            label="Download Raw API Output (JSON)",
                                            data=json.dumps(result["api_output"], indent=2, default=str),
                                            file_name=f"api_output_{os.path.basename(result['raw_file'])}.json",
                                            mime="application/json",
                                            key=f"download_api_{i}"
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
