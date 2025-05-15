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
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

# Does the API include whether LLM or RB was used in metadata?0

# Import necessary functions from api_rent_roll_verifier
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    API_BASE_URL,
    API_KEY
)

# Define folder names
JSON_OUTPUT_FOLDER = "JSON_OUTPUT"
CSV_OUTPUT_FOLDER = "CSV_OUTPUT"

# Initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

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

def process_file(file_path, temp_dir, sheet_name=None):
    """Process a single rent roll file using the API with bypassRB=False."""
    file_name = os.path.basename(file_path)
    st.write(f"Processing file: {file_name}")
    
    try:
        # Submit the job to the API with bypassRB=False (rules-based approach)
        job_id = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=False)
        
        if not job_id:
            st.error(f"Failed to submit job to API for {file_name}.")
            return None
        
        st.write(f"Job submitted successfully. Job ID: {job_id}")
        
        # Fetch the job results
        st.write(f"Fetching results for {file_name}...")
        api_output = fetch_job_results(job_id, max_retries=500, retry_delay=10)
        
        if not api_output:
            st.error(f"Failed to fetch job results from API for {file_name}.")
            return None
        
        # Check if the API fell back to LLM
        used_llm = False
        if isinstance(api_output, dict) and api_output.get("metadata") and api_output["metadata"].get("used_llm"):
            used_llm = True
            st.warning(f"API fell back to LLM for {file_name}")
        
        # Modify file name if LLM was used
        output_base_name = os.path.splitext(file_name)[0]
        if used_llm:
            output_base_name += "_LLM"
        
        # Save JSON output
        json_output_path = os.path.join(temp_dir, JSON_OUTPUT_FOLDER, f"{output_base_name}.json")
        with open(json_output_path, 'w') as f:
            json.dump(api_output, f, indent=2)
        st.write(f"Saved JSON output to {json_output_path}")
        
        # Convert to CSV and save
        csv_string = json_to_csv_string(api_output)
        if csv_string:
            csv_output_path = os.path.join(temp_dir, CSV_OUTPUT_FOLDER, f"{output_base_name}.csv")
            with open(csv_output_path, 'w') as f:
                f.write(csv_string)
            st.write(f"Saved CSV output to {csv_output_path}")
        else:
            st.warning(f"No data found in API output for {file_name} to save as CSV.")
            csv_output_path = None
        
        return {
            "file_name": file_name,
            "job_id": job_id,
            "json_path": json_output_path,
            "csv_path": csv_output_path
        }
    
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        st.text(traceback.format_exc())
        return None

def create_zip_file(temp_dir):
    """Create a ZIP file containing the JSON and CSV output folders."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all files from the output folders
        for folder_path in [os.path.join(temp_dir, JSON_OUTPUT_FOLDER), 
                           os.path.join(temp_dir, CSV_OUTPUT_FOLDER)]:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to zip, preserving folder structure relative to temp_dir
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, arcname)
    
    # Reset buffer position to the beginning
    zip_buffer.seek(0)
    return zip_buffer

def process_files_in_parallel(file_paths, temp_dir, sheet_name=None, max_workers=5):
    """Process multiple rent roll files in parallel, 5 at a time."""
    st.write(f"Processing {len(file_paths)} files in parallel (max {max_workers} at a time)...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, file_path, temp_dir, sheet_name): file_path 
                         for file_path in file_paths}
        
        # Process completed tasks as they finish
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            file_name = os.path.basename(file_path)
            try:
                result = future.result()
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
            progress_bar.progress(completed / len(file_paths))
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Print summary
    st.subheader("Processing Summary")
    st.write(f"Total files processed: {len(file_paths)}")
    st.write(f"Successful: {len(results)}")
    st.write(f"Failed: {len(file_paths) - len(results)}")
    
    return results

def run_streamlit_app():
    """Main Streamlit app function."""
    st.set_page_config(layout="wide")
    st.title("Parallel Rent Roll Processor")
    
    st.info("Upload rent roll files (.xlsx or .pdf) to process them in parallel using the rules-based API approach. Results will be saved as JSON and CSV files in a downloadable ZIP.")
    
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
    st.subheader("Upload Rent Roll Files")
    uploaded_files = st.file_uploader(
        "Upload Rent Roll Files (.xlsx or .pdf)",
        type=["xlsx", "pdf"],
        accept_multiple_files=True
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
        st.info("Note: Sheet name will be ignored for PDF files.")
    
    with col2:
        max_workers = st.slider("Maximum Parallel Workers", min_value=1, max_value=10, value=5)
        st.info("Number of files to process simultaneously.")
    
    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_type = "PDF" if file_ext == '.pdf' else "Excel"
            st.write(f"- {file.name} ({file_type})")
        
        if st.button("Process Files") or st.session_state.processing_complete:
            # Create a temporary directory to store files and outputs if not already created
            if not st.session_state.temp_dir:
                st.session_state.temp_dir = tempfile.mkdtemp()
                
                # Create output directories within the temporary directory
                os.makedirs(os.path.join(st.session_state.temp_dir, JSON_OUTPUT_FOLDER), exist_ok=True)
                os.makedirs(os.path.join(st.session_state.temp_dir, CSV_OUTPUT_FOLDER), exist_ok=True)
                
                st.write(f"Created temporary directory: {st.session_state.temp_dir}")
            
            # Use session state results if available, otherwise process files
            if st.session_state.processing_complete:
                results = st.session_state.results
            else:
                # Save uploaded files to the temp directory
                temp_paths = []
                for file in uploaded_files:
                    file_path = os.path.join(st.session_state.temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    temp_paths.append(file_path)
                
                # Process files in parallel
                results = process_files_in_parallel(
                    temp_paths,
                    st.session_state.temp_dir,
                    sheet_name=sheet_name if sheet_name else None,
                    max_workers=max_workers
                )
                
                # Save results to session state
                st.session_state.results = results
                st.session_state.processing_complete = True
            
            # Create a ZIP file with all results
            if not st.session_state.zip_buffer:
                st.session_state.zip_buffer = create_zip_file(st.session_state.temp_dir)
            
            # Add a download button for the ZIP file
            # Use a unique key based on a timestamp to prevent rerun issues
            download_key = f"download_results_{int(time.time())}"
            st.download_button(
                label="Download All Results as ZIP",
                data=st.session_state.zip_buffer,
                file_name="rent_roll_results.zip",
                mime="application/zip",
                key=download_key
            )
            
            # Display results
            if results:
                st.subheader("Processed Files")
                for result in results:
                    with st.expander(f"File: {result['file_name']}"):
                        st.write(f"Job ID: {result['job_id']}")
                        st.write(f"JSON Output: {result['json_path']}")
                        if result['csv_path']:
                            st.write(f"CSV Output: {result['csv_path']}")
                        else:
                            st.warning("No CSV output generated (possibly no data in API response).")
            else:
                st.error("No files were successfully processed. Check the logs for errors.")

if __name__ == "__main__":
    run_streamlit_app()
