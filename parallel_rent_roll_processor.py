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
import argparse
from typing import Dict, List, Any, Optional, Tuple

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
        print(f"Error converting JSON to CSV: {e}")
        return None

def process_file(file_path, temp_dir, sheet_name=None):
    """Process a single rent roll file using the API with bypassRB=False."""
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    
    try:
        # Submit the job to the API with bypassRB=False (rules-based approach)
        job_id = submit_job(file_path, doc_type="rent_roll", sheet_name=sheet_name, bypass_rb=False)
        
        if not job_id:
            print(f"Failed to submit job to API for {file_name}.")
            return None
        
        print(f"Job submitted successfully. Job ID: {job_id}")
        
        # Fetch the job results
        print(f"Fetching results for {file_name}...")
        api_output = fetch_job_results(job_id, max_retries=50, retry_delay=10)
        
        if not api_output:
            print(f"Failed to fetch job results from API for {file_name}.")
            return None
        
        # Save JSON output
        json_output_path = os.path.join(temp_dir, JSON_OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}.json")
        with open(json_output_path, 'w') as f:
            json.dump(api_output, f, indent=2)
        print(f"Saved JSON output to {json_output_path}")
        
        # Convert to CSV and save
        csv_string = json_to_csv_string(api_output)
        if csv_string:
            csv_output_path = os.path.join(temp_dir, CSV_OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}.csv")
            with open(csv_output_path, 'w') as f:
                f.write(csv_string)
            print(f"Saved CSV output to {csv_output_path}")
        else:
            print(f"No data found in API output for {file_name} to save as CSV.")
        
        return {
            "file_name": file_name,
            "job_id": job_id,
            "json_path": json_output_path,
            "csv_path": csv_output_path if csv_string else None
        }
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        traceback.print_exc()
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

def process_files_in_parallel(file_paths, sheet_name=None, max_workers=5, output_zip_path=None):
    """Process multiple rent roll files in parallel, 5 at a time."""
    # Create a temporary directory to store files and outputs
    temp_dir = tempfile.mkdtemp()
    
    # Create output directories
    os.makedirs(os.path.join(temp_dir, JSON_OUTPUT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, CSV_OUTPUT_FOLDER), exist_ok=True)
    
    print(f"Created temporary directory: {temp_dir}")
    print(f"Processing {len(file_paths)} files in parallel (max {max_workers} at a time)...")
    
    results = []
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, file_path, temp_dir, sheet_name): file_path 
                         for file_path in file_paths}
        
        # Process completed tasks as they finish
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            file_name = os.path.basename(file_path)
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[{i+1}/{len(file_paths)}] Successfully processed: {file_name}")
                else:
                    print(f"[{i+1}/{len(file_paths)}] Failed to process: {file_name}")
            except Exception as e:
                print(f"[{i+1}/{len(file_paths)}] Error processing {file_name}: {e}")
                traceback.print_exc()
    
    # Create a ZIP file with all results
    zip_buffer = create_zip_file(temp_dir)
    
    # Save the ZIP file if a path is provided
    if output_zip_path:
        with open(output_zip_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        print(f"Saved ZIP file to {output_zip_path}")
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total files processed: {len(file_paths)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(file_paths) - len(results)}")
    
    # Return the ZIP buffer for potential further use
    return zip_buffer, results

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Process rent roll files in parallel using the rules-based API approach.')
    parser.add_argument('files', nargs='+', help='Paths to rent roll files (.xlsx or .pdf)')
    parser.add_argument('--sheet-name', help='Sheet name for Excel files (optional)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel workers (default: 5)')
    parser.add_argument('--output-zip', default='rent_roll_results.zip', help='Path for the output ZIP file (default: rent_roll_results.zip)')
    
    args = parser.parse_args()
    
    # Process the files
    zip_buffer, results = process_files_in_parallel(
        args.files,
        sheet_name=args.sheet_name,
        max_workers=args.max_workers,
        output_zip_path=args.output_zip
    )
    
    print(f"\nProcessing complete. Results saved to {args.output_zip}")

if __name__ == "__main__":
    main()
