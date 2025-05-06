import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback
import tempfile
import time
import concurrent.futures

# Import the main script functions but don't run the main function
from rent_roll_verifier import (
    process_single_file,
    match_files,
    PROMPT_PART_1_CLAUDE,
    PROMPT_PART_2_GEMINI,
    PORTKEY_API_KEY,
    GEMINI_VIRTUAL_KEY,
    CLAUDE_VIRTUAL_KEY,
    FIELDS_TO_COMPARE,
    Portkey
)

def run_streamlit_app():
    """Main Streamlit app function"""
    st.set_page_config(layout="wide")
    st.title("Rent Roll LLM Verifier")

    st.info("Upload raw rent roll files (.xlsx or .pdf) and their corresponding verified versions (.xlsx) to compare LLM parsing accuracy.")

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
    
    # Display file counts
    if uploaded_raw_files and uploaded_verified_files:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Raw files uploaded: {len(uploaded_raw_files)}")
            for file in uploaded_raw_files:
                st.write(f"- {file.name}")
        
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
                    
                    # Initialize Portkey client once for all files
                    try:
                        portkey = Portkey(api_key=PORTKEY_API_KEY)
                        
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
                            output_diff_file = os.path.join(output_dir, f"comparison_diff_{file_name}.json")
                            
                            # Process the file pair
                            with st.spinner(f"Processing {file_name}..."):
                                accuracy, diffs, merged_df = process_single_file(
                                    raw_file, 
                                    verified_file, 
                                    output_diff_file,
                                    portkey
                                )
                            
                            if accuracy is not None:
                                results.append({
                                    "raw_file": raw_file,
                                    "verified_file": verified_file,
                                    "accuracy": accuracy,
                                    "diffs": diffs,
                                    "merged_df": merged_df
                                })
                            
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
                                                        for i, mismatch in enumerate(mismatches[:5]):
                                                            st.write(f"**Unit:** {mismatch['key']}")
                                                            st.write(f"**LLM:** '{mismatch['llm_value']}', **Verified:** '{mismatch['verified_value']}'")
                                                        if len(mismatches) > 5:
                                                            st.write(f"... and {len(mismatches) - 5} more mismatches")
                                        
                                        # Provide download for full diff details
                                        diff_output = {
                                            "raw_file": os.path.basename(result["raw_file"]),
                                            "verified_file": os.path.basename(result["verified_file"]),
                                            "accuracy_percent": result["accuracy"],
                                            "field_mismatches": result["diffs"]
                                        }
                                        st.download_button(
                                            label="Download Mismatch Details (JSON)",
                                            data=json.dumps(diff_output, indent=2, default=str),
                                            file_name=f"comparison_diff_{os.path.basename(result['raw_file'])}.json",
                                            mime="application/json",
                                            key=f"download_{i}"
                                        )
                                        
                                        # Display side-by-side comparison (optional, can be large)
                                        if result["merged_df"] is not None and st.checkbox("Show Detailed Comparison Table", key=f"show_table_{i}"):
                                            st.dataframe(result["merged_df"])
                        else:
                            st.error("No successful comparisons were completed. Check the logs for errors.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during verification: {e}")
                        st.text(traceback.format_exc())
                
                # Clean up temp files
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    st.warning(f"Could not remove temporary files: {e}")

if __name__ == "__main__":
    run_streamlit_app()
