import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback

# Import the main script functions but don't run the main function
from rent_roll_verifier import (
    extract_rent_roll_string,
    query_llm,
    compare_data,
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

    st.info("Upload the raw rent roll (.xlsx) and the verified version (.xlsx) to compare LLM parsing accuracy.")

    uploaded_raw_file = st.file_uploader("1. Upload Raw Rent Roll (.xlsx or .pdf)", type=["xlsx", "pdf"])
    uploaded_verified_file = st.file_uploader("2. Upload Verified Rent Roll (.xlsx)", type="xlsx")

    # Display file type information
    if uploaded_raw_file is not None:
        file_ext = os.path.splitext(uploaded_raw_file.name)[1].lower()
        if file_ext == '.pdf':
            st.info("PDF file detected. Data will be extracted from the PDF file.")

    if uploaded_raw_file and uploaded_verified_file:
        st.success("Files uploaded successfully!")

        # Save temporary files to run the main logic
        raw_temp_path = f"temp_{uploaded_raw_file.name}"
        verified_temp_path = f"temp_{uploaded_verified_file.name}"
        llm_output_filename = f"temp_llm_output_{uploaded_raw_file.name}.json"

        with open(raw_temp_path, "wb") as f:
            f.write(uploaded_raw_file.getbuffer())
        with open(verified_temp_path, "wb") as f:
            f.write(uploaded_verified_file.getbuffer())

        if st.button("Run Verification"):
            with st.spinner("Processing files and querying LLMs... This may take a minute."):
                # --- Run the main logic within Streamlit context ---
                rr_string_st = extract_rent_roll_string(raw_temp_path)
                if not rr_string_st:
                    st.error("Failed to read the raw rent roll file.")
                else:
                    try:
                        portkey_st = Portkey(api_key=PORTKEY_API_KEY)

                        # Query Gemini for first part (previously Claude)
                        st.write("Querying Gemini for first part...")
                        claude_model = "gemini-2.5-flash-preview-04-17"
                        claude_prompt = PROMPT_PART_1_CLAUDE + rr_string_st
                        claude_response_text = query_llm(portkey_st, claude_prompt, GEMINI_VIRTUAL_KEY, claude_model, "google")
                        
                        if claude_response_text is not None and isinstance(claude_response_text, str) and claude_response_text:
                            st.write("Gemini first part query successful.")
                        else:
                            st.warning("Gemini first part query returned empty or invalid response. Proceeding with second Gemini query only.")

                        # Query Gemini for second part
                        st.write("Querying Gemini for second part...")
                        gemini_model = "gemini-2.5-flash-preview-04-17"
                        gemini_prompt = PROMPT_PART_2_GEMINI + rr_string_st
                        gemini_json_output_st = query_llm(portkey_st, gemini_prompt, GEMINI_VIRTUAL_KEY, gemini_model, "google")

                        if not gemini_json_output_st:
                            st.error("Failed to get valid JSON output from Gemini.")
                        else:
                            st.write("Gemini query successful. Comparing data...")
                            
                            # Save LLM output for reference
                            with open(llm_output_filename, 'w') as f:
                                json.dump(gemini_json_output_st, f, indent=2)
                            
                            # Compare
                            accuracy_st, diffs_st, merged_df_st = compare_data(gemini_json_output_st, verified_temp_path)

                            if accuracy_st is not None:
                                st.subheader("Verification Results")
                                st.metric("Overall Accuracy", f"{accuracy_st:.2f}%")

                                # Calculate per-field accuracy for display
                                if diffs_st:
                                    field_stats = {}
                                    for field in FIELDS_TO_COMPARE:
                                        field_total = len(merged_df_st[merged_df_st['_merge'] == 'both'])
                                        field_mismatches = len(diffs_st.get(field, []))
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
                                    for field, mismatches in diffs_st.items():
                                        if mismatches:
                                            with st.expander(f"{field}: {len(mismatches)} mismatches"):
                                                for i, mismatch in enumerate(mismatches[:5]):
                                                    st.write(f"**Key:** {mismatch['key']}")
                                                    st.write(f"**LLM:** '{mismatch['llm_value']}', **Verified:** '{mismatch['verified_value']}'")
                                                if len(mismatches) > 5:
                                                    st.write(f"... and {len(mismatches) - 5} more mismatches")
                                
                                # Provide download for full diff details
                                diff_output_st = {
                                    "accuracy_percent": accuracy_st,
                                    "field_mismatches": diffs_st
                                }
                                st.download_button(
                                    label="Download Mismatch Details (JSON)",
                                    data=json.dumps(diff_output_st, indent=2, default=str),
                                    file_name="llm_comparison_diffs.json",
                                    mime="application/json",
                                )

                                # Display side-by-side comparison (optional, can be large)
                                if merged_df_st is not None and st.checkbox("Show Detailed Comparison Table"):
                                    st.dataframe(merged_df_st)

                            else:
                                st.error("Comparison failed. Check logs or input files.")

                    except Exception as e:
                        st.error(f"An error occurred during verification: {e}")
                        st.text(traceback.format_exc())

            # Clean up temp files
            try:
                # Only try to remove files that exist
                if os.path.exists(raw_temp_path):
                    os.remove(raw_temp_path)
                if os.path.exists(verified_temp_path):
                    os.remove(verified_temp_path)
                if os.path.exists(llm_output_filename):
                    os.remove(llm_output_filename)
                
                # Also clean up any debug files that might have been created
                debug_files = [
                    "raw_google_response.txt",
                    "fixed_json.txt",
                    "fixed_json_aggressive.txt",
                    "error_google_response.txt"
                ]
                for file in debug_files:
                    if os.path.exists(file):
                        os.remove(file)
                        
            except OSError as e:
                st.warning(f"Could not remove some temporary files: {e}")

if __name__ == "__main__":
    run_streamlit_app()
