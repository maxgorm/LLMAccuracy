import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback
import tempfile
import time
import copy

#This file is used to compare the outputs of 2 LLMs on a given RR using Thejan's API

# Import the API script functions
from api_rent_roll_verifier import (
    submit_job,
    fetch_job_results,
    normalize_value,
    compare_data,
    FIELDS_TO_COMPARE,
    API_BASE_URL,
    API_KEY
)

def calculate_total_cost(api_output):
    """Calculate total cost from llm_token_info array in API output."""
    if not api_output or not isinstance(api_output, dict):
        return 0.0, []
    
    # Check if we have job data
    job_data = api_output.get('job', {})
    if not isinstance(job_data, dict):
        return 0.0, []
    
    # Try to get the pre-calculated total_cost first
    total_cost = job_data.get('total_cost', 0.0)
    
    # Get the llm_token_info from the job object
    llm_token_info = job_data.get('llm_token_info', [])
    if not llm_token_info or not isinstance(llm_token_info, list):
        return float(total_cost), []
    
    # Create breakdown from individual token info entries
    cost_breakdown = []
    calculated_total = 0.0
    
    for token_info in llm_token_info:
        if isinstance(token_info, dict) and 'cost' in token_info:
            cost = float(token_info.get('cost', 0))
            calculated_total += cost
            
            # Create breakdown entry
            breakdown_entry = {
                'model': token_info.get('model', 'Unknown'),
                'cost': cost,
                'in_tokens': token_info.get('in_tokens', 0),
                'out_tokens': token_info.get('out_tokens', 0)
            }
            cost_breakdown.append(breakdown_entry)
    
    # Use the pre-calculated total if available, otherwise use our calculated total
    final_total = float(total_cost) if total_cost > 0 else calculated_total
    
    return final_total, cost_breakdown

def calculate_adjusted_batch_costs(api_output1, api_output2, api_output3, n_batch_calls=3):
    """
    Calculate adjusted costs for batch processing where only the slave LLM is run multiple times.
    The master LLM cost should only be counted once, while slave LLM cost is counted n_batch_calls times.
    """
    # Get individual costs and breakdowns
    cost1, breakdown1 = calculate_total_cost(api_output1)
    cost2, breakdown2 = calculate_total_cost(api_output2)
    cost3, breakdown3 = calculate_total_cost(api_output3)
    
    # Group costs by model type to identify master vs slave LLMs
    all_models = {}
    
    # Collect all model costs from all API calls
    for i, breakdown in enumerate([breakdown1, breakdown2, breakdown3], 1):
        for entry in breakdown:
            model = entry['model']
            if model not in all_models:
                all_models[model] = {
                    'costs': [],
                    'total_cost': 0,
                    'in_tokens': entry['in_tokens'],
                    'out_tokens': entry['out_tokens']
                }
            all_models[model]['costs'].append(entry['cost'])
            all_models[model]['total_cost'] += entry['cost']
    
    # Identify master and slave LLMs based on naming patterns
    # Master LLMs typically have names like claude-sonnet, claude-opus, etc.
    # Slave LLMs typically have names like gemini, gpt, etc.
    master_models = []
    slave_models = []
    
    for model in all_models.keys():
        model_lower = model.lower()
        if 'claude' in model_lower and ('sonnet' in model_lower or 'opus' in model_lower):
            master_models.append(model)
        else:
            slave_models.append(model)
    
    # Calculate adjusted costs
    adjusted_breakdown1 = []
    adjusted_breakdown2 = []
    adjusted_breakdown3 = []
    adjusted_total = 0
    
    # For master LLMs: only count cost from first API call
    for model in master_models:
        if model in [entry['model'] for entry in breakdown1]:
            # Find the entry in breakdown1
            for entry in breakdown1:
                if entry['model'] == model:
                    adjusted_breakdown1.append(entry)
                    adjusted_total += entry['cost']
                    break
            
            # Add zero-cost entries for API calls 2 and 3 to show they weren't charged
            adjusted_breakdown2.append({
                'model': model,
                'cost': 0.0,
                'in_tokens': 0,
                'out_tokens': 0
            })
            adjusted_breakdown3.append({
                'model': model,
                'cost': 0.0,
                'in_tokens': 0,
                'out_tokens': 0
            })
    
    # For slave LLMs: count cost from all API calls
    for model in slave_models:
        for i, breakdown in enumerate([breakdown1, breakdown2, breakdown3], 1):
            for entry in breakdown:
                if entry['model'] == model:
                    if i == 1:
                        adjusted_breakdown1.append(entry)
                    elif i == 2:
                        adjusted_breakdown2.append(entry)
                    else:
                        adjusted_breakdown3.append(entry)
                    adjusted_total += entry['cost']
    
    # Calculate individual API costs for display
    adjusted_cost1 = sum(entry['cost'] for entry in adjusted_breakdown1)
    adjusted_cost2 = sum(entry['cost'] for entry in adjusted_breakdown2)
    adjusted_cost3 = sum(entry['cost'] for entry in adjusted_breakdown3)
    
    return {
        'api1_cost': adjusted_cost1,
        'api1_cost_breakdown': adjusted_breakdown1,
        'api2_cost': adjusted_cost2,
        'api2_cost_breakdown': adjusted_breakdown2,
        'api3_cost': adjusted_cost3,
        'api3_cost_breakdown': adjusted_breakdown3,
        'total_cost': adjusted_total
    }

def generate_majority_consensus_output(api_output1, api_output2, api_output3):
    """Generate a synthetic API output representing the majority consensus from 3-way comparison."""
    
    # Check if API outputs have the expected structure
    for i, api_output in enumerate([api_output1, api_output2, api_output3], 1):
        if not api_output or "df" not in api_output or not api_output["df"]:
            st.error(f"Error: API output {i} is empty or not in the expected format.")
            return None
    
    # Convert API outputs to DataFrames
    df_api1 = pd.DataFrame(api_output1["df"])
    df_api2 = pd.DataFrame(api_output2["df"])
    df_api3 = pd.DataFrame(api_output3["df"])
    
    # Normalize column names
    for df in [df_api1, df_api2, df_api3]:
        df.columns = df.columns.str.strip().str.lower()
    
    # Check for missing columns and add them
    for i, df in enumerate([df_api1, df_api2, df_api3], 1):
        missing_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                df[col] = pd.NA
    
    # --- Data Type Conversion & Cleaning ---
    date_cols = ['move_in', 'lease_start', 'lease_end']
    numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
    
    for df in [df_api1, df_api2, df_api3]:
        # Convert date columns
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df[col] = pd.NA
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle boolean 'is_mtm'
        if 'is_mtm' in df.columns:
            df['is_mtm'] = df['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df['is_mtm'] = pd.to_numeric(df['is_mtm'], errors='coerce').fillna(0).astype(int)
    
    # --- Construct Match Key ---
    for df in [df_api1, df_api2, df_api3]:
        unit_col = df.get('unit_num', pd.Series(dtype=str))
        df['match_key'] = unit_col.fillna('').astype(str).str.strip().str.lower()
    
    # --- Three-way Merge ---
    # First merge API1 and API2
    merged_12 = df_api1.merge(df_api2, on="match_key", suffixes=("_api1", "_api2"), how="outer")
    # Then merge with API3
    merged_all = merged_12.merge(df_api3, on="match_key", suffixes=("", "_api3"), how="outer")
    
    # Handle suffix conflicts for API3 columns
    for col in FIELDS_TO_COMPARE:
        if col in merged_all.columns and f"{col}_api3" not in merged_all.columns:
            merged_all[f"{col}_api3"] = merged_all[col]
            if f"{col}_api1" in merged_all.columns:  # Only drop if we have the suffixed version
                merged_all = merged_all.drop(columns=[col])
    
    matched_rows = merged_all.dropna(subset=['match_key'])
    
    # Create majority consensus DataFrame
    majority_data = []
    
    for _, row in matched_rows.iterrows():
        majority_row = {}
        
        for field in FIELDS_TO_COMPARE:
            field_api1 = f"{field}_api1"
            field_api2 = f"{field}_api2"
            field_api3 = f"{field}_api3"
            
            val_api1 = row.get(field_api1)
            val_api2 = row.get(field_api2)
            val_api3 = row.get(field_api3)
            
            # Normalize values for comparison
            norm_vals = []
            raw_vals = [val_api1, val_api2, val_api3]
            
            # Import the field-specific normalization function
            from api_rent_roll_verifier import normalize_field_value
            
            for val in raw_vals:
                if field == 'is_mtm':
                    norm_vals.append(int(val) if not pd.isna(val) else 0)
                elif field in numeric_cols:
                    if pd.isna(val):
                        norm_vals.append(0)  # Treat NaN as 0 for comparison
                    else:
                        norm_vals.append(float(val))
                else:
                    norm_vals.append(normalize_field_value(field, val))
            
            # Determine majority value
            unique_vals = list(set(norm_vals))
            
            if len(unique_vals) == 1:
                # Perfect agreement - use any value (they're all the same)
                majority_row[field] = val_api1
            elif len(unique_vals) == 2:
                # Majority agreement - find the value that appears twice
                val_counts = {val: norm_vals.count(val) for val in unique_vals}
                majority_norm_val = max(val_counts.keys(), key=lambda x: val_counts[x])
                
                # Find the original raw value that corresponds to the majority normalized value
                for i, norm_val in enumerate(norm_vals):
                    if norm_val == majority_norm_val:
                        majority_row[field] = raw_vals[i]
                        break
            else:
                # Complete disagreement - use the first value as default
                # In a real scenario, you might want to handle this differently
                majority_row[field] = val_api1
        
        majority_data.append(majority_row)
    
    # Create synthetic API output in the same format as the original
    majority_api_output = {
        "df": majority_data,
        "job": {
            "status": "completed",
            "total_cost": 0.0,  # No cost for synthetic output
            "llm_token_info": []
        }
    }
    
    return majority_api_output

def compare_api_outputs_3way(api_output1, api_output2, api_output3):
    """Compares three API outputs and returns the differences with consensus analysis."""
    
    # Check if API outputs have the expected structure
    for i, api_output in enumerate([api_output1, api_output2, api_output3], 1):
        if not api_output or "df" not in api_output or not api_output["df"]:
            st.error(f"Error: API output {i} is empty or not in the expected format.")
            return None
    
    # Convert API outputs to DataFrames
    df_api1 = pd.DataFrame(api_output1["df"])
    df_api2 = pd.DataFrame(api_output2["df"])
    df_api3 = pd.DataFrame(api_output3["df"])
    
    # Normalize column names
    for df in [df_api1, df_api2, df_api3]:
        df.columns = df.columns.str.strip().str.lower()
    
    # Check for missing columns and add them
    for i, df in enumerate([df_api1, df_api2, df_api3], 1):
        missing_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df.columns)
        if missing_cols:
            st.warning(f"Warning: Missing expected columns in API output {i}: {missing_cols}")
            for col in missing_cols:
                df[col] = pd.NA
    
    # --- Data Type Conversion & Cleaning ---
    date_cols = ['move_in', 'lease_start', 'lease_end']
    numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
    
    for df in [df_api1, df_api2, df_api3]:
        # Convert date columns
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df[col] = pd.NA
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('nan', pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle boolean 'is_mtm'
        if 'is_mtm' in df.columns:
            df['is_mtm'] = df['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df['is_mtm'] = pd.to_numeric(df['is_mtm'], errors='coerce').fillna(0).astype(int)
    
    # --- Construct Match Key ---
    for df in [df_api1, df_api2, df_api3]:
        unit_col = df.get('unit_num', pd.Series(dtype=str))
        df['match_key'] = unit_col.fillna('').astype(str).str.strip().str.lower()
    
    # --- Three-way Merge ---
    # First merge API1 and API2
    merged_12 = df_api1.merge(df_api2, on="match_key", suffixes=("_api1", "_api2"), how="outer")
    # Then merge with API3
    merged_all = merged_12.merge(df_api3, on="match_key", suffixes=("", "_api3"), how="outer")
    
    # Handle suffix conflicts for API3 columns
    for col in FIELDS_TO_COMPARE:
        if col in merged_all.columns and f"{col}_api3" not in merged_all.columns:
            merged_all[f"{col}_api3"] = merged_all[col]
            if f"{col}_api1" in merged_all.columns:  # Only drop if we have the suffixed version
                merged_all = merged_all.drop(columns=[col])
    
    matched_rows = merged_all.dropna(subset=['match_key'])
    
    st.write(f"--- 3-Way Comparison Results ---")
    st.write(f"Total Rows Found: {len(matched_rows)}")
    
    # Initialize counters
    total_comparisons = 0
    perfect_agreement = 0  # All 3 match
    majority_agreement = 0  # 2 out of 3 match
    complete_disagreement = 0  # All 3 different
    
    # Track consensus patterns
    consensus_data = {}
    field_consensus_stats = {}
    api_outlier_count = {"api1": 0, "api2": 0, "api3": 0}
    
    for _, row in matched_rows.iterrows():
        for field in FIELDS_TO_COMPARE:
            field_api1 = f"{field}_api1"
            field_api2 = f"{field}_api2"
            field_api3 = f"{field}_api3"
            
            val_api1 = row.get(field_api1)
            val_api2 = row.get(field_api2)
            val_api3 = row.get(field_api3)
            
            # Normalize values for comparison
            norm_vals = []
            raw_vals = [val_api1, val_api2, val_api3]
            
            for val in raw_vals:
                if field == 'is_mtm':
                    norm_vals.append(int(val) if not pd.isna(val) else 0)
                elif field in numeric_cols:
                    if pd.isna(val):
                        norm_vals.append(0)  # Treat NaN as 0 for comparison
                    else:
                        norm_vals.append(float(val))
                else:
                    norm_vals.append(normalize_value(val))
            
            # Determine consensus pattern
            unique_vals = list(set(norm_vals))
            total_comparisons += 1
            
            if len(unique_vals) == 1:
                # Perfect agreement - all 3 match
                perfect_agreement += 1
                consensus_type = "perfect"
                outlier_api = None
            elif len(unique_vals) == 2:
                # Majority agreement - 2 match, 1 differs
                majority_agreement += 1
                consensus_type = "majority"
                # Find the outlier
                val_counts = {val: norm_vals.count(val) for val in unique_vals}
                outlier_val = min(val_counts.keys(), key=lambda x: val_counts[x])
                outlier_idx = norm_vals.index(outlier_val)
                outlier_api = f"api{outlier_idx + 1}"
                api_outlier_count[outlier_api] += 1
            else:
                # Complete disagreement - all 3 different
                complete_disagreement += 1
                consensus_type = "disagreement"
                outlier_api = None
            
            # Store detailed mismatch info
            if consensus_type != "perfect":
                if field not in consensus_data:
                    consensus_data[field] = []
                
                consensus_data[field].append({
                    "key": row["match_key"],
                    "api1_value": val_api1,
                    "api2_value": val_api2,
                    "api3_value": val_api3,
                    "consensus_type": consensus_type,
                    "outlier_api": outlier_api
                })
            
            # Track field-level consensus stats
            if field not in field_consensus_stats:
                field_consensus_stats[field] = {"perfect": 0, "majority": 0, "disagreement": 0, "total": 0}
            
            field_consensus_stats[field][consensus_type] += 1
            field_consensus_stats[field]["total"] += 1
    
    # Calculate accuracy metrics
    perfect_accuracy = round(100 * perfect_agreement / total_comparisons, 2) if total_comparisons > 0 else 0
    majority_accuracy = round(100 * (perfect_agreement + majority_agreement) / total_comparisons, 2) if total_comparisons > 0 else 0
    
    st.write(f"\n--- Consensus Analysis ---")
    st.write(f"Perfect Agreement (all 3 match): {perfect_accuracy}% ({perfect_agreement}/{total_comparisons} fields)")
    st.write(f"Majority Consensus (2+ match): {majority_accuracy}% ({perfect_agreement + majority_agreement}/{total_comparisons} fields)")
    st.write(f"Complete Disagreement: {round(100 * complete_disagreement / total_comparisons, 2)}% ({complete_disagreement}/{total_comparisons} fields)")
    
    st.write(f"\n--- Outlier Analysis ---")
    for api, count in api_outlier_count.items():
        percentage = round(100 * count / majority_agreement, 2) if majority_agreement > 0 else 0
        st.write(f"{api.upper()} was the outlier: {count} times ({percentage}% of majority cases)")
    
    # Calculate adjusted costs for batch processing
    adjusted_costs = calculate_adjusted_batch_costs(api_output1, api_output2, api_output3)
    
    return {
        "perfect_accuracy": perfect_accuracy,
        "majority_accuracy": majority_accuracy,
        "consensus_data": consensus_data,
        "field_consensus_stats": field_consensus_stats,
        "api_outlier_count": api_outlier_count,
        "merged_df": merged_all,
        "total_comparisons": total_comparisons,
        "perfect_agreement": perfect_agreement,
        "majority_agreement": majority_agreement,
        "complete_disagreement": complete_disagreement,
        "api1_cost": adjusted_costs["api1_cost"],
        "api1_cost_breakdown": adjusted_costs["api1_cost_breakdown"],
        "api2_cost": adjusted_costs["api2_cost"],
        "api2_cost_breakdown": adjusted_costs["api2_cost_breakdown"],
        "api3_cost": adjusted_costs["api3_cost"],
        "api3_cost_breakdown": adjusted_costs["api3_cost_breakdown"],
        "total_cost": adjusted_costs["total_cost"]
    }

def compare_api_outputs(api_output1, api_output2):
    """Compares two API outputs and returns the differences."""
    
    # Check if API outputs have the expected structure
    if not api_output1 or "df" not in api_output1 or not api_output1["df"]:
        st.error("Error: First API output is empty or not in the expected format.")
        return None, None, None, None
    
    if not api_output2 or "df" not in api_output2 or not api_output2["df"]:
        st.error("Error: Second API output is empty or not in the expected format.")
        return None, None, None, None
    
    # Convert API outputs to DataFrames
    df_api1 = pd.DataFrame(api_output1["df"])
    df_api2 = pd.DataFrame(api_output2["df"])
    
    # Normalize column names
    df_api1.columns = df_api1.columns.str.strip().str.lower()
    df_api2.columns = df_api2.columns.str.strip().str.lower()
    
    # Check for missing columns
    missing_api1_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_api1.columns)
    missing_api2_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_api2.columns)
    
    if missing_api1_cols:
        st.warning(f"Warning: Missing expected columns in first API output: {missing_api1_cols}")
        # Add missing columns with NaN to allow merge/comparison
        for col in missing_api1_cols:
            df_api1[col] = pd.NA
    
    if missing_api2_cols:
        st.warning(f"Warning: Missing expected columns in second API output: {missing_api2_cols}")
        # Add missing columns with NaN to allow merge/comparison
        for col in missing_api2_cols:
            df_api2[col] = pd.NA
    
    # --- Data Type Conversion & Cleaning ---
    # Convert date columns safely, coercing errors to NaT
    date_cols = ['move_in', 'lease_start', 'lease_end']
    
    for col in date_cols:
        if col in df_api1.columns:
            # Handle 'nan' strings before conversion
            df_api1[col] = df_api1[col].replace('nan', pd.NA)
            df_api1[col] = pd.to_datetime(df_api1[col], errors='coerce').dt.strftime('%m/%d/%Y')
        else:
            df_api1[col] = pd.NA  # Ensure column exists if missing
        
        if col in df_api2.columns:
            # Handle 'nan' strings before conversion
            df_api2[col] = df_api2[col].replace('nan', pd.NA)
            df_api2[col] = pd.to_datetime(df_api2[col], errors='coerce').dt.strftime('%m/%d/%Y')
        else:
            df_api2[col] = pd.NA  # Ensure column exists if missing
    
    # Convert numeric columns safely, coercing errors to NaN
    numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
    for col in numeric_cols:
        if col in df_api1.columns:
            df_api1[col] = df_api1[col].replace('nan', pd.NA)
            df_api1[col] = pd.to_numeric(df_api1[col], errors='coerce')
        if col in df_api2.columns:
            df_api2[col] = df_api2[col].replace('nan', pd.NA)
            df_api2[col] = pd.to_numeric(df_api2[col], errors='coerce')
    
    # Handle boolean 'is_mtm' (normalize 0/1, True/False, 'true'/'false' to 0/1)
    if 'is_mtm' in df_api1.columns:
        df_api1['is_mtm'] = df_api1['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
        df_api1['is_mtm'] = pd.to_numeric(df_api1['is_mtm'], errors='coerce').fillna(0).astype(int)  # Default to 0 (False) if conversion fails or missing
    if 'is_mtm' in df_api2.columns:
        df_api2['is_mtm'] = df_api2['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
        df_api2['is_mtm'] = pd.to_numeric(df_api2['is_mtm'], errors='coerce').fillna(0).astype(int)
    
    # --- Construct Match Key ---
    # Use unit_num as the match key
    unit_col_1 = df_api1.get('unit_num', pd.Series(dtype=str))  # Get column or empty series
    df_api1['match_key'] = unit_col_1.fillna('').astype(str).str.strip().str.lower()
    
    unit_col_2 = df_api2.get('unit_num', pd.Series(dtype=str))
    df_api2['match_key'] = unit_col_2.fillna('').astype(str).str.strip().str.lower()
    
    # --- Merge and Compare ---
    merged = df_api1.merge(df_api2, on="match_key", suffixes=("_api1", "_api2"), how="outer", indicator=True)
    
    total_comparisons, correct_comparisons = 0, 0
    diffs = {}
    unmatched_api1 = merged[merged['_merge'] == 'left_only']
    unmatched_api2 = merged[merged['_merge'] == 'right_only']
    matched = merged[merged['_merge'] == 'both']
    
    st.write(f"--- Comparison Results ---")
    st.write(f"Matched Rows: {len(matched)}")
    st.write(f"Unmatched API1 Rows (in API1 output but not API2): {len(unmatched_api1)}")
    st.write(f"Unmatched API2 Rows (in API2 output but not API1): {len(unmatched_api2)}")
    
    for _, row in matched.iterrows():
        for field in FIELDS_TO_COMPARE:
            field_api1 = f"{field}_api1"
            field_api2 = f"{field}_api2"
            
            # Check if columns exist before trying to access
            val_api1 = row.get(field_api1)
            val_api2 = row.get(field_api2)
            
            # Import the field-specific normalization function
            from api_rent_roll_verifier import normalize_field_value
            
            # Use field-specific normalization for comparison
            norm_api1 = normalize_field_value(field, val_api1)
            norm_api2 = normalize_field_value(field, val_api2)
            
            # Special handling for boolean 'is_mtm' (compare normalized 0/1)
            if field == 'is_mtm':
                # Already converted to 0/1 int above
                is_match = (int(val_api1) == int(val_api2))
            # Special handling for numeric fields (allow small tolerance?) - For now, exact match after normalization
            elif field in numeric_cols:
                # Handle potential NaN comparison carefully
                if pd.isna(val_api1) and pd.isna(val_api2):
                    is_match = True
                # Special case: treat 0 and nan as a match
                elif (pd.isna(val_api1) and val_api2 == 0) or (pd.isna(val_api2) and val_api1 == 0):
                    is_match = True
                elif pd.isna(val_api1) or pd.isna(val_api2):
                    is_match = False
                else:
                    # Could add tolerance here if needed: abs(val_api1 - val_api2) < tolerance
                    is_match = (float(val_api1) == float(val_api2))
            else:  # Default string comparison
                is_match = (norm_api1 == norm_api2)
            
            total_comparisons += 1
            if is_match:
                correct_comparisons += 1
            else:
                diffs.setdefault(field, []).append({
                    "key": row["match_key"],
                    "api1_value": val_api1,
                    "api2_value": val_api2
                })
    
    accuracy = round(100 * correct_comparisons / total_comparisons, 2) if total_comparisons > 0 else 0
    st.write(f"\nOverall Accuracy (on matched rows): {accuracy}% ({correct_comparisons}/{total_comparisons} fields)")
    
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
    
    # Calculate costs for each API output
    cost1, breakdown1 = calculate_total_cost(api_output1)
    cost2, breakdown2 = calculate_total_cost(api_output2)
    
    return accuracy, diffs, merged, field_stats, cost1, breakdown1, cost2, breakdown2, cost1 + cost2

def run_streamlit_app():
    """Main Streamlit app function for LLM comparison"""
    st.set_page_config(layout="wide")
    st.title("Rent Roll LLM Comparison Tool")
    
    # Initialize session state for persistent results and processing state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'final_summary' not in st.session_state:
        st.session_state.final_summary = None
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'completed_files' not in st.session_state:
        st.session_state.completed_files = {}
    if 'download_data' not in st.session_state:
        st.session_state.download_data = {}
    if 'download_files' not in st.session_state:
        st.session_state.download_files = {}
    
    # Create downloads directory
    downloads_dir = "downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Helper function to save file and create download link
    def create_download_file(data, filename, label):
        """Save data to file and return download link"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(downloads_dir, safe_filename)
        
        # Save the file
        with open(filepath, 'w') as f:
            if isinstance(data, dict) or isinstance(data, list):
                json.dump(data, f, indent=2, default=str)
            else:
                f.write(str(data))
        
        # Store in session state for tracking
        if 'download_files' not in st.session_state:
            st.session_state.download_files = {}
        st.session_state.download_files[label] = {
            'filepath': filepath,
            'filename': safe_filename,
            'created': timestamp
        }
        
        return filepath, safe_filename
    
    # Comparison Mode Selection
    st.subheader("Comparison Mode")
    comparison_mode = st.radio(
        "Select comparison mode:",
        ["2-Way Comparison", "3-Way Comparison"],
        help="Choose between comparing 2 or 3 different LLM configurations"
    )
    
    if comparison_mode == "2-Way Comparison":
        st.info("Upload one or more rent roll files (.xlsx or .pdf) to compare the outputs of two API calls with different LLM configurations.")
    else:
        st.info("Upload one or more rent roll files (.xlsx or .pdf) to compare the outputs of three API calls with different LLM configurations. This will show consensus analysis and identify outliers.")
        
        # Add verification options for 3-way comparison
        st.subheader("Verification Options")
        
        # Create columns for the two mutually exclusive options
        col1, col2 = st.columns(2)
        
        with col1:
            compare_to_verified = st.checkbox(
                "Compare to Verified", 
                value=False,
                help="After the 3-way comparison, compare the majority consensus result to a verified rent roll file"
            )
        
        with col2:
            compare_to_rules_based = st.checkbox(
                "Compare to Rules-Based", 
                value=False,
                help="After the 3-way comparison, compare the majority consensus result to the rules-based approach"
            )
        
        # Ensure mutual exclusivity
        if compare_to_verified and compare_to_rules_based:
            st.error("❌ **Mutual Exclusivity Error**: You can only select one comparison option at a time. Please choose either 'Compare to Verified' OR 'Compare to Rules-Based', not both.")
            st.stop()
        
        verified_file = None
        if compare_to_verified:
            st.info("📋 **Verification Mode Enabled**: Upload the verified rent roll file below. After the 3-way LLM comparison completes, the majority consensus result will be compared against this verified file.")
            verified_file = st.file_uploader(
                "Upload Verified Rent Roll File (.xlsx)", 
                type=["xlsx"],
                help="Upload the verified/ground truth version of the rent roll for comparison"
            )
            
            if not verified_file:
                st.warning("⚠️ Please upload a verified rent roll file to enable verification mode.")
        
        elif compare_to_rules_based:
            st.info("🔧 **Rules-Based Comparison Mode Enabled**: After the 3-way LLM comparison completes, the same file(s) will be processed using the rules-based approach and compared against the majority consensus result.")
    
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
    
    # Define supported models
    anthropic_models = [
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'claude-3-7-sonnet-latest',
        'claude-opus-4-20250514',
        'claude-sonnet-4-20250514'
    ]
    openai_models = [
        'gpt-4o-mini',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano'
    ]
    mistral_models = [
        'mistral-small-latest',
        'mistral-medium-2505',
        'magistral-medium-2506'
    ]
    google_models = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-2.5-pro'
    ]
    
    # Combine all models for selection
    all_models = anthropic_models + openai_models + mistral_models + google_models
    
    # Processing options
    with st.expander("LLM Configuration"):
        bypass_rb = st.checkbox("Bypass Rules-Based Approach", value=True, 
                               help="When checked, the API will always use LLMs instead of the rules-based approach")
        
        max_batch_rows = st.number_input("Max Batch Rows", 
                                        min_value=1, 
                                        max_value=200, 
                                        value=50,
                                        help="Maximum number of rows to process in each batch")
        
        if comparison_mode == "3-Way Comparison":
            n_batch_llm_calls = st.number_input("Number of LLM Calls", 
                                               min_value=2, 
                                               max_value=10, 
                                               value=3,
                                               help="Number of times to run the same LLM configuration for comparison")
        else:
            n_batch_llm_calls = 2  # For 2-way comparison
        
        st.info("Configure the LLM models that will be used for all comparison calls. The same configuration will be run multiple times to generate different results for comparison.")
        
        # Single LLM configuration for all calls
        st.subheader("LLM Configuration")
        col1, col2 = st.columns(2)
        with col1:
            llm_master = st.selectbox("Primary LLM", 
                                     all_models, 
                                     index=all_models.index('claude-sonnet-4-20250514'),
                                     help="Primary LLM for processing")
        with col2:
            llm_slave = st.selectbox("Secondary LLM", 
                                    all_models, 
                                    index=all_models.index('gemini-2.5-flash'),
                                    help="Secondary LLM for verification")
        
        # Display current configuration
        st.write("**Current Configuration:**")
        st.write(f"LLM Configuration: {llm_master} (master) + {llm_slave} (slave)")
        if comparison_mode == "3-Way Comparison":
            st.write(f"Number of LLM Calls: {n_batch_llm_calls}")
        else:
            st.write(f"Number of LLM Calls: 2 (for 2-way comparison)")
    
    # Show model categories for reference (outside the LLM Configuration expander)
    with st.expander("Available Models by Provider"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Anthropic Models:**")
            for model in anthropic_models:
                st.write(f"• {model}")
        with col2:
            st.write("**OpenAI Models:**")
            for model in openai_models:
                st.write(f"• {model}")
        with col3:
            st.write("**Mistral Models:**")
            for model in mistral_models:
                st.write(f"• {model}")
        with col4:
            st.write("**Google Models:**")
            for model in google_models:
                st.write(f"• {model}")
    
    # File upload
    st.subheader("Upload Rent Roll Files")
    uploaded_files = st.file_uploader(
        "Upload Rent Roll Files (.xlsx or .pdf)", 
        type=["xlsx", "pdf"],
        accept_multiple_files=True
    )
    
    # Optional sheet name selection
    sheet_name = st.text_input("Sheet Name (optional, leave blank for default)")
    st.info("Note: Sheet name will be ignored for PDF files.")
    
    # Display file counts
    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_type = "PDF" if file_ext == '.pdf' else "Excel"
            st.write(f"- {file.name} ({file_type})")
        
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        
        # Save all uploaded files to the temp directory
        temp_paths = []
        
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(file_path)
        
        # Create output directory for results
        output_dir = os.path.join(temp_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        if st.button("Run Comparison"):
            # Validation: Check if verification is enabled but no verified file is uploaded
            if comparison_mode == "3-Way Comparison" and compare_to_verified and not verified_file:
                st.error("❌ **Verification Error**: You have enabled 'Compare to Verified' but haven't uploaded a verified rent roll file. Please upload a verified file or disable the verification option.")
                st.stop()
            
            # Save verified file to temp directory if provided
            verified_file_path = None
            if comparison_mode == "3-Way Comparison" and compare_to_verified and verified_file:
                verified_file_path = os.path.join(temp_dir, f"verified_{verified_file.name}")
                with open(verified_file_path, "wb") as f:
                    f.write(verified_file.getbuffer())
                st.success(f"✅ Verified file saved: {verified_file.name}")
            
            # Process files with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a placeholder for results
            results_container = st.container()
            
            # Store results in session state to persist across page reloads
            st.session_state.results = []
            st.session_state.processing_complete = False
            
            # Process each file sequentially
            results = []
            successful_files = 0
            failed_files = 0
            
            for i, file_path in enumerate(temp_paths):
                file_name = os.path.basename(file_path)
                status_text.text(f"Processing file {i+1}/{len(temp_paths)}: {file_name}")
                progress_bar.progress((i) / len(temp_paths))
                
                # Create output files
                output_api1_file = os.path.join(output_dir, f"api1_output_{file_name}.json")
                output_api2_file = os.path.join(output_dir, f"api2_output_{file_name}.json")
                output_diff_file = os.path.join(output_dir, f"comparison_diff_{file_name}.json")
                if comparison_mode == "3-Way Comparison":
                    output_api3_file = os.path.join(output_dir, f"api3_output_{file_name}.json")
                
                # Process the file
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # Step 1: Submit single batch job to the API
                        st.write(f"Submitting {file_name} to API with {n_batch_llm_calls} LLM calls...")
                        
                        # Submit single batch job
                        job_id = submit_job(file_path, 
                                          token="", 
                                          doc_type="rent_roll", 
                                          sheet_name=sheet_name, 
                                          bypass_rb=bypass_rb,
                                          max_n_batch_rows=max_batch_rows,
                                          rr_master_llm=llm_master,
                                          rr_slave_llm=llm_slave,
                                          n_batch_llm_calls=n_batch_llm_calls)
                        
                        if not job_id:
                            st.error(f"Failed to submit batch job to API for {file_name}.")
                            continue
                        
                        st.write(f"Batch job submitted successfully. Job ID: {job_id}")
                        
                        # Step 2: Monitor batch job
                        st.write(f"Monitoring batch job for {file_name}...")
                        
                        # Create progress tracking for the single batch job
                        st.write("**Batch Job Progress:**")
                        file_progress_bar = st.progress(0)
                        file_status_text = st.empty()
                        
                        # Set up retry parameters
                        max_retries = 50
                        retry_delay = 15
                        
                        # Track completion status
                        batch_api_output = None
                        job_complete = False
                        
                        # Monitor batch job
                        for retry in range(max_retries):
                            file_status_text.text(f"Attempt {retry + 1}/{max_retries}: Checking batch job status...")
                            file_progress_bar.progress((retry + 1) / max_retries)
                            
                            temp_output = fetch_job_results(job_id, max_retries=1, retry_delay=0)
                            
                            if temp_output is not None:
                                if temp_output.get('job', {}).get('status') != 'processing':
                                    # Job is complete
                                    batch_api_output = temp_output
                                    job_complete = True
                                    file_status_text.text(f"Batch job completed successfully!")
                                    file_progress_bar.progress(1.0)
                                    break
                                else:
                                    file_status_text.text(f"Batch job still processing...")
                            else:
                                file_status_text.text(f"Batch job status check failed, retrying...")
                            
                            # Wait before next check
                            time.sleep(retry_delay)
                        else:
                            # Loop completed without job finishing
                            st.error(f"Maximum retries reached for batch job on {file_name}. Job may still be processing.")
                            continue
                        
                        # Verify we have the batch output
                        if not batch_api_output:
                            st.error(f"Failed to fetch batch job results from API for {file_name}.")
                            continue
                        
                        st.success(f"Batch job completed successfully for {file_name}!")
                        
                        # Step 3: Extract individual results from batch output
                        st.write(f"Extracting individual results from batch output...")
                        
                        # Import the extract_batch_results function
                        from api_rent_roll_verifier import extract_batch_results
                        
                        # Extract individual results
                        individual_results = extract_batch_results(batch_api_output, n_batch_llm_calls)
                        
                        if not individual_results or len(individual_results) != n_batch_llm_calls:
                            st.error(f"Failed to extract {n_batch_llm_calls} individual results from batch output for {file_name}.")
                            continue
                        
                        # Assign individual results
                        api_output1 = individual_results[0]
                        api_output2 = individual_results[1] if len(individual_results) > 1 else None
                        api_output3 = individual_results[2] if len(individual_results) > 2 else None
                        
                        # Verify we have all required outputs
                        if not api_output1:
                            st.error(f"Failed to extract first result from batch output for {file_name}.")
                            continue
                        
                        if comparison_mode == "2-Way Comparison" and not api_output2:
                            st.error(f"Failed to extract second result from batch output for {file_name}.")
                            continue
                        
                        if comparison_mode == "3-Way Comparison" and (not api_output2 or not api_output3):
                            st.error(f"Failed to extract all three results from batch output for {file_name}.")
                            continue
                        
                        # Save API outputs for reference
                        with open(output_api1_file, 'w') as f:
                            json.dump(api_output1, f, indent=2)
                        
                        with open(output_api2_file, 'w') as f:
                            json.dump(api_output2, f, indent=2)
                        
                        if comparison_mode == "3-Way Comparison":
                            with open(output_api3_file, 'w') as f:
                                json.dump(api_output3, f, indent=2)
                        
                        # Step 5: Compare API outputs
                        st.write(f"Comparing API outputs for {file_name}...")
                        
                        # Debug: Check API output structures
                        st.write("**Debug Information:**")
                        for i, api_output in enumerate([api_output1, api_output2, api_output3 if comparison_mode == "3-Way Comparison" else None], 1):
                            if api_output is not None:
                                if isinstance(api_output, dict):
                                    has_df = "df" in api_output
                                    df_content = api_output.get("df", "No 'df' key")
                                    if has_df and api_output["df"]:
                                        df_len = len(api_output["df"]) if isinstance(api_output["df"], list) else "Not a list"
                                        st.write(f"API Output {i}: Has 'df' key: {has_df}, df length: {df_len}")
                                    else:
                                        st.write(f"API Output {i}: Has 'df' key: {has_df}, df content: {type(df_content)}")
                                        if has_df:
                                            st.write(f"API Output {i} df value: {df_content}")
                                else:
                                    st.write(f"API Output {i}: Not a dictionary, type: {type(api_output)}")
                                
                                # Show the keys in the API output
                                if isinstance(api_output, dict):
                                    st.write(f"API Output {i} keys: {list(api_output.keys())}")
                            else:
                                st.write(f"API Output {i}: None")
                        
                        if comparison_mode == "2-Way Comparison":
                            comparison_result = compare_api_outputs(api_output1, api_output2)
                            accuracy, diffs, merged_df, field_stats, cost1, breakdown1, cost2, breakdown2, total_cost = comparison_result
                            
                            if accuracy is not None:
                                # Save detailed differences for 2-way
                                diff_output = {
                                    "file": file_path,
                                    "comparison_mode": "2-way",
                                    "accuracy_percent": accuracy,
                                    "field_mismatches": diffs,
                                    "llm_configuration": {
                                        "master_llm": llm_master,
                                        "slave_llm": llm_slave,
                                        "n_batch_llm_calls": n_batch_llm_calls,
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                }
                                
                                results.append({
                                    "file": file_path,
                                    "comparison_mode": "2-way",
                                    "accuracy": accuracy,
                                    "diffs": diffs,
                                    "merged_df": merged_df,
                                    "field_stats": field_stats,
                                    "api_output1": api_output1,
                                    "api_output2": api_output2,
                                    "api1_cost": cost1,
                                    "api1_cost_breakdown": breakdown1,
                                    "api2_cost": cost2,
                                    "api2_cost_breakdown": breakdown2,
                                    "total_cost": total_cost,
                                    "llm_config": {
                                        "master_llm": llm_master,
                                        "slave_llm": llm_slave,
                                        "n_batch_llm_calls": n_batch_llm_calls,
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                })
                        else:
                            comparison_result = compare_api_outputs_3way(api_output1, api_output2, api_output3)
                            
                            if comparison_result is not None:
                                # Initialize verification variables
                                verification_accuracy = None
                                verification_diffs = None
                                verification_merged_df = None
                                majority_consensus_output = None
                                
                                # Step 6: Run verification or rules-based comparison if enabled
                                if compare_to_verified and verified_file_path:
                                    st.write(f"🔍 **Starting Verification Process for {file_name}**")
                                    
                                    # Generate majority consensus output
                                    st.write("Generating majority consensus from 3-way comparison...")
                                    majority_consensus_output = generate_majority_consensus_output(api_output1, api_output2, api_output3)
                                    
                                    if majority_consensus_output:
                                        # Save majority consensus output for reference
                                        majority_output_file = os.path.join(output_dir, f"majority_consensus_{file_name}.json")
                                        with open(majority_output_file, 'w') as f:
                                            json.dump(majority_consensus_output, f, indent=2)
                                        
                                        st.write("Comparing majority consensus to verified file...")
                                        verification_accuracy, verification_diffs, verification_merged_df = compare_data(
                                            majority_consensus_output, 
                                            verified_file_path
                                        )
                                        
                                        if verification_accuracy is not None:
                                            st.success(f"✅ **Verification Complete**: {verification_accuracy:.2f}% accuracy against verified file")
                                            
                                            # Save verification results
                                            verification_output_file = os.path.join(output_dir, f"verification_results_{file_name}.json")
                                            verification_output = {
                                                "file": file_path,
                                                "verified_file": verified_file_path,
                                                "verification_accuracy_percent": verification_accuracy,
                                                "verification_mismatches": verification_diffs,
                                                "majority_consensus_used": True
                                            }
                                            with open(verification_output_file, 'w') as f:
                                                json.dump(verification_output, f, indent=2, default=str)
                                        else:
                                            st.error("❌ Verification failed. Check logs for details.")
                                    else:
                                        st.error("❌ Failed to generate majority consensus output for verification.")
                                
                                elif compare_to_rules_based:
                                    st.write(f"🔧 **Starting Rules-Based Comparison Process for {file_name}**")
                                    
                                    # Generate majority consensus output
                                    st.write("Generating majority consensus from 3-way comparison...")
                                    majority_consensus_output = generate_majority_consensus_output(api_output1, api_output2, api_output3)
                                    
                                    if majority_consensus_output:
                                        # Save majority consensus output for reference
                                        majority_output_file = os.path.join(output_dir, f"majority_consensus_{file_name}.json")
                                        with open(majority_output_file, 'w') as f:
                                            json.dump(majority_consensus_output, f, indent=2)
                                        
                                        # Submit job to API with rules-based approach (bypass_rb=False)
                                        st.write("Submitting file to API with rules-based approach...")
                                        rules_based_job_id = submit_job(file_path, 
                                                               token="", 
                                                               doc_type="rent_roll", 
                                                               sheet_name=sheet_name, 
                                                               bypass_rb=False,  # Use rules-based approach
                                                               max_n_batch_rows=max_batch_rows,
                                                               rr_master_llm=llm_master,  # LLM config doesn't matter for rules-based
                                                               rr_slave_llm=llm_slave)
                                        
                                        if rules_based_job_id:
                                            st.write(f"Rules-based job submitted successfully. Job ID: {rules_based_job_id}")
                                            
                                            # Monitor rules-based job
                                            st.write("Monitoring rules-based API call...")
                                            rules_based_progress_bar = st.progress(0)
                                            rules_based_status_text = st.empty()
                                            
                                            rules_based_output = None
                                            rules_based_complete = False
                                            
                                            # Monitor rules-based job
                                            for retry in range(max_retries):
                                                rules_based_status_text.text(f"Attempt {retry + 1}/{max_retries}: Checking rules-based job status...")
                                                rules_based_progress_bar.progress((retry + 1) / max_retries)
                                                
                                                temp_rules_output = fetch_job_results(rules_based_job_id, max_retries=1, retry_delay=0)
                                                
                                                if temp_rules_output is not None:
                                                    if temp_rules_output.get('job', {}).get('status') != 'processing':
                                                        # Job is complete
                                                        rules_based_output = temp_rules_output
                                                        rules_based_complete = True
                                                        rules_based_status_text.text(f"Rules-based job completed successfully!")
                                                        rules_based_progress_bar.progress(1.0)
                                                        break
                                                    else:
                                                        rules_based_status_text.text(f"Rules-based job still processing...")
                                                else:
                                                    rules_based_status_text.text(f"Rules-based job status check failed, retrying...")
                                                
                                                # Wait before next check
                                                time.sleep(retry_delay)
                                            else:
                                                st.error(f"Maximum retries reached for rules-based job on {file_name}. Job may still be processing.")
                                                rules_based_complete = False
                                            
                                            if rules_based_complete and rules_based_output:
                                                # Save rules-based output for reference
                                                rules_based_output_file = os.path.join(output_dir, f"rules_based_output_{file_name}.json")
                                                with open(rules_based_output_file, 'w') as f:
                                                    json.dump(rules_based_output, f, indent=2)
                                                
                                                st.write("Comparing majority consensus to rules-based approach...")
                                                
                                                # Create a temporary file to use compare_data function
                                                # Since compare_data expects a file path for the second argument, we need to create a temporary Excel file
                                                # from the rules-based output, but that's complex. Instead, let's modify the comparison logic.
                                                
                                                # For now, let's use a simpler approach - directly compare the DataFrames
                                                try:
                                                    # Convert rules-based output to the same format as majority consensus
                                                    if rules_based_output and "df" in rules_based_output and rules_based_output["df"]:
                                                        # Use the existing comparison logic but adapt it for rules-based comparison
                                                        df_majority = pd.DataFrame(majority_consensus_output["df"])
                                                        df_rules = pd.DataFrame(rules_based_output["df"])
                                                        
                                                        # Normalize column names
                                                        df_majority.columns = df_majority.columns.str.strip().str.lower()
                                                        df_rules.columns = df_rules.columns.str.strip().str.lower()
                                                        
                                                        # Add missing columns
                                                        for df in [df_majority, df_rules]:
                                                            missing_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df.columns)
                                                            for col in missing_cols:
                                                                df[col] = pd.NA
                                                        
                                                        # Data type conversion (same as in compare_data)
                                                        date_cols = ['move_in', 'lease_start', 'lease_end']
                                                        numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
                                                        
                                                        for df in [df_majority, df_rules]:
                                                            # Convert date columns
                                                            for col in date_cols:
                                                                if col in df.columns:
                                                                    df[col] = df[col].replace('nan', pd.NA)
                                                                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%m/%d/%Y')
                                                                else:
                                                                    df[col] = pd.NA
                                                            
                                                            # Convert numeric columns
                                                            for col in numeric_cols:
                                                                if col in df.columns:
                                                                    df[col] = df[col].replace('nan', pd.NA)
                                                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                                            
                                                            # Handle boolean 'is_mtm'
                                                            if 'is_mtm' in df.columns:
                                                                df['is_mtm'] = df['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
                                                                df['is_mtm'] = pd.to_numeric(df['is_mtm'], errors='coerce').fillna(0).astype(int)
                                                        
                                                        # Use row position for matching (same as compare_data)
                                                        df_majority['match_key'] = range(len(df_majority))
                                                        df_rules['match_key'] = range(len(df_rules))
                                                        
                                                        # Trim to shorter length if needed
                                                        if len(df_majority) != len(df_rules):
                                                            st.warning(f"Number of rows in majority consensus ({len(df_majority)}) does not match rules-based output ({len(df_rules)})")
                                                            min_rows = min(len(df_majority), len(df_rules))
                                                            df_majority = df_majority.iloc[:min_rows].copy()
                                                            df_rules = df_rules.iloc[:min_rows].copy()
                                                            df_majority['match_key'] = range(len(df_majority))
                                                            df_rules['match_key'] = range(len(df_rules))
                                                        
                                                        # Merge and compare
                                                        merged = df_majority.merge(df_rules, on="match_key", suffixes=("_majority", "_rules"), how="outer", indicator=True)
                                                        matched = merged[merged['_merge'] == 'both']
                                                        
                                                        total_comparisons, correct_comparisons = 0, 0
                                                        rules_based_diffs = {}
                                                        
                                                        for _, row in matched.iterrows():
                                                            for field in FIELDS_TO_COMPARE:
                                                                field_majority = f"{field}_majority"
                                                                field_rules = f"{field}_rules"
                                                                
                                                                val_majority = row.get(field_majority)
                                                                val_rules = row.get(field_rules)
                                                                
                                                                # Import the field-specific normalization function
                                                                from api_rent_roll_verifier import normalize_field_value
                                                                
                                                                # Use field-specific normalization for comparison
                                                                norm_majority = normalize_field_value(field, val_majority)
                                                                norm_rules = normalize_field_value(field, val_rules)
                                                                
                                                                # Special handling for different field types
                                                                if field == 'is_mtm':
                                                                    is_match = (int(val_majority) == int(val_rules))
                                                                elif field in numeric_cols:
                                                                    if pd.isna(val_majority) and pd.isna(val_rules):
                                                                        is_match = True
                                                                    elif (pd.isna(val_majority) and val_rules == 0) or (pd.isna(val_rules) and val_majority == 0):
                                                                        is_match = True
                                                                    elif pd.isna(val_majority) or pd.isna(val_rules):
                                                                        is_match = False
                                                                    else:
                                                                        is_match = (float(val_majority) == float(val_rules))
                                                                else:
                                                                    is_match = (norm_majority == norm_rules)
                                                                
                                                                total_comparisons += 1
                                                                if is_match:
                                                                    correct_comparisons += 1
                                                                else:
                                                                    rules_based_diffs.setdefault(field, []).append({
                                                                        "key": row["match_key"],
                                                                        "api_value": val_majority,  # Majority consensus value
                                                                        "rules_value": val_rules    # Rules-based value
                                                                    })
                                                        
                                                        verification_accuracy = round(100 * correct_comparisons / total_comparisons, 2) if total_comparisons > 0 else 0
                                                        verification_diffs = rules_based_diffs
                                                        verification_merged_df = merged
                                                        
                                                        st.success(f"✅ **Rules-Based Comparison Complete**: {verification_accuracy:.2f}% accuracy against rules-based approach")
                                                        
                                                        # Save rules-based comparison results
                                                        rules_comparison_output_file = os.path.join(output_dir, f"rules_based_comparison_{file_name}.json")
                                                        rules_comparison_output = {
                                                            "file": file_path,
                                                            "rules_based_accuracy_percent": verification_accuracy,
                                                            "rules_based_mismatches": verification_diffs,
                                                            "majority_consensus_used": True
                                                        }
                                                        with open(rules_comparison_output_file, 'w') as f:
                                                            json.dump(rules_comparison_output, f, indent=2, default=str)
                                                    
                                                    else:
                                                        st.error("❌ Rules-based output is empty or not in expected format.")
                                                        verification_accuracy = None
                                                        verification_diffs = None
                                                        verification_merged_df = None
                                                
                                                except Exception as e:
                                                    st.error(f"❌ Error during rules-based comparison: {e}")
                                                    st.text(traceback.format_exc())
                                                    verification_accuracy = None
                                                    verification_diffs = None
                                                    verification_merged_df = None
                                            
                                            else:
                                                st.error("❌ Rules-based job failed or did not complete.")
                                                verification_accuracy = None
                                                verification_diffs = None
                                                verification_merged_df = None
                                        
                                        else:
                                            st.error("❌ Failed to submit rules-based job to API.")
                                            verification_accuracy = None
                                            verification_diffs = None
                                            verification_merged_df = None
                                    
                                    else:
                                        st.error("❌ Failed to generate majority consensus output for rules-based comparison.")
                                        verification_accuracy = None
                                        verification_diffs = None
                                        verification_merged_df = None
                                
                                # Save detailed differences for 3-way (including verification if performed)
                                diff_output = {
                                    "file": file_path,
                                    "comparison_mode": "3-way",
                                    "perfect_accuracy_percent": comparison_result["perfect_accuracy"],
                                    "majority_accuracy_percent": comparison_result["majority_accuracy"],
                                    "consensus_data": comparison_result["consensus_data"],
                                    "field_consensus_stats": comparison_result["field_consensus_stats"],
                                    "api_outlier_count": comparison_result["api_outlier_count"],
                                    "llm_configuration": {
                                        "master_llm": llm_master,
                                        "slave_llm": llm_slave,
                                        "n_batch_llm_calls": n_batch_llm_calls,
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    }
                                }
                                
                                # Add verification or rules-based results to diff_output if performed
                                if compare_to_verified and verified_file_path:
                                    diff_output["verification_enabled"] = True
                                    diff_output["verified_file"] = verified_file_path
                                    if verification_accuracy is not None:
                                        diff_output["verification_accuracy_percent"] = verification_accuracy
                                        diff_output["verification_mismatches"] = verification_diffs
                                    else:
                                        diff_output["verification_failed"] = True
                                elif compare_to_rules_based:
                                    diff_output["rules_based_enabled"] = True
                                    if verification_accuracy is not None:
                                        diff_output["rules_based_accuracy_percent"] = verification_accuracy
                                        diff_output["rules_based_mismatches"] = verification_diffs
                                    else:
                                        diff_output["rules_based_failed"] = True
                                else:
                                    diff_output["verification_enabled"] = False
                                    diff_output["rules_based_enabled"] = False
                                
                                results.append({
                                    "file": file_path,
                                    "comparison_mode": "3-way",
                                    "perfect_accuracy": comparison_result["perfect_accuracy"],
                                    "majority_accuracy": comparison_result["majority_accuracy"],
                                    "consensus_data": comparison_result["consensus_data"],
                                    "field_consensus_stats": comparison_result["field_consensus_stats"],
                                    "api_outlier_count": comparison_result["api_outlier_count"],
                                    "merged_df": comparison_result["merged_df"],
                                    "api_output1": api_output1,
                                    "api_output2": api_output2,
                                    "api_output3": api_output3,
                                    "api1_cost": comparison_result["api1_cost"],
                                    "api1_cost_breakdown": comparison_result["api1_cost_breakdown"],
                                    "api2_cost": comparison_result["api2_cost"],
                                    "api2_cost_breakdown": comparison_result["api2_cost_breakdown"],
                                    "api3_cost": comparison_result["api3_cost"],
                                    "api3_cost_breakdown": comparison_result["api3_cost_breakdown"],
                                    "total_cost": comparison_result["total_cost"],
                                    "llm_config": {
                                        "master_llm": llm_master,
                                        "slave_llm": llm_slave,
                                        "n_batch_llm_calls": n_batch_llm_calls,
                                        "max_batch_rows": max_batch_rows,
                                        "bypass_rb": bypass_rb
                                    },
                                    # Add verification/rules-based results to the result object
                                    "verification_enabled": compare_to_verified and verified_file_path is not None,
                                    "rules_based_enabled": compare_to_rules_based,
                                    "verification_accuracy": verification_accuracy,
                                    "verification_diffs": verification_diffs,
                                    "verification_merged_df": verification_merged_df,
                                    "majority_consensus_output": majority_consensus_output
                                })
                            else:
                                st.error(f"3-way comparison failed for {file_name}. Check logs for details.")
                                continue
                        
                        # Save the diff output
                        with open(output_diff_file, 'w') as f:
                            json.dump(diff_output, f, indent=2, default=str)
                        
                        # Add to session state results
                        st.session_state.results.append(results[-1])
                        successful_files += 1
                        
                        # Store download data in session state
                        st.session_state.download_data[file_name] = {
                            "diff_output": diff_output,
                            "result": results[-1]
                        }
                        
                        # Show individual file completion with both LLM and rules-based accuracy
                        if comparison_mode == "2-Way Comparison":
                            st.success(f"✅ **{file_name} completed successfully!** - Accuracy: {results[-1]['accuracy']:.2f}%")
                        else:
                            # Show 3-way LLM accuracy and rules-based accuracy if available
                            llm_accuracy_text = f"Perfect: {results[-1]['perfect_accuracy']:.2f}%, Majority: {results[-1]['majority_accuracy']:.2f}%"
                            if results[-1].get("rules_based_enabled") and results[-1].get("verification_accuracy") is not None:
                                rules_accuracy_text = f", Rules-Based: {results[-1]['verification_accuracy']:.2f}%"
                                st.success(f"✅ **{file_name} completed successfully!** - LLM ({llm_accuracy_text}){rules_accuracy_text}")
                            elif results[-1].get("verification_enabled") and results[-1].get("verification_accuracy") is not None:
                                verification_accuracy_text = f", Verification: {results[-1]['verification_accuracy']:.2f}%"
                                st.success(f"✅ **{file_name} completed successfully!** - LLM ({llm_accuracy_text}){verification_accuracy_text}")
                            else:
                                st.success(f"✅ **{file_name} completed successfully!** - LLM ({llm_accuracy_text})")
                    
                    except Exception as e:
                        st.error(f"❌ **Error processing {file_name}**: {e}")
                        st.text(traceback.format_exc())
                        failed_files += 1
                        # Continue to next file instead of stopping
                        continue
                
                # Update progress
                progress_value = min(1.0, (i + 1) / len(temp_paths))
                progress_bar.progress(progress_value)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Mark processing as complete and store final summary
            st.session_state.processing_complete = True
            st.session_state.final_summary = {
                "total_files": len(temp_paths),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "comparison_mode": comparison_mode,
                "rules_based_enabled": comparison_mode == "3-Way Comparison" and compare_to_rules_based,
                "verification_enabled": comparison_mode == "3-Way Comparison" and compare_to_verified
            }
            
            # Display final summary
            st.subheader("📊 Processing Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(temp_paths))
            with col2:
                st.metric("Successful", successful_files, delta=f"{successful_files}/{len(temp_paths)}")
            with col3:
                st.metric("Failed", failed_files, delta=f"{failed_files}/{len(temp_paths)}" if failed_files > 0 else None)
            
            if failed_files > 0:
                st.warning(f"⚠️ {failed_files} file(s) failed to process. Check the error messages above for details.")
            
            # Generate final combined report for all files
            if results:
                # Create combined report
                combined_report = {
                    "processing_summary": {
                        "total_files": len(temp_paths),
                        "successful_files": successful_files,
                        "failed_files": failed_files,
                        "comparison_mode": comparison_mode,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "individual_results": []
                }
                
                # Add each file's results to the combined report
                for result in results:
                    file_summary = {
                        "file": os.path.basename(result["file"]),
                        "comparison_mode": result["comparison_mode"],
                        "total_cost": result["total_cost"]
                    }
                    
                    if result["comparison_mode"] == "2-way":
                        file_summary["accuracy"] = result["accuracy"]
                    else:
                        file_summary["perfect_accuracy"] = result["perfect_accuracy"]
                        file_summary["majority_accuracy"] = result["majority_accuracy"]
                        
                        # Add verification/rules-based results if available
                        if result.get("verification_enabled") and result.get("verification_accuracy") is not None:
                            file_summary["verification_accuracy"] = result["verification_accuracy"]
                        elif result.get("rules_based_enabled") and result.get("verification_accuracy") is not None:
                            file_summary["rules_based_accuracy"] = result["verification_accuracy"]
                    
                    combined_report["individual_results"].append(file_summary)
                
                # Create and save combined report file
                filepath, filename = create_download_file(
                    combined_report, 
                    f"combined_comparison_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    "Combined Report"
                )
                st.success(f"📥 **Combined Report Ready**: [Download {filename}](./{filepath})")
            
            # Clean up temp files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except OSError as e:
                st.warning(f"Could not remove temporary files: {e}")
    
    # Display results from session state (outside the button block so they persist)
    if st.session_state.results:
        st.subheader("Comparison Results")
        
        # Calculate average accuracy based on comparison mode
        if st.session_state.results[0]["comparison_mode"] == "2-way":
            avg_accuracy = sum(r["accuracy"] for r in st.session_state.results) / len(st.session_state.results)
            st.metric("Average Accuracy Across All Files", f"{avg_accuracy:.2f}%")
        else:
            avg_perfect = sum(r["perfect_accuracy"] for r in st.session_state.results) / len(st.session_state.results)
            avg_majority = sum(r["majority_accuracy"] for r in st.session_state.results) / len(st.session_state.results)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Perfect Agreement", f"{avg_perfect:.2f}%")
            with col2:
                st.metric("Average Majority Consensus", f"{avg_majority:.2f}%")
        
        # Create tabs for each file
        tabs = st.tabs([os.path.basename(r["file"]) for r in st.session_state.results])
        
        for i, (tab, result) in enumerate(zip(tabs, st.session_state.results)):
            with tab:
                if result["comparison_mode"] == "2-way":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("File Accuracy", f"{result['accuracy']:.2f}%")
                    with col2:
                        st.metric("Total Cost", f"${result['total_cost']}")
                else:
                    # Show verification or rules-based results if available
                    if result.get("verification_enabled") and result.get("verification_accuracy") is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Perfect Agreement", f"{result['perfect_accuracy']:.2f}%")
                        with col2:
                            st.metric("Majority Consensus", f"{result['majority_accuracy']:.2f}%")
                        with col3:
                            st.metric("Verification Accuracy", f"{result['verification_accuracy']:.2f}%")
                        with col4:
                            st.metric("Total Cost", f"${result['total_cost']}")
                    elif result.get("rules_based_enabled") and result.get("verification_accuracy") is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Perfect Agreement", f"{result['perfect_accuracy']:.2f}%")
                        with col2:
                            st.metric("Majority Consensus", f"{result['majority_accuracy']:.2f}%")
                        with col3:
                            st.metric("Rules-Based Accuracy", f"{result['verification_accuracy']:.2f}%")
                        with col4:
                            st.metric("Total Cost", f"${result['total_cost']}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Perfect Agreement", f"{result['perfect_accuracy']:.2f}%")
                        with col2:
                            st.metric("Majority Consensus", f"{result['majority_accuracy']:.2f}%")
                        with col3:
                            st.metric("Total Cost", f"${result['total_cost']}")
                
                # Summary Section
                st.subheader("📊 Results Summary")
                if result["comparison_mode"] == "2-way":
                    st.write(f"**2-Way LLM Comparison Accuracy:** {result['accuracy']:.2f}%")
                else:
                    summary_text = f"**3-Way LLM Results:** Perfect Agreement: {result['perfect_accuracy']:.2f}%, Majority Consensus: {result['majority_accuracy']:.2f}%"
                    
                    if result.get("verification_enabled") and result.get("verification_accuracy") is not None:
                        summary_text += f"\n\n**Verification Against Ground Truth:** {result['verification_accuracy']:.2f}% accuracy"
                    elif result.get("rules_based_enabled") and result.get("verification_accuracy") is not None:
                        summary_text += f"\n\n**Rules-Based Comparison:** {result['verification_accuracy']:.2f}% accuracy"
                    
                    st.write(summary_text)
                
                st.write(f"**Total Processing Cost:** ${result['total_cost']}")
                
                # Display cost breakdown
                with st.expander("💰 Cost Breakdown"):
                    if result["comparison_mode"] == "2-way":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**First API Call Cost:**")
                            st.write(f"**Total: ${result['api1_cost']:.4f}**")
                            if result['api1_cost_breakdown']:
                                # Only show LLMs with cost > 0
                                used_llms = [breakdown for breakdown in result['api1_cost_breakdown'] if breakdown['cost'] > 0]
                                if used_llms:
                                    for breakdown in used_llms:
                                        st.write(f"• **{breakdown['model']}**: ${breakdown['cost']:.4f}")
                                        st.write(f"  - Input tokens: {breakdown['in_tokens']:,}")
                                        st.write(f"  - Output tokens: {breakdown['out_tokens']:,}")
                                else:
                                    st.write("No LLMs used (all costs are $0)")
                            else:
                                st.write("No cost data available")
                        
                        with col2:
                            st.write("**Second API Call Cost:**")
                            st.write(f"**Total: ${result['api2_cost']:.4f}**")
                            if result['api2_cost_breakdown']:
                                # Only show LLMs with cost > 0
                                used_llms = [breakdown for breakdown in result['api2_cost_breakdown'] if breakdown['cost'] > 0]
                                if used_llms:
                                    for breakdown in used_llms:
                                        st.write(f"• **{breakdown['model']}**: ${breakdown['cost']:.4f}")
                                        st.write(f"  - Input tokens: {breakdown['in_tokens']:,}")
                                        st.write(f"  - Output tokens: {breakdown['out_tokens']:,}")
                                else:
                                    st.write("No LLMs used (all costs are $0)")
                            else:
                                st.write("No cost data available")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**First API Call Cost:**")
                            st.write(f"**Total: ${result['api1_cost']:.4f}**")
                            if result['api1_cost_breakdown']:
                                # Only show LLMs with cost > 0
                                used_llms = [breakdown for breakdown in result['api1_cost_breakdown'] if breakdown['cost'] > 0]
                                if used_llms:
                                    for breakdown in used_llms:
                                        st.write(f"• **{breakdown['model']}**: ${breakdown['cost']:.4f}")
                                        st.write(f"  - Input tokens: {breakdown['in_tokens']:,}")
                                        st.write(f"  - Output tokens: {breakdown['out_tokens']:,}")
                                else:
                                    st.write("No LLMs used (all costs are $0)")
                            else:
                                st.write("No cost data available")
                        
                        with col2:
                            st.write("**Second API Call Cost:**")
                            st.write(f"**Total: ${result['api2_cost']:.4f}**")
                            if result['api2_cost_breakdown']:
                                # Only show LLMs with cost > 0
                                used_llms = [breakdown for breakdown in result['api2_cost_breakdown'] if breakdown['cost'] > 0]
                                if used_llms:
                                    for breakdown in used_llms:
                                        st.write(f"• **{breakdown['model']}**: ${breakdown['cost']:.4f}")
                                        st.write(f"  - Input tokens: {breakdown['in_tokens']:,}")
                                        st.write(f"  - Output tokens: {breakdown['out_tokens']:,}")
                                else:
                                    st.write("No LLMs used (all costs are $0)")
                            else:
                                st.write("No cost data available")
                        
                        with col3:
                            st.write("**Third API Call Cost:**")
                            st.write(f"**Total: ${result['api3_cost']:.4f}**")
                            if result['api3_cost_breakdown']:
                                # Only show LLMs with cost > 0
                                used_llms = [breakdown for breakdown in result['api3_cost_breakdown'] if breakdown['cost'] > 0]
                                if used_llms:
                                    for breakdown in used_llms:
                                        st.write(f"• **{breakdown['model']}**: ${breakdown['cost']:.4f}")
                                        st.write(f"  - Input tokens: {breakdown['in_tokens']:,}")
                                        st.write(f"  - Output tokens: {breakdown['out_tokens']:,}")
                                else:
                                    st.write("No LLMs used (all costs are $0)")
                            else:
                                st.write("No cost data available")
                    
                    st.write(f"**Combined Total Cost: ${result['total_cost']:.4f}**")
                
                # Display LLM configuration used for this comparison
                with st.expander("LLM Configuration Used"):
                    st.write("**Batch LLM Configuration:**")
                    st.write(f"• Primary LLM: {result['llm_config']['master_llm']}")
                    st.write(f"• Secondary LLM: {result['llm_config']['slave_llm']}")
                    st.write(f"• Number of LLM Calls: {result['llm_config']['n_batch_llm_calls']}")
                    st.write("**Other Settings:**")
                    st.write(f"• Max Batch Rows: {result['llm_config']['max_batch_rows']}")
                    st.write(f"• Bypass Rules-Based: {result['llm_config']['bypass_rb']}")
                
                if result["comparison_mode"] == "2-way":
                    # Display per-field accuracy for 2-way
                    st.subheader("Per-Field Accuracy")
                    field_df = pd.DataFrame({
                        "Field": [field for field in result["field_stats"].keys()],
                        "Accuracy (%)": [stats["accuracy"] for stats in result["field_stats"].values()],
                        "Mismatches": [stats["mismatches"] for stats in result["field_stats"].values()],
                        "Total": [stats["total"] for stats in result["field_stats"].values()]
                    })
                    field_df = field_df.sort_values("Accuracy (%)")
                    st.dataframe(field_df)
                    
                    # Display mismatch examples for 2-way
                    st.subheader("Field Mismatches")
                    all_mismatches = []
                    for field, mismatches in result["diffs"].items():
                        for mismatch in mismatches:
                            all_mismatches.append({
                                "Field": field,
                                "Unit": mismatch['key'],
                                "API Call 1 Value": mismatch['api1_value'],
                                "API Call 2 Value": mismatch['api2_value']
                            })
                    
                    if all_mismatches:
                        # Convert all values to strings to avoid Arrow serialization issues
                        for mismatch in all_mismatches:
                            mismatch["API Call 1 Value"] = str(mismatch["API Call 1 Value"]) if mismatch["API Call 1 Value"] is not None else "None"
                            mismatch["API Call 2 Value"] = str(mismatch["API Call 2 Value"]) if mismatch["API Call 2 Value"] is not None else "None"
                        
                        mismatches_df = pd.DataFrame(all_mismatches)
                        st.dataframe(mismatches_df)
                        
                        # Text list of differences
                        st.subheader("Text List of Differences")
                        for mismatch in all_mismatches:
                            st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, API Call 1: {mismatch['API Call 1 Value']}, API Call 2: {mismatch['API Call 2 Value']}")
                    else:
                        st.write("No differences found between API calls.")
                
                else:
                    # Display 3-way consensus analysis
                    st.subheader("Consensus Analysis")
                    
                    # Outlier frequency
                    st.subheader("Outlier Frequency")
                    outlier_df = pd.DataFrame({
                        "API": ["API 1", "API 2", "API 3"],
                        "Times as Outlier": [
                            result["api_outlier_count"]["api1"],
                            result["api_outlier_count"]["api2"],
                            result["api_outlier_count"]["api3"]
                        ]
                    })
                    st.dataframe(outlier_df)
                    
                    # Per-field consensus stats
                    st.subheader("Per-Field Consensus")
                    field_consensus_df = pd.DataFrame({
                        "Field": [field for field in result["field_consensus_stats"].keys()],
                        "Perfect Agreement": [stats["perfect"] for stats in result["field_consensus_stats"].values()],
                        "Majority Agreement": [stats["majority"] for stats in result["field_consensus_stats"].values()],
                        "Complete Disagreement": [stats["disagreement"] for stats in result["field_consensus_stats"].values()],
                        "Total": [stats["total"] for stats in result["field_consensus_stats"].values()]
                    })
                    st.dataframe(field_consensus_df)
                    
                    # Display consensus mismatches
                    st.subheader("Consensus Mismatches")
                    all_consensus_mismatches = []
                    for field, mismatches in result["consensus_data"].items():
                        for mismatch in mismatches:
                            consensus_type_display = {
                                "majority": "Majority (2 agree, 1 differs)",
                                "disagreement": "Complete Disagreement"
                            }.get(mismatch['consensus_type'], mismatch['consensus_type'])
                            
                            mismatch_row = {
                                "Field": field,
                                "Unit": mismatch['key'],
                                "API Call 1 Value": mismatch['api1_value'],
                                "API Call 2 Value": mismatch['api2_value'],
                                "API Call 3 Value": mismatch['api3_value'],
                                "Consensus Type": consensus_type_display
                            }
                            
                            if mismatch['outlier_api']:
                                mismatch_row["Outlier"] = mismatch['outlier_api'].upper()
                            else:
                                mismatch_row["Outlier"] = "N/A"
                            
                            all_consensus_mismatches.append(mismatch_row)
                    
                    if all_consensus_mismatches:
                        # Convert all values to strings to avoid Arrow serialization issues
                        for mismatch in all_consensus_mismatches:
                            mismatch["API Call 1 Value"] = str(mismatch["API Call 1 Value"]) if mismatch["API Call 1 Value"] is not None else "None"
                            mismatch["API Call 2 Value"] = str(mismatch["API Call 2 Value"]) if mismatch["API Call 2 Value"] is not None else "None"
                            mismatch["API Call 3 Value"] = str(mismatch["API Call 3 Value"]) if mismatch["API Call 3 Value"] is not None else "None"
                        
                        consensus_df = pd.DataFrame(all_consensus_mismatches)
                        st.dataframe(consensus_df)
                        
                        # Text list of consensus differences
                        st.subheader("Text List of Consensus Differences")
                        for mismatch in all_consensus_mismatches:
                            outlier_text = f" (Outlier: {mismatch['Outlier']})" if mismatch['Outlier'] != "N/A" else ""
                            st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, API1: {mismatch['API Call 1 Value']}, API2: {mismatch['API Call 2 Value']}, API3: {mismatch['API Call 3 Value']} - {mismatch['Consensus Type']}{outlier_text}")
                    else:
                        st.write("Perfect consensus across all API calls!")
                
                # Provide file-based downloads
                st.subheader("📥 Download Files")
                if result["comparison_mode"] == "2-way":
                    col1, col2, col3 = st.columns(3)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                
                # Get the diff_output from session state
                diff_output = st.session_state.download_data.get(os.path.basename(result["file"]), {}).get("diff_output", {})
                
                with col1:
                    # Create and save comparison details file
                    filepath, filename = create_download_file(
                        diff_output, 
                        f"comparison_diff_{os.path.basename(result['file'])}.json",
                        f"Comparison Details {i}"
                    )
                    st.success(f"📊 **Comparison Details**: [Download {filename}](./{filepath})")
                
                with col2:
                    # Create and save first API output file
                    filepath, filename = create_download_file(
                        result["api_output1"], 
                        f"api1_output_{os.path.basename(result['file'])}.json",
                        f"API Output 1 {i}"
                    )
                    st.success(f"🔧 **API Output 1**: [Download {filename}](./{filepath})")
                
                with col3:
                    # Create and save second API output file
                    filepath, filename = create_download_file(
                        result["api_output2"], 
                        f"api2_output_{os.path.basename(result['file'])}.json",
                        f"API Output 2 {i}"
                    )
                    st.success(f"🔧 **API Output 2**: [Download {filename}](./{filepath})")
                
                if result["comparison_mode"] == "3-way":
                    with col4:
                        # Create and save third API output file
                        filepath, filename = create_download_file(
                            result["api_output3"], 
                            f"api3_output_{os.path.basename(result['file'])}.json",
                            f"API Output 3 {i}"
                        )
                        st.success(f"🔧 **API Output 3**: [Download {filename}](./{filepath})")
                
                # Add verification results display for 3-way comparison
                if result.get("verification_enabled") and result.get("verification_accuracy") is not None:
                    st.subheader("🔍 Verification Results")
                    st.success(f"Verification completed with {result['verification_accuracy']:.2f}% accuracy against verified file")
                    
                    # Calculate per-field verification accuracy
                    if result.get("verification_diffs"):
                        verification_field_stats = {}
                        verification_merged_df = result.get("verification_merged_df")
                        
                        if verification_merged_df is not None:
                            matched_verification_rows = verification_merged_df[verification_merged_df['_merge'] == 'both']
                            
                            for field in FIELDS_TO_COMPARE:
                                field_total = len(matched_verification_rows)
                                field_mismatches = len(result["verification_diffs"].get(field, []))
                                field_correct = field_total - field_mismatches
                                field_accuracy = round(100 * field_correct / field_total, 2) if field_total > 0 else 0
                                verification_field_stats[field] = {
                                    "accuracy": field_accuracy,
                                    "correct": field_correct,
                                    "total": field_total,
                                    "mismatches": field_mismatches
                                }
                            
                            # Display per-field verification accuracy
                            st.subheader("Per-Field Verification Accuracy")
                            verification_field_df = pd.DataFrame({
                                "Field": [field for field in verification_field_stats.keys()],
                                "Accuracy (%)": [stats["accuracy"] for stats in verification_field_stats.values()],
                                "Mismatches": [stats["mismatches"] for stats in verification_field_stats.values()],
                                "Total": [stats["total"] for stats in verification_field_stats.values()]
                            })
                            verification_field_df = verification_field_df.sort_values("Accuracy (%)")
                            st.dataframe(verification_field_df)
                            
                            # Display verification mismatches
                            st.subheader("Verification Mismatches")
                            all_verification_mismatches = []
                            for field, mismatches in result["verification_diffs"].items():
                                for mismatch in mismatches:
                                    all_verification_mismatches.append({
                                        "Field": field,
                                        "Unit": mismatch['key'],
                                        "Majority Consensus Value": mismatch['api_value'],
                                        "Verified Value": mismatch['verified_value']
                                    })
                            
                            if all_verification_mismatches:
                                # Convert all values to strings to avoid Arrow serialization issues
                                for mismatch in all_verification_mismatches:
                                    mismatch["Majority Consensus Value"] = str(mismatch["Majority Consensus Value"]) if mismatch["Majority Consensus Value"] is not None else "None"
                                    mismatch["Verified Value"] = str(mismatch["Verified Value"]) if mismatch["Verified Value"] is not None else "None"
                                
                                verification_mismatches_df = pd.DataFrame(all_verification_mismatches)
                                st.dataframe(verification_mismatches_df)
                                
                                # Text list of verification differences
                                st.subheader("Text List of Verification Differences")
                                for mismatch in all_verification_mismatches:
                                    st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, Majority Consensus: {mismatch['Majority Consensus Value']}, Verified: {mismatch['Verified Value']}")
                            else:
                                st.write("Perfect match between majority consensus and verified file!")
                    
                    # Create and save majority consensus file
                    if result.get("majority_consensus_output"):
                        filepath, filename = create_download_file(
                            result["majority_consensus_output"], 
                            f"majority_consensus_{os.path.basename(result['file'])}.json",
                            f"Majority Consensus {i}"
                        )
                        st.success(f"🎯 **Majority Consensus**: [Download {filename}](./{filepath})")
                
                # Add rules-based results display for 3-way comparison
                elif result.get("rules_based_enabled") and result.get("verification_accuracy") is not None:
                    st.subheader("🔧 Rules-Based Comparison Results")
                    st.success(f"Rules-based comparison completed with {result['verification_accuracy']:.2f}% accuracy against rules-based approach")
                    
                    # Calculate per-field rules-based accuracy
                    if result.get("verification_diffs"):
                        rules_based_field_stats = {}
                        rules_based_merged_df = result.get("verification_merged_df")
                        
                        if rules_based_merged_df is not None:
                            matched_rules_based_rows = rules_based_merged_df[rules_based_merged_df['_merge'] == 'both']
                            
                            for field in FIELDS_TO_COMPARE:
                                field_total = len(matched_rules_based_rows)
                                field_mismatches = len(result["verification_diffs"].get(field, []))
                                field_correct = field_total - field_mismatches
                                field_accuracy = round(100 * field_correct / field_total, 2) if field_total > 0 else 0
                                rules_based_field_stats[field] = {
                                    "accuracy": field_accuracy,
                                    "correct": field_correct,
                                    "total": field_total,
                                    "mismatches": field_mismatches
                                }
                            
                            # Display per-field rules-based accuracy
                            st.subheader("Per-Field Rules-Based Accuracy")
                            rules_based_field_df = pd.DataFrame({
                                "Field": [field for field in rules_based_field_stats.keys()],
                                "Accuracy (%)": [stats["accuracy"] for stats in rules_based_field_stats.values()],
                                "Mismatches": [stats["mismatches"] for stats in rules_based_field_stats.values()],
                                "Total": [stats["total"] for stats in rules_based_field_stats.values()]
                            })
                            rules_based_field_df = rules_based_field_df.sort_values("Accuracy (%)")
                            st.dataframe(rules_based_field_df)
                            
                            # Display rules-based mismatches
                            st.subheader("Rules-Based Mismatches")
                            all_rules_based_mismatches = []
                            for field, mismatches in result["verification_diffs"].items():
                                for mismatch in mismatches:
                                    all_rules_based_mismatches.append({
                                        "Field": field,
                                        "Unit": mismatch['key'],
                                        "Majority Consensus Value": mismatch['api_value'],
                                        "Rules-Based Value": mismatch['rules_value']
                                    })
                            
                            if all_rules_based_mismatches:
                                # Convert all values to strings to avoid Arrow serialization issues
                                for mismatch in all_rules_based_mismatches:
                                    mismatch["Majority Consensus Value"] = str(mismatch["Majority Consensus Value"]) if mismatch["Majority Consensus Value"] is not None else "None"
                                    mismatch["Rules-Based Value"] = str(mismatch["Rules-Based Value"]) if mismatch["Rules-Based Value"] is not None else "None"
                                
                                rules_based_mismatches_df = pd.DataFrame(all_rules_based_mismatches)
                                st.dataframe(rules_based_mismatches_df)
                                
                                # Text list of rules-based differences
                                st.subheader("Text List of Rules-Based Differences")
                                for mismatch in all_rules_based_mismatches:
                                    st.write(f"Cell: {mismatch['Unit']}.{mismatch['Field']}, Majority Consensus: {mismatch['Majority Consensus Value']}, Rules-Based: {mismatch['Rules-Based Value']}")
                            else:
                                st.write("Perfect match between majority consensus and rules-based approach!")
                    
                    # Create and save majority consensus file for rules-based results
                    if result.get("majority_consensus_output"):
                        filepath, filename = create_download_file(
                            result["majority_consensus_output"], 
                            f"majority_consensus_{os.path.basename(result['file'])}.json",
                            f"Majority Consensus Rules {i}"
                        )
                        st.success(f"🎯 **Majority Consensus**: [Download {filename}](./{filepath})")
                
                # Display side-by-side comparison (optional, can be large)
                if result["merged_df"] is not None and st.checkbox("Show Detailed Comparison Table", key=f"show_table_{i}"):
                    st.dataframe(result["merged_df"])
    
    # Display downloads section if we have completed files in session state
    if st.session_state.download_data:
        st.subheader("📥 Download Individual File Results")
        st.info("Download results for individual files that have completed processing.")
        
        # Create columns for download buttons
        download_cols = st.columns(min(3, len(st.session_state.download_data)))
        
        for idx, (file_name, data) in enumerate(st.session_state.download_data.items()):
            col_idx = idx % len(download_cols)
            with download_cols[col_idx]:
                st.write(f"**{file_name}**")
                
                # Download comparison details
                st.download_button(
                    label=f"📊 Comparison Details",
                    data=json.dumps(data["diff_output"], indent=2, default=str),
                    file_name=f"comparison_diff_{file_name}.json",
                    mime="application/json",
                    key=f"persistent_download_diff_{file_name}",
                    help="Download detailed comparison results"
                )
                
                # Download API outputs
                result = data["result"]
                st.download_button(
                    label=f"🔧 API Output 1",
                    data=json.dumps(result["api_output1"], indent=2, default=str),
                    file_name=f"api1_output_{file_name}.json",
                    mime="application/json",
                    key=f"persistent_download_api1_{file_name}",
                    help="Download first API call output"
                )
                
                st.download_button(
                    label=f"🔧 API Output 2",
                    data=json.dumps(result["api_output2"], indent=2, default=str),
                    file_name=f"api2_output_{file_name}.json",
                    mime="application/json",
                    key=f"persistent_download_api2_{file_name}",
                    help="Download second API call output"
                )
                
                # Third API output for 3-way comparison
                if result.get("comparison_mode") == "3-way" and result.get("api_output3"):
                    st.download_button(
                        label=f"🔧 API Output 3",
                        data=json.dumps(result["api_output3"], indent=2, default=str),
                        file_name=f"api3_output_{file_name}.json",
                        mime="application/json",
                        key=f"persistent_download_api3_{file_name}",
                        help="Download third API call output"
                    )
                
                # Majority consensus output if available
                if result.get("majority_consensus_output"):
                    st.download_button(
                        label=f"🎯 Majority Consensus",
                        data=json.dumps(result["majority_consensus_output"], indent=2, default=str),
                        file_name=f"majority_consensus_{file_name}.json",
                        mime="application/json",
                        key=f"persistent_download_majority_{file_name}",
                        help="Download majority consensus output"
                    )
        
        # Clear downloads button
        if st.button("🗑️ Clear All Downloads", help="Clear all download data and reset the downloads section"):
            st.session_state.download_data = {}
            st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
