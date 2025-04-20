import pandas as pd
import json
from portkey_ai import Portkey
import openpyxl # Needed for pandas to read .xlsx
from datetime import datetime
import re # For potential string cleaning
import os
import pdfplumber
import tabula

# --- LLM Prompts ---

PROMPT_PART_1_CLAUDE = """ You are an expert real estate analyst specializing in rent roll document analysis. Your task is to examine a section of a rent roll document, 
accurately identify the existing column headers, and extract specific information for each unit in a token-efficient manner. First of all you need to understand the 
column headers of the rent roll. Carefully analyze what is present and what is not. Also understand the data structure and read/understand all the instrcutions 
carefully before providing the output. 1. Rent Roll Extract: - The rent roll extract is provided between the <RR> </RR> tags. This may be a partial section of a 
larger document. So it could cut off certain unit information. - 'NULL' is used to signify blank cells in the RR. Do not include NULL in your outputs because they 
are just blank cells. 2. Column Header Identification: - Carefully examine the first few rows of the data grid to identify the actual column headers. - Be aware 
that column headers may span multiple rows. Combine multi-row headers into a single string, separating parts with a space. - Only include headers that are explicitly 
present in the data grid. Do not infer or add headers that are not there. """

PROMPT_PART_2_GEMINI = """ 3. Data Extraction and Processing: - Identify Unit Blocks: - Recognize that information for a single unit may span multiple rows. - 
The first row of a unit block typically contains the unit identifier (e.g., unit number or letter) and primary lease information. - Subsequent rows without a unit 
identifier in the 'Unit' column are part of the same unit's data block. - Extract information for each unit entry in the provided section, based on the identified 
column headers. - Convert all dates to MM/DD/YYYY format without any time information. - In multi row rent rolls, capture all the rows to extract information related 
to that unit. - Do not try to alter or remove units or unit details or fix any mistakes. Report everything as it is. 4. Information to Extract: - row_index: It's 
the row number which is also given in RR, report it as it is. - unit_num: The unique identifier or the unit number for each rental unit as given in the rent roll. - 
unit_num can never be 'nan'. - Its very important you always capture the correct full unit_num even with spaces. - unit_num in your output units should have the 
same pattern as the units in RR_FIRST_30_OUTPUT. - Make sure to keep unit_num coherent and consistent between your current output and RR_FIRST_30_OUTPUT. - e.g. 
If RR_FIRST_30_OUTPUT has unit_num such as '608 501', you output should also have a similar type unit_num, not something like '601'. - For any unit numbers 
containing spaces. Replace those spaces with hyphen(-). - e.g. 'X Y' -> 'X-Y' (correct), 'X' (incorrect), 'Y' (incorrect) - e.g. '608 501' -> '608-501' (correct), 
'608' (incorrect), '501' (incorrect) - e.g. '22 701' -> '22-701' (correct), '22' (incorrect), '701' (incorrect) - unit_num can NEVER be 'nan'. - 
Even in multi-row rent rolls, unit_num is usually in the first row of the unit rows. - unit_type: The specific layout or type of the unit (e.g., 1BR/1BA, Studio). 
- unit_type can NEVER be 'nan'. If not directly provided, derive it accordingly. - It's like a code that is used to identify the type of the unit, floorplan, 
etc. - unit_type and unit_num are not the same thing. unit_type is not as unique as unit_num. - sqft: The total square footage of the unit. - br: The number 
of bedrooms in the unit. - Look carefully for this number in the unit. Sometimes they can be in different names. - Sometimes, It could be inside a breakdown. 
Look for words such as bed and bath. - Both br and bath could be in the same column like a floorplan or unit type. - Always give it as one decimal float. - 
If not provided within the rent roll, put 'nan' as the value. - bath: The number of bathrooms in the unit. - Look carefully for this number in the unit. 
Sometimes they can be in different names. - Sometimes, It could be inside a breakdown. Look for words such as bed and bath. - Both br and bath could be 
in the same column like a floorplan or unit type. - Always give it as one decimal float. - If not provided within the rent roll, put 'nan' as the value. - 
tenant: The name of the current tenant. Use 'nan' for vacant units. - move_in: Move in date is the date the tenant moved into the unit (MM/DD/YYYY format). - 
lease_start: The start date of the current lease agreement (MM/DD/YYYY format). - lease_end: The end date of the current lease agreement (MM/DD/YYYY format). 
- rent_charge: The actual base lease rent paid by the tenant. - This always exist in the rent roll and can NEVER be null. - Do not try to calculate or derive. 
Report as it is. - Do not confuse rent_charge with rent_market. - In case the rent roll has a charge_code_bd and it has the rent charge in it, pick this from 
the 'rent_charge'. - NEVER use a rent subsidy as 'rent_charge'. If a unit has a rent subsidy, that means 'rent_charge' is the resident's portion of the rent 
and not the net rent or rent subsidy. - rent_gov_subsidy: The portion of rent covered by any government housing assistance program or government subsidy, 
including Section 8, Housing Choice Vouchers, HAP payments, and any federal/state housing subsidies. - Look in columns such as HAP rent, section 8 etc. 
It could also be within the charge code breakdown. - In case there are no rent_gov_subsidy amount, make this 'nan'. - is_mtm: A flag indicating whether the 
unit is operating under a month-to-month (MTM) lease status rather than a fixed-term lease agreement. - Output 1 (true) to indicate MTM pricing is in 
effect: - The unit has MTM/MO/M-T-M flag or indicator in any status column - OR any MTM premium charges exist. i.e. 'mtm_charge' > 0 - OR lease dates show 
expired/holdover status continuing as MTM. - Output 0 (false) if there are no MTM pricing in effect. - IMPORTANT: Do not use 'nan' here. Just output 0 (false) 
if there are no MTM pricing in effect. - mtm_charge: Additional charges or premiums specifically applied when a unit operates on a month-to-month basis, 
separate from the base rent amount. - Must only exist when 'is_mtm' = 1 (true) - Check dedicated MTM columns, charge code breakdowns, or rent differentials - 
Must be converted to actual dollar amount if shown as percentage. e.g. 10% MTM premium on 1000 rent = 100 mtm_charge - If no MTM premium found, output 
'nan' for mtm_charge. - rent_market: This is the current market rent of a unit in that area. - Extract Market Rent from the column that provides it 
directly. - One good way to spot the Market Rent is for the same type of unit this is same most of the time. - Do not confuse 'rent_market' with 
'rent_charge'. - 'rent_market' is an assumed figure by the property owner while the 'rent_charge' is the actual amount paid by the tenat. - Market 
Rent represents the potential rent for the unit based on current market conditions, not necessarily what the current tenant is paying. - This 
value should be present for both occupied and vacant units. - If Market Rent value for a unit is not explicitly provided, do not attempt to 
infer or calculate it. In such cases, report it as 'nan'. 5. Column Mapping and Inference: - Map each piece of information to its actual 
column name from the data grid. - If a standard piece of information doesn't have a corresponding column in the original data, use the standard 
name for both parts of the tuple in col_map. - Do not infer or guess column names that are not present in the data grid. - If occupancy is not 
explicitly provided, derive it based on Current Lease Charge as described above. - IMPORTANT: In any case where there is no column name for a 
standard column, use the standard name for both the standard_column_name and actual_column_name_from_data_grid in the col_map. 6. Output Format: - 
Your response should be in JSON format, following the structure shown in the <EXAMPLE_OUTPUT_JSON> </EXAMPLE_OUTPUT_JSON> section. - Do not 
include any text outside of the JSON object in your response. - Do not use any markdown formatting in your response. - Respond with a JSON 
object containing "units". - IMPORTANT: I also need the row_index column for each unit in the output. - Follow the structure shown in the 
<EXAMPLE_OUTPUT_JSON> </EXAMPLE_OUTPUT_JSON> section. - Include only the JSON object in your response, without any additional text or formatting. 
- When you're using 'nan' for a certain value. ALWAYS use the string 'nan' (with quotes) instead of bare nan. - VERY IMPORTANT: Output all units 
from the RR. You are not allowed to miss any units. - IMPORTANT: Do not refrain from outputting certain units just because those units also 
contained in RR_FIRST_30_INPUT. - If the RR data structure is drastically different from the structure of 'RR_FIRST_30_INPUT', then just 
return an empty list for 'units'. - I'm also attaching the first 30 rows extract RR_FIRST_30_INPUT of the rent roll and your processed 
output RR_FIRST_30_OUTPUT for your reference. - This will give you an understanding of how you processed the input and column headers which 
might be missing from the current input. - I need you to keep the same headers exactly as the RR_FIRST_30_OUTPUT and keep the same 
structure. - In case RR_FIRST_30_INPUT and RR_FIRST_30_OUTPUT are not given this is your first input. """

# --- Configuration ---
PORTKEY_API_KEY = "3jxFpu/1D/jkEUJTB5YwWUuW6Knb" # Replace with your actual Portkey API key
GEMINI_VIRTUAL_KEY = "gemini-6563d9"
CLAUDE_VIRTUAL_KEY = "anthropic-virtu-9d07c3" # Assuming this is the correct virtual key for Claude Sonnet

FIELDS_TO_COMPARE = [
    "unit_num", "unit_type", "sqft", "br", "bath", "tenant", "move_in",
    "lease_start", "lease_end", "rent_charge", "rent_gov_subsidy",
    "is_mtm", "mtm_charge", "rent_market"
]

# --- Helper Functions ---

def find_header_row(df):
    """Finds the header row index by looking for 'unit' (case-insensitive)."""
    for i, row in df.iterrows():
        if any('unit' in str(cell).lower() for cell in row.dropna()):
            return i
    return None # Header row not found

def extract_rent_roll_string(file_path):
    """Reads the Excel or PDF file, finds the header row containing 'unit', and extracts data from the next row onwards."""
    try:
        # Check file extension to determine processing method
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            print(f"Processing PDF file: {file_path}")
            return extract_from_pdf(file_path)
        else:  # Default to Excel processing for .xlsx and other formats
            print(f"Processing Excel file: {file_path}")
            return extract_from_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def extract_from_excel(file_path):
    """Extract data from Excel file."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl', header=None)  # Read all data first

        header_row_index = find_header_row(df)

        if header_row_index is None:
            print(f"Warning: Could not find header row containing 'unit' in {file_path}. Reading entire sheet.")
            # Fallback to reading the whole sheet if header isn't found
            df.fillna('NULL', inplace=True)
            rr_string = df.to_string(index=False, header=False)
            return f"<RR>\n{rr_string}\n</RR>"

        # Read the relevant part of the sheet again, starting from the row *after* the header
        df_data = pd.read_excel(file_path, engine='openpyxl', header=header_row_index)

        # Convert only the data rows to string
        df_data.fillna('NULL', inplace=True)
        # Include header in the string representation for the LLM context
        df_subset = df.iloc[header_row_index:].copy()
        # Convert to string *before* filling NaN to avoid dtype issues
        df_subset = df_subset.astype(str)
        df_subset.fillna('NULL', inplace=True)
        rr_string = df_subset.to_string(index=False, header=False)  # Convert data including header row

        print(f"Extracted data starting from row {header_row_index + 1} (0-indexed).")
        return f"<RR>\n{rr_string}\n</RR>"
    except Exception as e:
        print(f"Error reading Excel file {file_path}: {e}")
        return None

def extract_from_pdf(file_path):
    """Extract data from PDF file using multiple methods and select the best result."""
    try:
        # Try tabula-py first (Java-based, good for structured tables)
        print("Attempting to extract tables with tabula-py...")
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        
        if tables and len(tables) > 0:
            # Find the table that likely contains the rent roll data (look for 'unit' column)
            rent_roll_table = None
            for table in tables:
                # Check if any column name contains 'unit' (case-insensitive)
                if any('unit' in str(col).lower() for col in table.columns):
                    rent_roll_table = table
                    break
                
                # If no column header contains 'unit', check the first few rows
                for i in range(min(5, len(table))):
                    if any('unit' in str(cell).lower() for cell in table.iloc[i]):
                        # Found 'unit' in a row, treat this as a potential header row
                        rent_roll_table = table
                        break
                
                if rent_roll_table is not None:
                    break
            
            if rent_roll_table is not None:
                # Process the found table
                rent_roll_table.fillna('NULL', inplace=True)
                rr_string = rent_roll_table.to_string(index=False)
                print("Successfully extracted table with tabula-py.")
                return f"<RR>\n{rr_string}\n</RR>"
        
        # If tabula-py didn't find a suitable table, try pdfplumber
        print("Attempting to extract tables with pdfplumber...")
        all_tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        
        if all_tables:
            # Find the table that likely contains the rent roll data
            rent_roll_table = None
            for table in all_tables:
                # Check if any cell in the first few rows contains 'unit' (case-insensitive)
                for row_idx, row in enumerate(table[:5]):  # Check first 5 rows
                    if any('unit' in str(cell).lower() for cell in row if cell):
                        rent_roll_table = table
                        break
                
                if rent_roll_table is not None:
                    break
            
            if rent_roll_table is not None:
                # Convert the table to a DataFrame
                df = pd.DataFrame(rent_roll_table[1:], columns=rent_roll_table[0])
                df.fillna('NULL', inplace=True)
                rr_string = df.to_string(index=False)
                print("Successfully extracted table with pdfplumber.")
                return f"<RR>\n{rr_string}\n</RR>"
        
        # If both methods failed, extract all text as a fallback
        print("Table extraction failed. Extracting all text as fallback...")
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        if text.strip():
            print("Extracted text from PDF (no structured tables found).")
            return f"<RR>\n{text}\n</RR>"
        
        print("Failed to extract any meaningful data from the PDF.")
        return None
    
    except Exception as e:
        print(f"Error extracting data from PDF {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def query_llm(portkey_client, prompt, model_virtual_key, model_name, provider_name):
    """Sends a prompt to the specified LLM via Portkey, including the provider header."""
    headers = {
        "x-portkey-provider": provider_name
    }
    try:
        # Set the virtual key directly on the client for this call
        portkey_client.virtual_key = model_virtual_key
        # Pass the provider header
        response = portkey_client.with_options(headers=headers).chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name, # Model name might be less critical when virtual key + provider is set
            max_tokens=63999,
            temperature=0 # For deterministic output
        )
        content = response.choices[0].message.content

        # --- Attempt JSON parsing only if expected (e.g., for Gemini) ---
        if provider_name == "google": # Assume only Google/Gemini is expected to return JSON for now
            print(f"Attempting to parse JSON response from {model_name} ({provider_name})...")
            
            # Save the raw response for debugging
            with open(f"raw_{provider_name}_response.txt", "w") as f:
                f.write(content)
            print(f"Saved raw response to raw_{provider_name}_response.txt")
            
            # Attempt to find JSON within potential markdown fences
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                json_str = match.group(1)
                print("Found JSON within code blocks")
            else:
                # Assume the whole content is JSON if no fences are found
                json_str = content
                print("No code blocks found, treating entire response as JSON")

            # Clean potential leading/trailing whitespace or non-JSON text before parsing
            json_str = json_str.strip()
            
            # Advanced JSON fixing
            try:
                # First attempt to fix common JSON errors
                # 1. Missing comma between objects
                json_str = re.sub(r'}\s*{', '},{', json_str)
                
                # 2. Missing comma after any key-value pair followed by another key
                # This is a more general version of the previous fix
                json_str = re.sub(r'(:\s*[^,{}\[\]]+)\s*\n\s*(")', r'\1,\n\2', json_str)
                
                # 3. Missing comma after a number followed by a key
                json_str = re.sub(r'(\d+)\s*\n\s*(")', r'\1,\n\2', json_str)
                
                # 4. Missing comma after a quoted string followed by a key
                json_str = re.sub(r'("[^"]*")\s*\n\s*(")', r'\1,\n\2', json_str)
                
                # 5. Missing comma after a boolean or null followed by a key
                json_str = re.sub(r'(true|false|null)\s*\n\s*(")', r'\1,\n\2', json_str)
                
                # 6. Fix trailing commas in arrays or objects
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                
                # 7. Fix missing quotes around keys
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                
                # Extract the JSON object if it's embedded in other text
                if not json_str.startswith('{') or not json_str.endswith('}'):
                    # If it doesn't look like a JSON object, try finding one within
                    start = json_str.find('{')
                    end = json_str.rfind('}')
                    if start != -1 and end != -1 and start < end:
                        json_str = json_str[start:end+1]
                        print(f"Extracted JSON object from position {start} to {end}")
                    else:
                        # Raise specific error if JSON structure not found
                        raise ValueError(f"Could not extract valid JSON object from {model_name} response.")
                
                # Save the fixed JSON for debugging
                with open("fixed_json.txt", "w") as f:
                    f.write(json_str)
                print("Saved fixed JSON to fixed_json.txt")
                
                # Try to parse the fixed JSON
                return json.loads(json_str)
                
            except json.JSONDecodeError as e:
                print(f"Initial JSON fixing failed: {e}")
                print("Attempting more aggressive JSON repair...")
                
                # More aggressive fixing for specific error cases
                error_msg = str(e)
                
                # Handle specific error: Expecting ',' delimiter
                if "Expecting ',' delimiter" in error_msg:
                    # Extract the line and column from the error message
                    match = re.search(r'line (\d+) column (\d+)', error_msg)
                    if match:
                        line_num = int(match.group(1))
                        col_num = int(match.group(2))
                        
                        # Split the JSON string into lines
                        lines = json_str.split('\n')
                        
                        # Make sure we have enough lines
                        if line_num <= len(lines):
                            # Get the problematic line
                            line = lines[line_num - 1]
                            
                            # Insert a comma at the specified column
                            if col_num <= len(line):
                                fixed_line = line[:col_num] + ',' + line[col_num:]
                                lines[line_num - 1] = fixed_line
                                
                                # Rejoin the lines
                                json_str = '\n'.join(lines)
                                print(f"Inserted comma at line {line_num}, column {col_num}")
                                
                                # Save the fixed JSON for debugging
                                with open("fixed_json_aggressive.txt", "w") as f:
                                    f.write(json_str)
                                print("Saved aggressively fixed JSON to fixed_json_aggressive.txt")
                                
                                # Try to parse the fixed JSON again
                                try:
                                    return json.loads(json_str)
                                except json.JSONDecodeError as e2:
                                    print(f"Aggressive JSON fixing also failed: {e2}")
                
                # If we get here, all fixing attempts have failed
                print("All JSON fixing attempts failed. Returning None.")
                return None
        else:
            # For other providers (like Claude here), return the raw content
            print(f"Returning raw text response from {model_name} ({provider_name}).")
            return content.strip() # Return raw string content

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {model_name} ({provider_name}): {e}")
        print("Raw response content:")
        print(content[:500] + "..." if len(content) > 500 else content)  # Print first 500 chars for debugging
        
        # Save the problematic response for offline analysis
        try:
            with open(f"error_{provider_name}_response.txt", "w") as f:
                f.write(content)
            print(f"Saved error response to error_{provider_name}_response.txt")
        except Exception as write_err:
            print(f"Could not save error response: {write_err}")
            
        return None
    except Exception as e:
        print(f"Error querying {model_name} via Portkey: {e}")
        return None

def normalize_value(value):
    """Normalizes values for comparison (string, lowercase, strip whitespace)."""
    if pd.isna(value):
        return "" # Treat NaN/None as empty string for comparison
    return str(value).strip().lower()

def compare_data(llm_output_json, verified_file_path):
    """Compares LLM JSON output with the verified Excel file."""
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

        # Convert LLM output JSON to DataFrame
        if not llm_output_json or "units" not in llm_output_json or not llm_output_json["units"]:
             print("Error: LLM output is empty or not in the expected format.")
             return None, None, None

        df_llm = pd.DataFrame(llm_output_json["units"])
        # Normalize LLM column names (already lowercase from prompt spec, but good practice)
        df_llm.columns = df_llm.columns.str.strip().str.lower()
        missing_llm_cols = {f.lower() for f in FIELDS_TO_COMPARE} - set(df_llm.columns)
        if missing_llm_cols:
             print(f"Warning: Missing expected columns in LLM output: {missing_llm_cols}")
             # Add missing columns with NaN to allow merge/comparison
             for col in missing_llm_cols:
                 df_llm[col] = pd.NA


        # --- Data Type Conversion & Cleaning ---
        # Convert date columns safely, coercing errors to NaT
        date_cols_llm = ['move_in', 'lease_start', 'lease_end']
        date_cols_verified = ['move_in', 'lease_start', 'lease_end'] # Assuming same names after lowercasing

        for col in date_cols_llm:
            if col in df_llm.columns:
                # Handle 'nan' strings before conversion
                df_llm[col] = df_llm[col].replace('nan', pd.NA)
                df_llm[col] = pd.to_datetime(df_llm[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                 df_llm[col] = pd.NA # Ensure column exists if missing

        for col in date_cols_verified:
            if col in df_verified.columns:
                df_verified[col] = pd.to_datetime(df_verified[col], errors='coerce').dt.strftime('%m/%d/%Y')
            else:
                df_verified[col] = pd.NA # Ensure column exists if missing

        # Convert numeric columns safely, coercing errors to NaN
        numeric_cols = ['sqft', 'br', 'bath', 'rent_charge', 'rent_gov_subsidy', 'mtm_charge', 'rent_market']
        for col in numeric_cols:
            if col in df_llm.columns:
                df_llm[col] = df_llm[col].replace('nan', pd.NA)
                df_llm[col] = pd.to_numeric(df_llm[col], errors='coerce')
            if col in df_verified.columns:
                df_verified[col] = pd.to_numeric(df_verified[col], errors='coerce')

        # Handle boolean 'is_mtm' (normalize 0/1, True/False, 'true'/'false' to 0/1)
        if 'is_mtm' in df_llm.columns:
            df_llm['is_mtm'] = df_llm['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_llm['is_mtm'] = pd.to_numeric(df_llm['is_mtm'], errors='coerce').fillna(0).astype(int) # Default to 0 (False) if conversion fails or missing
        if 'is_mtm' in df_verified.columns:
            df_verified['is_mtm'] = df_verified['is_mtm'].replace({'nan': pd.NA, 'true': 1, 'false': 0, True: 1, False: 0})
            df_verified['is_mtm'] = pd.to_numeric(df_verified['is_mtm'], errors='coerce').fillna(0).astype(int)


        # --- Construct Match Key ---
        # Check if move_in dates are available in LLM output
        has_move_in_dates = not df_llm['move_in'].isna().all() and not (df_llm['move_in'] == 'nan').all()
        
        if has_move_in_dates:
            # Use normalized tenant name and move-in date using vectorized operations
            print("Using tenant name and move-in date for matching...")
            tenant_col_v = df_verified.get('tenant', pd.Series(dtype=str)) # Get column or empty series
            move_in_col_v = df_verified.get('move_in', pd.Series(dtype=str))
            df_verified['match_key'] = tenant_col_v.fillna('').astype(str).str.strip().str.lower() + "|" + \
                                      move_in_col_v.fillna('').astype(str).str.strip().str.lower()

            tenant_col_l = df_llm.get('tenant', pd.Series(dtype=str))
            move_in_col_l = df_llm.get('move_in', pd.Series(dtype=str))
            df_llm['match_key'] = tenant_col_l.fillna('').astype(str).str.strip().str.lower() + "|" + \
                                 move_in_col_l.fillna('').astype(str).str.strip().str.lower()
        else:
            # Use only tenant name for matching if move_in dates are not available
            print("Move-in dates not available in LLM output. Using only tenant name for matching...")
            tenant_col_v = df_verified.get('tenant', pd.Series(dtype=str))
            df_verified['match_key'] = tenant_col_v.fillna('').astype(str).str.strip().str.lower()
            
            tenant_col_l = df_llm.get('tenant', pd.Series(dtype=str))
            df_llm['match_key'] = tenant_col_l.fillna('').astype(str).str.strip().str.lower()
            
            # Remove commas from tenant names for better matching
            df_verified['match_key'] = df_verified['match_key'].str.replace(',', '')
            df_llm['match_key'] = df_llm['match_key'].str.replace(',', '')

        # --- Merge and Compare ---
        merged = df_llm.merge(df_verified, on="match_key", suffixes=("_llm", "_verified"), how="outer", indicator=True)

        total_comparisons, correct_comparisons = 0, 0
        diffs = {}
        unmatched_llm = merged[merged['_merge'] == 'left_only']
        unmatched_verified = merged[merged['_merge'] == 'right_only']
        matched = merged[merged['_merge'] == 'both']

        print(f"\n--- Comparison Results ---")
        print(f"Matched Rows: {len(matched)}")
        print(f"Unmatched LLM Rows (in LLM output but not Verified): {len(unmatched_llm)}")
        print(f"Unmatched Verified Rows (in Verified but not LLM output): {len(unmatched_verified)}")

        for _, row in matched.iterrows():
            for field in FIELDS_TO_COMPARE:
                field_llm = f"{field}_llm"
                field_verified = f"{field}_verified"

                # Check if columns exist before trying to access
                val_llm = row.get(field_llm)
                val_verified = row.get(field_verified)

                # Normalize for comparison
                norm_llm = normalize_value(val_llm)
                norm_verified = normalize_value(val_verified)

                # Special handling for boolean 'is_mtm' (compare normalized 0/1)
                if field == 'is_mtm':
                    # Already converted to 0/1 int above
                    is_match = (int(val_llm) == int(val_verified))
                # Special handling for numeric fields (allow small tolerance?) - For now, exact match after normalization
                elif field in numeric_cols:
                     # Handle potential NaN comparison carefully
                     if pd.isna(val_llm) and pd.isna(val_verified):
                         is_match = True
                     elif pd.isna(val_llm) or pd.isna(val_verified):
                         is_match = False
                     else:
                         # Could add tolerance here if needed: abs(val_llm - val_verified) < tolerance
                         is_match = (float(val_llm) == float(val_verified))
                else: # Default string comparison
                    is_match = (norm_llm == norm_verified)

                total_comparisons += 1
                if is_match:
                    correct_comparisons += 1
                else:
                    diffs.setdefault(field, []).append({
                        "key": row["match_key"],
                        "llm_value": val_llm,
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
                    print(f"      LLM: '{mismatch['llm_value']}', Verified: '{mismatch['verified_value']}'")
                if len(mismatches) > 3:
                    print(f"    - ... ({len(mismatches) - 3} more mismatches)")

        # Report unmatched rows
        if not unmatched_llm.empty:
            print("\n--- Unmatched LLM Rows (Examples) ---")
            print(unmatched_llm[['match_key'] + [f for f in FIELDS_TO_COMPARE if f in unmatched_llm.columns]].head())

        if not unmatched_verified.empty:
            print("\n--- Unmatched Verified Rows (Examples) ---")
            print(unmatched_verified[['match_key'] + [f for f in FIELDS_TO_COMPARE if f in unmatched_verified.columns]].head())


        return accuracy, diffs, merged # Return merged for potential further analysis/export

    except FileNotFoundError:
        print(f"Error: Verified file not found at {verified_file_path}")
        return None, None, None
    except KeyError as e:
         print(f"Error: Missing expected key during processing: {e}")
         print("This might be due to unexpected LLM output format or missing columns in input files.")
         return None, None, None
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None, None, None


# --- Main Execution Logic ---

def main(raw_file, verified_file, output_diff_file=None):
    """Main function to run the verification process."""

    print(f"Processing raw file: {raw_file}")
    print(f"Comparing against verified file: {verified_file}")

    # Step 1: Extract Rent Roll String
    rr_string = extract_rent_roll_string(raw_file)
    if not rr_string:
        return # Error handled in function

    # Initialize Portkey Client
    # Assumes API key is handled by Portkey client (e.g., via env var or direct init)
    try:
        # Pass the key directly if needed, or rely on environment variables
        portkey = Portkey(api_key=PORTKEY_API_KEY)
    except Exception as e:
        print(f"Error initializing Portkey client: {e}")
        return

    # Step 2a: Query Gemini for first part (previously Claude)
    print("\nQuerying Gemini for first part (Step 2a)...")
    claude_model = "gemini-2.5-pro-exp-03-25" # Using Gemini as requested
    claude_prompt = PROMPT_PART_1_CLAUDE + rr_string
    # We call Gemini but don't use its direct output based on the second Gemini prompt's structure
    try:
        # Store Gemini's response (now expected as raw text)
        claude_response_text = query_llm(portkey, claude_prompt, GEMINI_VIRTUAL_KEY, claude_model, "google")
        if claude_response_text is not None:
             # Check if it's a string and not empty
            if isinstance(claude_response_text, str) and claude_response_text:
                print("Claude query successful (returned text, not directly used in next step).")
                # print("Claude Raw Text Response:\n", claude_response_text[:500] + "...") # Print snippet for debug
            else:
                 # Handle cases where query_llm might return None even for non-JSON providers on error
                 print("Claude query executed but returned None or empty response.")
        else:
             print("Claude query failed or returned None.")
            # Decide if this is a fatal error or if we can proceed to Gemini anyway
            # For now, let's proceed to Gemini even if Claude fails, as Gemini uses the original rr_string
    except Exception as e:
        print(f"Error during Claude query: {e}. Proceeding to Gemini...")


    # Step 2b: Query Gemini
    print("Querying Gemini (Step 2b)...")
    gemini_model = "gemini-2.5-flash-preview-04-17" # Using gemini-2.5-pro-exp-03-25 as requested
    # Gemini prompt uses the original rr_string according to its content
    gemini_prompt = PROMPT_PART_2_GEMINI + rr_string
    gemini_json_output = query_llm(portkey, gemini_prompt, GEMINI_VIRTUAL_KEY, gemini_model, "google")

    if not gemini_json_output:
        print("Failed to get valid JSON output from Gemini. Aborting comparison.")
        return

    # Save intermediate LLM output for inspection
    llm_output_filename = "llm_output.json"
    try:
        with open(llm_output_filename, 'w') as f:
            json.dump(gemini_json_output, f, indent=2)
        print(f"\nSaved Gemini JSON output to {llm_output_filename}")
    except Exception as e:
        print(f"Warning: Could not save LLM output JSON: {e}")


    # Step 3: Compare LLM output with Verified Data
    print("\nComparing LLM output with verified data...")
    accuracy, diffs, merged_df = compare_data(gemini_json_output, verified_file)

    if accuracy is not None and diffs is not None and output_diff_file:
        print(f"\nSaving differences to {output_diff_file}...")
        try:
            # Save detailed differences
            diff_output = {
                "accuracy_percent": accuracy,
                "field_mismatches": diffs,
                # Optionally include unmatched rows if needed
                # "unmatched_llm_rows": merged_df[merged_df['_merge'] == 'left_only'].to_dict('records'),
                # "unmatched_verified_rows": merged_df[merged_df['_merge'] == 'right_only'].to_dict('records')
            }
            with open(output_diff_file, 'w') as f:
                json.dump(diff_output, f, indent=2, default=str) # Use default=str for non-serializable types like NaT
            print("Differences saved.")
        except Exception as e:
            print(f"Error saving differences JSON: {e}")

    print("\nVerification process complete.")


# --- Streamlit App ---
# This function is called when the script is run via streamlit
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(layout="wide")
    st.title("Rent Roll LLM Verifier")

    st.info("Upload the raw rent roll (.xlsx) and the verified version (.xlsx) to compare LLM parsing accuracy.")

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
            with st.spinner("Processing files and querying LLMs... This may take a minute."):
                # --- Re-run the main logic within Streamlit context ---
                rr_string_st = extract_rent_roll_string(raw_temp_path)
                if not rr_string_st:
                    st.error("Failed to read the raw rent roll file.")
                else:
                    try:
                        portkey_st = Portkey(api_key=PORTKEY_API_KEY)

                        # Query Claude first (as in the main pipeline)
                        st.write("Querying Claude...")
                        claude_model = "claude-3-7-sonnet-latest"
                        claude_prompt = PROMPT_PART_1_CLAUDE + rr_string_st
                        claude_response_text = query_llm(portkey_st, claude_prompt, CLAUDE_VIRTUAL_KEY, claude_model, "anthropic")
                        
                        if claude_response_text is not None and isinstance(claude_response_text, str) and claude_response_text:
                            st.write("Claude query successful.")
                        else:
                            st.warning("Claude query returned empty or invalid response. Proceeding with Gemini only.")

                        # Query Gemini
                        st.write("Querying Gemini...")
                        gemini_model = "gemini-2.5-flash-preview-04-17"
                        gemini_prompt = PROMPT_PART_2_GEMINI + rr_string_st
                        gemini_json_output_st = query_llm(portkey_st, gemini_prompt, GEMINI_VIRTUAL_KEY, gemini_model, "google")

                        if not gemini_json_output_st:
                            st.error("Failed to get valid JSON output from Gemini.")
                        else:
                            st.write("Gemini query successful. Comparing data...")
                            
                            # Save LLM output for reference
                            llm_output_filename = f"temp_llm_output_{uploaded_raw_file.name}.json"
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
                        import traceback
                        st.text(traceback.format_exc())

            # Clean up temp files
            import os
            try:
                os.remove(raw_temp_path)
                os.remove(verified_temp_path)
                os.remove(llm_output_filename)
            except OSError as e:
                st.warning(f"Could not remove temporary files: {e}")

# This is a special check to determine if we're running in Streamlit
# It will be set to True when imported by Streamlit
# and False when run directly as a script
_STREAMLIT_MODE = False

def set_streamlit_mode():
    """Set the global flag to indicate we're running in Streamlit mode"""
    global _STREAMLIT_MODE
    _STREAMLIT_MODE = True

if __name__ == "__main__":
    # Check if running via Streamlit by looking for the streamlit module in sys.modules
    import sys
    
    # If 'streamlit' is in sys.modules, we're being run by Streamlit
    if 'streamlit' in sys.modules:
        # We're running in Streamlit mode
        try:
            import streamlit as st
            # Set the flag to indicate we're in Streamlit mode
            set_streamlit_mode()
            # Run the Streamlit app
            run_streamlit_app()
        except ImportError:
            print("Streamlit is not installed. Please install it with: pip install streamlit")
    else:
        # We're running as a regular Python script
        # Check if command-line arguments are provided
        if len(sys.argv) > 1:
            # Use command-line arguments
            raw_file = sys.argv[1]
            verified_file = sys.argv[2] if len(sys.argv) > 2 else "Saddlebrook I RR 02-23-24.xlsx Verified.xlsx"
            output_diff_file = sys.argv[3] if len(sys.argv) > 3 else "comparison_diff.json"
            
            print(f"Using command-line arguments:")
            print(f"Raw file: {raw_file}")
            print(f"Verified file: {verified_file}")
            print(f"Output diff file: {output_diff_file}")
            
            main(raw_file, verified_file, output_diff_file)
        else:
            # Use the provided sample files
            RAW_RENT_ROLL_FILE = "Saddlebrook I RR 02-23-24.xlsx"
            VERIFIED_RENT_ROLL_FILE = "Saddlebrook I RR 02-23-24.xlsx Verified.xlsx"
            OUTPUT_DIFF_JSON = "comparison_diff.json" # Optional: Set to None to disable saving diffs
            
            main(RAW_RENT_ROLL_FILE, VERIFIED_RENT_ROLL_FILE, OUTPUT_DIFF_JSON)

# End of script
