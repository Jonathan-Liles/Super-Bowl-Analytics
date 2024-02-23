import pandas as pd
import re

# Specify the path to your CSV file
csv_file_path = 'Final Keywords_2024.csv'

# Use pandas to read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Now you can work with the DataFrame (e.g., display it, manipulate data, etc.)
location_column = df['location']

# Dictionary of state abbreviations
state_abbreviations = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

# Result list to store all detected states
detected_states = []

# Function to check if a string contains a state or its abbreviation
def check_for_state(input_str):
    try:
        if input_str is not None:
            # Check for both state names and abbreviations
            for state_code, state_name in state_abbreviations.items():
                if re.search(rf'\b{state_name}\b', input_str, flags=re.IGNORECASE):
                    return state_name
                elif re.search(rf'\b{state_code}\b', input_str, flags=re.IGNORECASE):
                    return state_name
            # If no state is found, return None
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Iterate through the data
for location in location_column:
    try:
        # Check for a state in the location
        state = check_for_state(location)
        
        # If a state is found, append it to the list
        if state:
            detected_states.append(state)
    except Exception as e:
        print(f"Skipped '{location}' due to an error: {e}")
        continue

# Create a DataFrame with all detected states
detected_states_df = pd.DataFrame({'State': detected_states})

# Specify the path to your new CSV file
output_csv_file_path = 'all_detected_states.csv'

# Use pandas to write the DataFrame to a new CSV file
detected_states_df.to_csv(output_csv_file_path, index=False)

print(f'All detected states have been written to {output_csv_file_path}')