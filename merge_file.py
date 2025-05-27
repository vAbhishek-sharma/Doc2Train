import pandas as pd
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Folder containing CSV files
folder_path = "./to_merge"
output_file = f"./merged/merged_output_{timestamp}.csv"

# Get list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Merge all CSV files
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into {output_file}")
