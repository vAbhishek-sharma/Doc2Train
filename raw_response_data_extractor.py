import os
import csv
import json

def extract_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    user_input_split = content.split("=== User Input ===\n", 1)
    if len(user_input_split) < 2:
        return None  # Skip files that don't have the expected format

    section_data, remainder = user_input_split[1].split("=== AI Response ===\n", 1)
    json_data = remainder.strip()

    try:
        json.loads(json_data)  # Validate JSON format
    except json.JSONDecodeError:
        return None  # Skip files with invalid JSON

    return section_data.strip(), json_data, content.strip()

def process_files(directory, output_csv):
    rows = []

    for index, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            extracted_data = extract_data_from_file(file_path)
            if extracted_data:
                section_data, json_data, text_data = extracted_data
                rows.append([index + 1, section_data, json_data, text_data])

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "section_data", "json_data", "text_data"])
        writer.writerows(rows)

    print(f"CSV file saved: {output_csv}")




# Get user input
directory = input("Enter the path to the directory containing .txt files: ")
output_csv = input("Enter the output CSV file name: ")

process_files(directory, output_csv)
