import os
import pandas as pd

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base directory relative to the script's location
base_dir = os.path.join(script_dir, "../TE_new/")

# Modes and their file structure
modes = [f"mode{i}_new" for i in range(1, 7)]

# Initialize a list to hold results
results = []

# Loop through each mode folder
for mode in modes:
    mode_path = os.path.join(base_dir, mode)
    
    # Check if the mode directory exists
    if not os.path.exists(mode_path):
        print(f"Directory {mode_path} does not exist. Skipping.")
        continue
    
    # Process each CSV file in the mode directory
    for file in os.listdir(mode_path):
        if file.endswith(".csv"):
            file_path = os.path.join(mode_path, file)
            try:
                # Read CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Ignore the first row (label row)
                data = df.iloc[1:]
                
                # Calculate average dimensionality
                avg_dimensionality = data.apply(lambda row: row.count(), axis=1).mean()
                
                # Record result
                results.append({
                    "Mode": mode,
                    "File": file,
                    "Average_Dimensionality": avg_dimensionality,
                    "Less_Than_53": avg_dimensionality < 53
                })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Convert results to a DataFrame for better visibility
if results:
    results_df = pd.DataFrame(results)

    # Print the results
    print("\nDimensionality Analysis Results:")
    print(results_df)

    # Save results to a CSV file
    output_file = os.path.join(script_dir, "dimensionality_analysis_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults have been saved to {output_file}")

    # Check for any files with dimensionality less than 53
    if results_df["Less_Than_53"].any():
        print("Some files have dimensionality less than 53. Check the results table for details.")
    else:
        print("All files have dimensionality of 53 or greater.")
else:
    print("No valid files processed.")