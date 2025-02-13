import pandas as pd

# File paths
input_file = "combined_depth.csv"
output_file = input_file

try:
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Ensure required columns exist in the input file
    required_columns = ['pred_range', 'Est_pixel_length', 'Length']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the input file.")
            exit()

    # Initialize lists to store calculated values
    est_length_values = []
    diff_length_values = []

    # Iterate through the DataFrame row by row
    for index, row in df.iterrows():
        try:
            # Convert values to float for calculation
            scaled_pred_range = float(row['pred_range'])
            est_pixel_length = float(row['Est_pixel_length'])
            length = float(row['Length'])

            # Calculate est_length
            est_length = scaled_pred_range * est_pixel_length*0.9 / 2100
            est_length_values.append(est_length)

            # Calculate absolute difference
            diff_length = abs(length - est_length)
            diff_length_values.append(diff_length)
        except ValueError:
            # Handle non-numeric or missing data by assigning NaN
            est_length_values.append(None)
            diff_length_values.append(None)

    # Add the calculated values as new columns
    df['est_length'] = est_length_values
    df['diff_length'] = diff_length_values

    # Save the updated DataFrame back to the file
    df.to_csv(output_file, index=False)
    print(f"The 'est_length' and 'diff_length' columns have been calculated and saved to {output_file}.")

except FileNotFoundError:
    print(f"Error: The file {input_file} does not exist. Please verify the path.")
except Exception as e:
    print(f"An error occurred: {e}")
