import pandas as pd
import glob


def average_csv_files_exclude_columns(input_folder, output_file):
    """
    Averages the numeric content of CSV files in a folder, excluding the first two columns.

    Parameters:
        input_folder (str): The folder containing input CSV files.
        output_file (str): The path for the resulting averaged CSV file.
    """
    # Get a list of all CSV files in the folder
    csv_files = glob.glob(f"{input_folder}/*.csv")
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Initialize variables for processing
    text_columns = None
    numeric_sum_df = None
    count = 0

    for file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)

            # Split the text and numeric columns
            text_part = df.iloc[:, :2]  # First two columns (text)
            numeric_part = df.iloc[:, 2:]  # Remaining columns (numeric)

            # Ensure text columns are consistent across files
            if text_columns is None:
                text_columns = text_part
            elif not text_part.equals(text_columns):
                raise ValueError(f"Inconsistent text columns in file: {file}")

            # Accumulate numeric columns
            if numeric_sum_df is None:
                numeric_sum_df = numeric_part
            else:
                numeric_sum_df += numeric_part
            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if count == 0:
        print("No valid CSV files were processed.")
        return

    # Average the numeric columns
    numeric_average_df = numeric_sum_df / count

    # Combine text and averaged numeric columns
    result_df = pd.concat([text_columns, numeric_average_df], axis=1)

    # Save the result to the output file
    result_df.to_csv(output_file, index=False)
    print(f"Averaged CSV saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_folder = "../data/Walking_Results"
    output_file = "../data/Walking_Results/averaged_results.csv"
    average_csv_files_exclude_columns(input_folder, output_file)
