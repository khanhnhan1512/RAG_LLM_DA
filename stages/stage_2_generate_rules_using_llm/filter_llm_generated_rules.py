import pandas as pd

def filter_llm_generated_rules(file_path, thresholds):
    """
    Filters LLM generated rules based on given thresholds for specified columns.

    Parameters:
    - file_path: str, path to the CSV file containing LLM generated rules.
    - thresholds: dict, mapping of column names to their corresponding threshold values.

    Returns:
    - None; writes two files: one for kept content and one for eliminated content.
    """
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter based on thresholds
    temp_df = df.copy()

    # Apply filtering based on each threshold
    for column, threshold in thresholds.items():
        if column in temp_df.columns:
            # Calculate the cutoff value based on the threshold
            cutoff_index = int(len(temp_df) * threshold)
            # Sort by the column and keep the top values
            temp_df = temp_df.nlargest(cutoff_index, column)

    # Get kept and eliminated content
    kept_content = temp_df
    eliminated_content = df[~df.index.isin(temp_df.index)]

    # Define output file paths
    kept_file_path = 'result/icews14/stage_2/kept_llm_generated_rules.csv'
    eliminated_file_path = 'result/icews14/stage_2/eliminated_llm_generated_rules.csv'

    # Write kept content to a new CSV file
    kept_content.to_csv(kept_file_path, index=False)

    # Write eliminated content to a new CSV file
    eliminated_content.to_csv(eliminated_file_path, index=False)

# Example usage
file_path = 'result/icews14/stage_2/20241219_llm_generated_llm_rules.csv'
thresholds = {
    'confidence_score': 0.9,
    'body_supp_count': 0.9
}

filter_llm_generated_rules(file_path, thresholds)
