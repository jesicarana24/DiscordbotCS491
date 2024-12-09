import pandas as pd

def process_unorganized_data(df):
    """
    Organize unstructured FAQ data into a clean, structured format.

    Args:
        df (pd.DataFrame): DataFrame containing unorganized FAQ data with the following required columns:
            - 'Question'
            - 'Answer'
            - 'Context'
            - 'Intent'
            - 'Entities'
            - 'Difficulty'
            - 'Volatility Level'
            - 'Comments'

    Returns:
        pd.DataFrame: Structured DataFrame with the same columns.
    """
    required_columns = [
        'Question', 'Answer', 'Context', 'Intent', 
        'Entities', 'Difficulty', 'Volatility Level', 'Comments'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    # Ensure only required columns are present and in the correct order
    organized_df = df[required_columns].copy()

    return organized_df

# Load unorganized data
try:
    unorganized_df = pd.read_csv('faq-data.csv')
    print("[INFO] Unorganized data loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("The file 'faq-data.csv' was not found. Please ensure it exists in the working directory.")

# Process data
try:
    organized_df = process_unorganized_data(unorganized_df)
    organized_df.to_csv('organized_faq_data.csv', index=False)
    print("[INFO] Organized data saved to 'organized_faq_data.csv'.")
    print(organized_df)
except ValueError as ve:
    print(f"[ERROR] {ve}")
