import pandas as pd

def process_unorganized_data(df):
    """
    This function takes unorganized data from a DataFrame and organizes it into a structured DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing unorganized data.

    Returns:
        pd.DataFrame: Organized DataFrame.
    """
    # Initialize a dictionary to store structured data
    structured_data = {
        'Question': [],
        'Answer': [],
        'Context': [],
        'Intent': [],
        'Entities': [],
        'Difficulty': [],
        'Volatility Level': [],
        'Comments': []
    }

    # Iterate over each row in the DataFrame and append it to the corresponding list
    for _, row in df.iterrows():
        structured_data['Question'].append(row['Question'])
        structured_data['Answer'].append(row['Answer'])
        structured_data['Context'].append(row['Context'])
        structured_data['Intent'].append(row['Intent'])
        structured_data['Entities'].append(row['Entities'])
        structured_data['Difficulty'].append(row['Difficulty'])
        structured_data['Volatility Level'].append(row['Volatility Level'])
        structured_data['Comments'].append(row['Comments'])

    # Convert the structured data into a DataFrame
    organized_df = pd.DataFrame(structured_data)
    return organized_df

# Load the unorganized data from the CSV file
unorganized_df = pd.read_csv('faq-data.csv')

# Process the unorganized data into a structured DataFrame
organized_df = process_unorganized_data(unorganized_df)

# Save the organized DataFrame to a new CSV file
organized_df.to_csv('organized_faq_data.csv', index=False)

# Print the organized DataFrame to view it in the terminal
print(organized_df)
