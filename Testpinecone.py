import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define the index name
index_name = 'capstone-project'
index = pc.Index(index_name)

# Function to retrieve specific vector and metadata by ID
def retrieve_vector_data(vector_id):
    result = index.fetch(ids=[vector_id])
    return result

# Example usage
if __name__ == "__main__":
    vector_id = ["row-9","row-13"]  # Replace with the vector ID you want to retrieve
    result = retrieve_vector_data(vector_id)

    print(f"Vector ID: {vector_id}")
    if vector_id in result['vectors']:
        data = result['vectors'][vector_id]
        metadata = data.get('metadata', {})
        
        # Extract the question and answer from metadata if available
        question = metadata.get('Question', 'No question found in metadata')
        answer = metadata.get('Answer', 'No answer found in metadata')

        # Display the question and answer
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        # If you want to display all metadata
        print(f"Full Metadata: {metadata}")  
    else:
        print(f"No data found for vector ID: {vector_id}")
