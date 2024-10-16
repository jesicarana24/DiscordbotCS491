import openai
from responses import retrieve_documents, generate_response
import math
from pinecone import Pinecone, ServerlessSpec

# Function to check if a vector contains valid numbers
def is_valid_vector(vector):
    return all(math.isfinite(v) for v in vector)

def test_retrieval_and_response(query):
    try:
        # Retrieve documents using the query
        retrieved_docs = retrieve_documents(query)

        # Check if the returned vectors are valid
        if 'matches' in retrieved_docs and len(retrieved_docs['matches']) > 0:
            generated_answer = generate_response(retrieved_docs, query)
            print(f"Query: {query}")
            print(f"Generated response: {generated_answer}")
        else:
            print(f"No relevant documents found for query: {query}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Test the document retrieval and response generation
if __name__ == '__main__':  
    queries = [
        "What is the startup you co-founded?",
        "How do I submit my project proposal?",
        "When is the final presentation due?"
    ]

    for query in queries:
        test_retrieval_and_response(query)
        print("-" * 50)  # Separator between queries
