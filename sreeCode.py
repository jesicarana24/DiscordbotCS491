import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import math
import numpy as np
import time

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI clients
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')  # Initialize the OpenAI client with the correct API key

# Define the index name
index_name = 'capstone-project'
index = pc.Index(index_name)

# Step 1: Function to generate a query embedding from OpenAI
def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            
            # Ensure the embedding length is 1536
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")

            return embedding
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"Waiting for {wait} seconds before retrying...")
                time.sleep(wait)
            else:
                print("Maximum retries exceeded.")
                raise

# Step 2: Function to query Pinecone with an embedding
def query_pinecone(query_embedding):
    # Convert embedding to a list of floats to ensure proper formatting
    query_vector = [float(e) for e in query_embedding]
    
    # Query Pinecone for relevant vectors (adjust `top_k` as needed)
    result = index.query(
        vector=query_vector,
        top_k=3,  # Retrieve the top 3 most similar embeddings
        include_values=True,  # Retrieve values (embeddings)
        include_metadata=True  # Retrieve metadata
    )
    
    return result

# Step 3: Function to read questions and answers from a file
def load_questions_from_file(file_path):
    qa_pairs = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        current_question = None
        current_answer = None
        
        for line in lines:
            if line.startswith("Question:"):
                if current_question and current_answer:
                    qa_pairs.append({'question': current_question, 'answer': current_answer})
                current_question = line.strip().replace("Question: ", "")
                current_answer = None
            elif line.startswith("Answer:"):
                current_answer = line.strip().replace("Answer: ", "")
        
        # Append the last QA pair
        if current_question and current_answer:
            qa_pairs.append({'question': current_question, 'answer': current_answer})
    
    return qa_pairs

# Step 4: Find the most relevant answer from the loaded QA pairs
def find_answer_from_file(qa_pairs, query):
    for pair in qa_pairs:
        if query.lower() in pair['question'].lower():
            return pair['answer']
    return "Sorry, I don't have an answer for that."

# Step 5: Generate a response based on retrieved documents from Pinecone or file
def generate_response(retrieved_docs, query, qa_pairs):
    # Try to find an answer from the file
    answer_from_file = find_answer_from_file(qa_pairs, query)
    
    if answer_from_file:
        # If an answer is found in the file, respond with that
        return f"Hello, the answer to your question is: {answer_from_file}"
    
    # If no answer from file, fall back to Pinecone query result
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('Question', 'N/A')} Answer: {doc['metadata'].get('Answer', 'N/A')}"
        for doc in retrieved_docs['matches']
    ])
    
    # Use the combined documents and the query to form the prompt
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}."
    
    # Use OpenAI's chat-completion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use 'gpt-4' if needed
        messages=[
            {"role": "system", "content": "You are an assistant that helps answer questions based on retrieved documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200  # Adjust the token limit based on your needs
    )
    
    return response['choices'][0]['message']['content'].strip()

# Example Usage
if __name__ == "__main__":
    # Define the query you want to search for
    query_text = "what are the capstone project options?"
    
    # Load questions and answers from the file
    qa_pairs = load_questions_from_file('message.txt')
    
    # Step 1: Get the embedding for the query
    query_embedding = get_embedding(query_text)
    
    # Step 2: Query Pinecone using the query embedding
    retrieved_documents = query_pinecone(query_embedding)
    
    # Step 3: Generate a response based on the retrieved documents and include a hello greeting
    generated_response = generate_response(retrieved_documents, query_text, qa_pairs)
    
    # Output the generated response
    print(f"Generated response:\n{generated_response}")