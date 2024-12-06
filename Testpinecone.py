import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = 'capstone-project'
index = pc.Index(index_name)

def retrieve_vector_data(vector_id):
    result = index.fetch(ids=[vector_id])
    return result

if __name__ == "__main__":
    vector_id = ["row-9","row-13"]  
    result = retrieve_vector_data(vector_id)

    print(f"Vector ID: {vector_id}")
    if vector_id in result['vectors']:
        data = result['vectors'][vector_id]
        metadata = data.get('metadata', {})
        
        question = metadata.get('Question', 'No question found in metadata')
        answer = metadata.get('Answer', 'No answer found in metadata')

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        print(f"Full Metadata: {metadata}")  
    else:
        print(f"No data found for vector ID: {vector_id}")
