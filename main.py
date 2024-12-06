import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import math
import numpy as np
import openai
import time

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = 'capstone-project'
index = pc.Index(index_name)

def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
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

def query_pinecone(query_embedding):
    query_vector = [float(e) for e in query_embedding]
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_values=True,
        include_metadata=True
    )
    return result

def generate_response(retrieved_docs, query):
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('Question', 'N/A')} Answer: {doc['metadata'].get('Answer', 'N/A')}"
        for doc in retrieved_docs['matches']
    ])
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps answer questions based on retrieved documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} is now running!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    user_message = str(message.content).lower()
    if user_message in ["hey", "hello", "hi"]:
        await message.channel.send("Hello! How can I help you today?")
        return
    try:
        query_embedding = get_embedding(user_message)
        retrieved_docs = query_pinecone(query_embedding)
        response = generate_response(retrieved_docs, user_message)
        await message.channel.send(response)
    except Exception as e:
        await message.channel.send(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    client.run(TOKEN)
