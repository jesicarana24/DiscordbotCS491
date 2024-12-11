import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import time
import openai

# Load environment variables
load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

# Pinecone setup
index_name = 'capstone-project-jesica-2'
index = pc.Index(index_name)

# Generate embeddings for a text input
def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                raise e

# Query Pinecone for answers
def query_pinecone(query_embedding, top_k=3):
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result

# Generate a response using GPT
def generate_response(retrieved_docs, query):
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('Question', 'N/A')} Answer: {doc['metadata'].get('Answer', 'N/A')}"
        for doc in retrieved_docs['matches']
    ])
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}."
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps answer questions based on retrieved documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

# Upsert new Q&A pairs to Pinecone
def upsert_question_answer(question, answer):
    embedding = get_embedding(question)
    unique_id = f"qa-{hash(question)}"
    index.upsert(vectors=[{
        "id": unique_id,
        "values": embedding,
        "metadata": {"Question": question, "Answer": answer}
    }])

# Discord Bot Setup
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

    user_message = str(message.content).strip()

    # Basic greeting
    if user_message.lower() in ["hi", "hello", "hey"]:
        await message.channel.send("Hello!")
        return

    # Explicit learning command
    if user_message.lower() == "i want you to learn something":
        await message.channel.send("Sure! What's the question you'd like me to learn?")
        new_question = await client.wait_for("message", check=lambda m: m.author == message.author)
        await message.channel.send("Got it. What's the answer to that question?")
        new_answer = await client.wait_for("message", check=lambda m: m.author == message.author)

        # Add new Q&A to Pinecone
        upsert_question_answer(new_question.content, new_answer.content)
        await message.channel.send("Got it! I've learned something new.")
        return

    # Default behavior: Query Pinecone for answers
    try:
        query_embedding = get_embedding(user_message)
        retrieved_docs = query_pinecone(query_embedding)
        
        if retrieved_docs['matches'] and retrieved_docs['matches'][0]['score'] > 0.8:
            response = generate_response(retrieved_docs, user_message)
            await message.channel.send(response)
        else:
            await message.channel.send("I don't know the answer yet. Can you teach me?")
            new_answer = await client.wait_for("message", check=lambda m: m.author == message.author)
            upsert_question_answer(user_message, new_answer.content)
            await message.channel.send("Got it! I've learned something new.")
    except Exception as e:
        await message.channel.send(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    client.run(TOKEN)
