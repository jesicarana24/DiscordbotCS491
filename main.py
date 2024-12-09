import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import pandas as pd
import asyncio
from datetime import datetime, timezone

load_dotenv()

# Initialize Pinecone and OpenAI API keys
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = 'capstone-project'
index = pc.Index(index_name)

async def get_embedding(text, retries=3, backoff_factor=60):
    """Generate embedding using OpenAI."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")
            return embedding
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"[DEBUG] Retrying in {wait} seconds...")
                await asyncio.sleep(wait)
            else:
                print("[ERROR] Maximum retries exceeded.")
                return None

async def upsert_dataset_to_pinecone(file_path):
    """Upsert the FAQ dataset into Pinecone."""
    try:
        print(f"[INFO] Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            question = row.get('Question', '').strip()
            answer = row.get('Answer', '').strip()
            context = row.get('Context', 'General').strip()

            if not question or not answer:
                print(f"[WARNING] Skipping incomplete entry: {row}")
                continue

            metadata = {
                "OriginalQuery": question,
                "CorrectedAnswer": answer,
                "Context": context,
                "Timestamp": datetime.now(timezone.utc).isoformat()
            }

            embedding = await get_embedding(question)
            if embedding:
                vector_id = f"query-{hash(question)}"
                index.upsert(vectors=[(vector_id, embedding, metadata)])
                print(f"[DEBUG] Upserted: {question}")
            else:
                print(f"[ERROR] Skipping question due to embedding failure: {question}")
        print("[INFO] Dataset successfully upserted to Pinecone.")
    except Exception as e:
        print(f"[ERROR] Failed to upsert dataset: {e}")

async def upsert_to_pinecone(query, corrected_answer):
    """Upsert or overwrite a corrected query and response into Pinecone."""
    try:
        embedding = await get_embedding(query)
        if embedding is None:
            print("[ERROR] Skipping upsert due to missing embedding.")
            return

        vector_id = f"query-{hash(query)}"
        metadata = {
            "OriginalQuery": query,
            "CorrectedAnswer": corrected_answer,
            "Timestamp": datetime.now(timezone.utc).isoformat()
        }

        index.upsert(vectors=[(vector_id, embedding, metadata)])
        print(f"[DEBUG] Upserted correction: Query: '{query}', Answer: '{corrected_answer}'")
    except Exception as e:
        print(f"[ERROR] Failed to upsert correction to Pinecone: {e}")

async def find_correction_in_pinecone(query):
    """Retrieve the latest corrected response for a query from Pinecone."""
    try:
        embedding = await get_embedding(query)
        if embedding is None:
            print("[ERROR] Skipping correction lookup due to missing embedding.")
            return None

        result = index.query(
            vector=embedding,
            top_k=5, 
            include_metadata=True
        )

        corrections = [
            match["metadata"]
            for match in result["matches"]
            if "CorrectedAnswer" in match["metadata"]
        ]

        if corrections:
            corrections.sort(key=lambda x: x["Timestamp"], reverse=True)
            latest_correction = corrections[0]["CorrectedAnswer"]
            print(f"[DEBUG] Latest correction retrieved: {latest_correction}")
            return latest_correction

        print("[DEBUG] No correction found in Pinecone.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to query Pinecone for corrections: {e}")
        return None

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"[INFO] {client.user} is now running!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_message = message.content.strip()

    try:
        existing_answer = await find_correction_in_pinecone(user_message)
        if existing_answer:
            prompt = (
                f"Original Response: {existing_answer}\n"
                f"User Feedback: {user_message}\n\n"
                "If the user's message is feedback or correction, provide the updated response. "
                "If it is not, say 'No correction detected' and keep the original response."
            )
            chat_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that processes corrections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            corrected_data = chat_response['choices'][0]['message']['content'].strip()

            if "No correction detected" not in corrected_data:
                await upsert_to_pinecone(user_message, corrected_data)
                await message.channel.send(corrected_data)
            else:
                await message.channel.send(existing_answer)
            return

        embedding = await get_embedding(user_message)
        if embedding is None:
            await message.channel.send("An error occurred while processing your request.")
            return

        result = index.query(vector=embedding, top_k=3, include_metadata=True)
        response = generate_response(result, user_message)
        await message.channel.send(response)

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        await message.channel.send("An unexpected error occurred. Please try again later.")

def generate_response(result, query):
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('OriginalQuery', 'N/A')} Answer: {doc['metadata'].get('CorrectedAnswer', 'N/A')}"
        for doc in result["matches"]
    ])
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] Failed to generate response from OpenAI: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(upsert_dataset_to_pinecone('organized_faq_data.csv'))
        client.run(TOKEN)
    except Exception as e:
        print(f"[ERROR] Bot failed to start: {e}")
