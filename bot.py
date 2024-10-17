import discord
import os
from dotenv import load_dotenv
from main import get_embedding, query_pinecone, generate_response  # Import functions from main.py

# Load environment variables
load_dotenv()

# Initialize Discord client
TOKEN = os.getenv('TOKEN')  # The bot token from your .env file

# Set up intents
intents = discord.Intents.default()
intents.message_content = True  # Make sure this is enabled in the developer portal

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} is now running!')

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    user_message = str(message.content)

    try:
        # Step 1: Get the embedding for the query from Discord message
        query_embedding = get_embedding(user_message)
        
        # Step 2: Query Pinecone using the query embedding
        retrieved_docs = query_pinecone(query_embedding)
        
        # Step 3: Generate a response based on the retrieved documents
        response = generate_response(retrieved_docs, user_message)
        
        # Send the response back to the user in Discord
        await message.channel.send(response)
        
    except Exception as e:
        await message.channel.send(f"An error occurred: {str(e)}")

# Run the bot
client.run(TOKEN)
