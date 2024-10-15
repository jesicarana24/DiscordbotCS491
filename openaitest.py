import openai
import os
from dotenv import load_dotenv

# Load the environment variables containing your API key
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Check your API key by making a simple request using the new interface
try:
    # Make a simple chat completion request using the new interface
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" if available in your account
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "This is a test to verify my OpenAI API key."}
        ]
    )

    # If the request is successful, print the response
    print("OpenAI API connection is successful.")
    print("Response: ", response['choices'][0]['message']['content'])

except openai.error.AuthenticationError:
    print("Failed to authenticate. Please check your API key.")
except openai.error.OpenAIError as e:
    print(f"An OpenAI API error occurred: {e}")
