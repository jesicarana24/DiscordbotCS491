# Capstone FAQ Discord Bot

Capstone FAQ Bot is a tool designed to automate responses to frequently asked questions on a Discord server. Leveraging a Retrieval-Augmented Generation (RAG) model with Pinecone, this bot provides quick and accurate answers based on course and FAQ data, making it a valuable resource for students and TAs.

## Features

- **Automated FAQ Responses:** Uses a RAG model to generate accurate answers from a structured FAQ database.
- **Pinecone Integration:** Efficient data retrieval with Pinecone as the vector storage solution.
- **Modular Design:** Built to support future integrations with additional platforms.
- **Feedback Collection:** Collects user feedback on responses to improve the bot’s performance.

## Technologies Used

- **Bot Framework:** Python with Discord.py
- **Data Retrieval:** Pinecone and LangChain for RAG-based retrieval and response generation
- **Data Storage:** CSV files for FAQ data
- **Project Management:** JIRA and GitHub for version control and SCRUM-based workflow management

## Getting Started

To set up and run the Capstone FAQ Bot, follow these steps:

### Prerequisites

- Python 3.x
- Pinecone API Access
- Discord API Token

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jesicarana24/DiscordbotCS491.git
   cd DiscordbotCS491
   ```
2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ``` 
4. **Configure Environment Variables:**
   Create a .env file in the project root directory with the following variables:
    ``` bash
    DISCORD_TOKEN=your_discord_token
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=your_pinecone_environment
    ```
5. **Run the Bot:**
   ``` bash
   python bot.py
   ```

## Usage

1. Add the Capstone FAQ Bot to a Discord server.
2. In the designated channel, type a question or keyword related to the course content.
3. The bot will respond with relevant information pulled from the FAQ data stored in Pinecone.

License

This project is licensed under the MIT License - see the LICENSE file for details.

