import requests
import os
import json
import streamlit as st
import openai as client
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from pydantic import BaseModel, Field

# Functions for each tool
def search_wikipedia(inputs):
    query = inputs["query"]
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(query)

def search_duckduckgo(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)

def scrape_website(inputs):
    url = inputs["url"]
    response = requests.get(url)
    return response.text

def save_text(inputs):
    text = inputs["text"]
    filename = inputs["filename"]
    with open(filename, 'w') as file:
        file.write(text)
    return f"Text saved to {filename}"

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Searches Wikipedia for a given query and returns the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search on Wikipedia.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "Searches DuckDuckGo for a given query and returns the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search on DuckDuckGo.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrapes content from the specified URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape content from.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_text",
            "description": "Saves the provided text to a file at the specified filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to save.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename to save text to.",
                    }
                },
                "required": ["text", "filename"],
            },
        },
    },
]

assistant = client.beta.assistants.create(
    name="Invest Assistant",
    instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
    model="gpt-4-turbo",
    tools=functions,
)

thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Search Wikipedia for information about the Eiffel Tower",
        }
    ]
)

st.title('A.Investment')

api_key = st.sidebar.text_input("Enter your OpenAI API Key")
st.sidebar.markdown("Find the code at [GitHub Repository](https://github.com/Juun0830/GPTs/tree/main/fullstackgpt/challenge)")

if api_key:
    client.api_key = api_key
    st.sidebar.success("API Key loaded successfully!")

    def create_thread(initial_message):
        thread = client.beta.threads.runs.create(
            messages=[{"role": "user", "content": initial_message}]
        )
        return thread.id

    def send_message(thread_id, content):
        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=content)

    def get_messages(thread_id):
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return messages

    thread_id = st.text_input("Enter Thread ID:")
    if st.button("Create New Thread"):
        initial_message = st.text_input("Enter Initial Message:")
        if initial_message:
            thread_id = create_thread(initial_message)
            st.write(f"Thread created! Thread ID: {thread_id}")

    if thread_id:
        if st.button("Refresh Conversation"):
            messages = get_messages(thread_id)
            for message in messages:
                st.write(f"{message['role']}: {message['content']}")

