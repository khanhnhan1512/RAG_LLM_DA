import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def connect_llm_chat(settings):
    """
    Connect to the OpenAI Chat API.
    """
    chat_model = None
    try:
        chat_model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=settings["model"],
            response_format={"type": "json_object"},
            max_retries=3,
            timeout=40,
            max_tokens=4096
        )
    except Exception as e:
        print(f"Error connecting to OpenAI Chat API: {e}")
        return None
    return chat_model

def connect_llm_embeddings(settings):
    """
    Connect to the OpenAI Embeddings API.
    """
    embed_model = None
    try:
        embed_model = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=settings["model"],
            chunk_size=1
        )
    except Exception as e:
        print(f"Error connecting to OpenAI Embeddings API: {e}")
        return None
    return embed_model