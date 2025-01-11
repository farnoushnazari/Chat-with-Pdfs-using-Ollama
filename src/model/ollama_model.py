import ollama
from langchain_ollama import ChatOllama
from src.utils.config import cfg

model = None

def get_model() -> ChatOllama:
    global model
    if model is None:
        ollama.pull(cfg['model_name'])
        model = ChatOllama(model=cfg['model_name'], temperature=0.1)
    return model