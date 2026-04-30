"""
Cliente OpenAI-compatível compartilhado entre os scripts.

Funciona com qualquer servidor que exponha a API da OpenAI:
- LM Studio:   http://localhost:1234/v1   (default)
- Ollama:      http://localhost:11434/v1
- vLLM:        http://localhost:8000/v1
- llama.cpp:   http://localhost:8080/v1

Configuração via .env (ou variáveis de ambiente):
    OPENAI_BASE_URL   — URL base do servidor
    OPENAI_API_KEY    — chave de API (qualquer string para servidores locais)
    EMBED_MODEL       — id do modelo de embedding padrão
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-bge-m3")


def make_client(base_url: str | None = None) -> OpenAI:
    url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    # LM Studio / Ollama ignoram a key, mas o SDK da OpenAI exige qualquer string
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    return OpenAI(base_url=url, api_key=api_key)


def embed(client: OpenAI, model: str, text: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding
