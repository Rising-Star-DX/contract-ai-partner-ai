import os

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, AsyncAzureOpenAI, AzureOpenAI

from app.common.constants import EMBEDDING_MODEL, PROMPT_MODEL

load_dotenv() # 루트로 고정

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding_async_client() -> AsyncAzureOpenAI:
  httpx_client = httpx.AsyncClient()
  return AsyncAzureOpenAI(
      api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
      api_version="2023-05-15",
      azure_endpoint=os.getenv("AZURE_EMBEDDING_OPENAI_ENDPOINT"),
      http_client=httpx_client
  )


def get_embedding_sync_client() -> AzureOpenAI:
  return AzureOpenAI(
      api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
      api_version="2023-05-15",
      azure_endpoint=os.getenv("AZURE_EMBEDDING_OPENAI_ENDPOINT")
  )

embedding_deployment_name = EMBEDDING_MODEL


def get_prompt_async_client() -> AsyncAzureOpenAI:
  httpx_client = httpx.AsyncClient(
      limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
      timeout=httpx.Timeout(30.0),
      http2=False
  )
  return AsyncAzureOpenAI(
      api_key=os.getenv("AZURE_PROMPT_API_KEY"),
      api_version="2025-01-01-preview",
      azure_endpoint=os.getenv("AZURE_PROMPT_OPENAI_ENDPOINT"),
      http_client=httpx_client
  )

prompt_deployment_name = PROMPT_MODEL