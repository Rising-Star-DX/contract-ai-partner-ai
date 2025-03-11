import os
import uuid
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.models.vector import VectorPayload
from app.schemas.chunk_schema import ArticleChunk
from app.schemas.pdf_request import PDFRequest
from app.containers.service_container import embedding_service, prompt_service

import time
# Qdrant 클라이언트 설정
qdrant_db_client = QdrantClient(url="http://localhost:6333")



def embed_chunks(chunks: List[ArticleChunk], collection_name: str,
    pdf_request: PDFRequest) -> None:
  points = []
  ensure_qdrant_collection(collection_name)

  for article in chunks:
    article_number = article.article_number

    for clause in article.clauses:
      clause_content = clause.clause_content
      clause_number = clause.clause_number

      combined_text = f"조 {article_number}, 항 {clause_number}: {clause_content}"
      clause_vector = vectorize_text(combined_text)

      payload = VectorPayload(
          standard_id=pdf_request.standardId,
          category=pdf_request.category,
          incorrect_text="",
          proof_text=clause_content,
          corrected_text=""
      )

      points.append(
          PointStruct(
              id=str(uuid.uuid4()),
              vector=clause_vector,
              payload=payload.to_dict()
          )
      )

  upload_points_to_qdrant(collection_name, points)


def ensure_qdrant_collection(collection_name: str) -> None:
  exists = qdrant_db_client.collection_exists(collection_name=collection_name)
  if not exists:
    create_qdrant_collection(collection_name)


def create_qdrant_collection(collection_name: str):
  return qdrant_db_client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
  )


def upload_points_to_qdrant(collection_name, points):
  qdrant_db_client.upsert(collection_name=collection_name, points=points)
