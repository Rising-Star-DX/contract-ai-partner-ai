import uuid
from datetime import datetime
from typing import List

from app.blueprints.standard.standard_exception import StandardException
from app.clients.qdrant_client import qdrant_db_client
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.common.exception.error_code import ErrorCode
from app.models.vector import VectorPayload
from app.schemas.chunk_schema import ArticleChunk
from app.schemas.document_request import DocumentRequest
from app.containers.service_container import embedding_service, prompt_service


def vectorize_and_save(chunks: List[ArticleChunk], collection_name: str,
    pdf_request: DocumentRequest) -> None:
  points = []
  ensure_qdrant_collection(collection_name)

  for article in chunks:
    article_number = article.article_title

    if len(article.clauses) == 0:
      continue

    for clause in article.clauses:
      if len(clause.clause_content) <= 1:
        continue

      clause_content = clause.clause_content
      combined_text = f"조 {article_number}, 항 {clause.clause_number}: {clause_content}"

      # 1️⃣ Openai 벡터화
      clause_vector = embedding_service.embed_text(combined_text)

      # 2️⃣ Openai LLM 기반 교정 문구 생성
      result = prompt_service.make_correction_data(clause_content)

      # if result.startswith("{") and result.endswith("}"):
      #   json_result = json.loads(result)  # 파싱 성공 시
      # else:
      #   continue

      payload = VectorPayload(
          standard_id=pdf_request.id,
          category=pdf_request.categoryName,
          incorrect_text=result["incorrect_text"],
          proof_text=clause_content,
          corrected_text=result["corrected_text"],
          created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
  if len(points) == 0:
    raise StandardException(ErrorCode.CHUNKING_FAIL)
  qdrant_db_client.upsert(collection_name=collection_name, points=points)
