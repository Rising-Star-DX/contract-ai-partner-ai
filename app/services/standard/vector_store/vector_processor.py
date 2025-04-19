import asyncio
import logging
import time
import uuid
from typing import List

import numpy as np
from qdrant_client.models import PointStruct

from app.clients.openai_clients import get_prompt_async_client, \
  get_embedding_async_client
from app.clients.qdrant_client import get_qdrant_client
from app.containers.service_container import embedding_service
from app.schemas.chunk_schema import ClauseChunk
from app.schemas.document_request import DocumentRequest
from app.services.common.qdrant_utils import ensure_qdrant_collection, \
  upload_points_to_qdrant
from app.services.standard.vector_store.payload_builder import \
  make_word_payload, make_clause_payload


async def vectorize_and_save(chunks: List[ClauseChunk],
    pdf_request: DocumentRequest) -> None:
  qd_client = get_qdrant_client()
  await ensure_qdrant_collection(qd_client, pdf_request.categoryName)

  semaphore = asyncio.Semaphore(5)
  async with get_prompt_async_client() as prompt_client:
    task_fn = make_word_payload if pdf_request.categoryName == "법률용어" else make_clause_payload
    results = await asyncio.gather(*[
      task_fn(prompt_client, article, pdf_request, semaphore)
      for article in chunks
    ])

  embedding_inputs = [point.embedding_input() for point in results if point]

  async with get_embedding_async_client() as embedding_client:
    start_time = time.perf_counter()
    embeddings = await embedding_service.batch_embed_texts(embedding_client,
                                                           embedding_inputs)
    logging.info(f"임베딩 묶음 소요 시간: {time.perf_counter() - start_time:.4f}초")

  final_points = []
  for payload, vector in zip(results, embeddings):
    if vector and isinstance(vector, list) and not any(
        np.isnan(x) for x in vector):
      final_points.append(build_point(payload, vector))

  await upload_points_to_qdrant(qd_client, pdf_request.categoryName,
                                final_points)


def build_point(payload, embedding: List[float]) -> PointStruct:
  return PointStruct(
      id=str(uuid.uuid4()),
      vector=embedding,
      payload=payload.to_dict()
  )
