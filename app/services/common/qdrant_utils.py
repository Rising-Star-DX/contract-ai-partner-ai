from httpx import ConnectTimeout
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import VectorParams, Distance

from app.blueprints.standard.standard_exception import StandardException
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode


async def ensure_qdrant_collection(qd_client: AsyncQdrantClient,
    collection_name: str) -> None:
  try:
    exists = await qd_client.collection_exists(collection_name=collection_name)
    if not exists:
      await create_qdrant_collection(qd_client, collection_name)

  except (ConnectTimeout, ResponseHandlingException):
    raise CommonException(ErrorCode.QDRANT_NOT_STARTED)


async def create_qdrant_collection(qd_client: AsyncQdrantClient,
    collection_name: str):
  try:
    return await qd_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
  except (ConnectTimeout, ResponseHandlingException):
    raise CommonException(ErrorCode.QDRANT_CONNECTION_TIMEOUT)


async def upload_points_to_qdrant(qd_client: AsyncQdrantClient, collection_name,
    points):
  if not points:
    raise StandardException(ErrorCode.NO_POINTS_GENERATED)

  try:
    await qd_client.upsert(collection_name=collection_name, points=points)
  except (ConnectTimeout, ResponseHandlingException):
    raise CommonException(ErrorCode.QDRANT_CONNECTION_TIMEOUT)
