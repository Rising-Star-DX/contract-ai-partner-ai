import asyncio
import logging
import time
from asyncio import Semaphore
from typing import List, Optional, Any

import fitz
from openai import AsyncAzureOpenAI
from qdrant_client import models, AsyncQdrantClient
from qdrant_client.http.models import QueryResponse

from app.blueprints.agreement.agreement_exception import AgreementException
from app.clients.openai_clients import get_prompt_async_client, \
  get_embedding_async_client
from app.clients.qdrant_client import get_qdrant_client
from app.common.chunk_status import ChunkProcessResult, ChunkProcessStatus
from app.common.constants import ARTICLE_CLAUSE_SEPARATOR, \
  CLAUSE_TEXT_SEPARATOR, MAX_RETRIES, LLM_TIMEOUT
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode
from app.containers.service_container import embedding_service, prompt_service
from app.schemas.analysis_response import RagResult, SearchResult
from app.schemas.document_request import DocumentRequest
from app.services.common.qdrant_utils import ensure_qdrant_collection

SEARCH_COUNT = 3
VIOLATION_THRESHOLD = 0.75
LLM_REQUIRED_KEYS = {"correctedText", "proofText", "violation_score"}


async def vectorize_and_calculate_similarity(
    combined_chunks: List[RagResult], document_request: DocumentRequest,
    byte_type_pdf: fitz.Document) -> List[RagResult]:
  qd_client = get_qdrant_client()
  await ensure_qdrant_collection(qd_client, document_request.categoryName)

  embedding_inputs = await prepare_embedding_inputs(combined_chunks)

  start_time = time.time()
  async with get_embedding_async_client() as embedding_client:
    embeddings = await embedding_service.batch_embed_texts(
        embedding_client, embedding_inputs)
  logging.info(f"임베딩 묶음 소요 시간: {time.time() - start_time}")

  semaphore = asyncio.Semaphore(5)
  async with get_prompt_async_client() as prompt_client:
    tasks = [
      process_clause(qd_client, prompt_client, chunk, embedding,
                     document_request.categoryName, semaphore, byte_type_pdf)
      for chunk, embedding in zip(combined_chunks, embeddings)
    ]
    results = await asyncio.gather(*tasks)

  success_results: List[RagResult] = [r.result for r in results if
                                      r.status == ChunkProcessStatus.SUCCESS and r.result is not None]
  failure_score = sum(r.status == ChunkProcessStatus.FAILURE for r in results)

  if not success_results and failure_score == len(combined_chunks):
    raise AgreementException(ErrorCode.CHUNK_ANALYSIS_FAILED)

  return success_results


async def prepare_embedding_inputs(chunks: List[RagResult]) -> List[str]:
  inputs = []
  for chunk in chunks:
    parts = chunk.incorrect_text.split(ARTICLE_CLAUSE_SEPARATOR, 1)
    title = parts[0].strip() if len(parts) == 2 else ""
    content = parts[1].strip() if len(parts) == 2 else parts[0].strip()
    inputs.append(f"{title} {content}")
  return inputs


async def process_clause(qd_client: AsyncQdrantClient,
    prompt_client: AsyncAzureOpenAI, rag_result: RagResult,
    embedding: List[float], collection_name: str, semaphore: Semaphore,
    byte_type_pdf: fitz.Document) -> ChunkProcessResult:

  search_results = await search_qdrant(semaphore, collection_name, embedding,
                                       qd_client)
  parse_incorrect_text(rag_result)
  corrected_result = await generate_clause_correction(prompt_client,
                                                      rag_result.incorrect_text,
                                                      search_results)
  if not corrected_result:
    return ChunkProcessResult(status=ChunkProcessStatus.FAILURE)

  try:
    score = float(corrected_result["violation_score"])
  except (KeyError, ValueError, TypeError):
    logging.warning(f"violation_score 추출 실패")
    return ChunkProcessResult(status=ChunkProcessStatus.FAILURE)

  if score < VIOLATION_THRESHOLD:
    return ChunkProcessResult(status=ChunkProcessStatus.SUCCESS)

  rag_result.accuracy = score
  all_positions = \
    await find_text_positions(rag_result.incorrect_text, byte_type_pdf)
  positions = await extract_positions_by_page(all_positions)

  set_result_data(corrected_result, rag_result, positions)
  if any(not clause.position for clause in rag_result.clause_data):
    logging.warning(f"원문 일치 position값 불러오지 못함")
    return ChunkProcessResult(status=ChunkProcessStatus.SUCCESS)

  return ChunkProcessResult(status=ChunkProcessStatus.SUCCESS,
                            result=rag_result)


def parse_incorrect_text(rag_result: RagResult) -> None:
  try:
    clause_parts = rag_result.incorrect_text.split(
        ARTICLE_CLAUSE_SEPARATOR, 1)
  except Exception:
    raise AgreementException(ErrorCode.NO_SEPARATOR_FOUND)

  rag_result.incorrect_text = clause_parts[-1]


async def search_qdrant(semaphore: Semaphore, collection_name: str,
    embedding: List[float],
    qd_client: AsyncQdrantClient) -> List[SearchResult]:
  search_results = await search_collection(qd_client, semaphore,
                                           collection_name, embedding)
  legal_term_results = await search_collection(qd_client, semaphore,
                                               "법률용어", embedding)
  return gather_search_results(search_results, legal_term_results)


async def search_collection(qd_client: AsyncQdrantClient,
    semaphore: Semaphore, collection_name: str,
    embedding: List[float]) -> QueryResponse:
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      async with semaphore:
        search_results = await qd_client.query_points(
            collection_name=collection_name,
            query=embedding,
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=SEARCH_COUNT,
            with_payload=True
        )
      break
    except Exception as e:
      if attempt == MAX_RETRIES:
        raise CommonException(ErrorCode.QDRANT_SEARCH_FAILED)
      logging.warning(
          f"[search_collection]: Qdrant Search 재요청 발생 {attempt}/{MAX_RETRIES} {e}")
      await asyncio.sleep(1)

  if search_results is None or not search_results.points:
    raise AgreementException(ErrorCode.NO_POINTS_FOUND)

  return search_results


def gather_search_results(search_results: QueryResponse,
    legal_term_results: QueryResponse) -> List[
  SearchResult]:
  merged_results: List[SearchResult] = []

  for i in range(SEARCH_COUNT):
    clause_payload = search_results.points[i].payload if i < len(
      search_results.points) else {}
    term_payload = legal_term_results.points[i].payload if i < len(
      legal_term_results.points) else {}

    merged_results.append(
        SearchResult(
            proof_text=clause_payload.get("proof_text", ""),
            incorrect_text=clause_payload.get("incorrect_text", ""),
            corrected_text=clause_payload.get("corrected_text", ""),
            term=term_payload.get("term", ""),
            meaning_difference=term_payload.get("meaning_difference", ""),
            definition=term_payload.get("definition", ""),
            keywords=term_payload.get("keywords", "")
        )
    )

  return merged_results


async def generate_clause_correction(prompt_client: AsyncAzureOpenAI,
    article_content: str, search_results: List[SearchResult]) -> Optional[
  dict[str, Any]]:
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      result = await asyncio.wait_for(
          prompt_service.correct_contract(
              prompt_client,
              search_results,
              clause_content=article_content.replace("\n", " ")
          ),
          timeout=LLM_TIMEOUT
      )
      if is_correct_response_format(result):
        return result
      logging.warning(
          f"[generate_clause_correction]: llm 응답 필수 키 누락 / str형 반환 안됨")

    except Exception as e:
      if isinstance(e, asyncio.TimeoutError):
        if attempt == MAX_RETRIES:
          raise CommonException(ErrorCode.LLM_RESPONSE_TIMEOUT)
      else:
        if attempt == MAX_RETRIES:
          raise AgreementException(ErrorCode.AGREEMENT_REVIEW_FAIL)
        logging.warning(
            f"query_points: 계약서 LLM 재요청 발생 {attempt}/{MAX_RETRIES} {e}")
        await asyncio.sleep(0.5 * attempt)

  return None


def is_correct_response_format(result: dict[str, str]) -> bool:
  return (
      isinstance(result, dict)
      and LLM_REQUIRED_KEYS.issubset(result.keys())
      and all(isinstance(result[key], str) for key in LLM_REQUIRED_KEYS)
  )


def set_result_data(corrected_result: Optional[
  dict[str, Any]], rag_result: RagResult, positions: List[List]):
  rag_result.incorrect_text = (
    rag_result.incorrect_text.split(ARTICLE_CLAUSE_SEPARATOR, 1)[-1]
    .replace(CLAUSE_TEXT_SEPARATOR, "")
    .replace("\n", "")
    .replace("", '"')
  )

  value = corrected_result["correctedText"]
  rag_result.corrected_text = " ".join(value) if isinstance(value,
                                                            list) else value
  value = corrected_result["proofText"]
  rag_result.proof_text = " ".join(value) if isinstance(value, list) else value

  rag_result.clause_data[0].position = positions[0]
  if positions[1]:
    rag_result.clause_data[1].position = positions[1]


async def find_text_positions(clause_content: str,
    pdf_document: fitz.Document) -> dict[int, List[dict]]:
  all_positions = {}  # 페이지별로 위치 정보를 저장할 딕셔너리

  # "!!!"을 기준으로 더 나눠서 각각을 위치 찾기
  clause_parts = clause_content.split('!!!')

  # 모든 페이지를 검색
  for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)  # 페이지 로드

    # 페이지 크기 얻기 (페이지의 너비와 높이)
    page_width = float(page.rect.width)  # 명시적으로 float로 처리
    page_height = float(page.rect.height)  # 명시적으로 float로 처리

    page_positions = []  # 현재 페이지에 대한 위치 정보를 담을 리스트

    # 각 문장에 대해 위치를 찾기
    for part in clause_parts:
      part = part.strip()  # 앞뒤 공백 제거
      if part == "":
        continue

      # 문장의 위치를 찾기 위해 search_for 사용
      text_instances = page.search_for(part)

      # y값을 기준으로 묶을 변수
      grouped_positions = {}

      # 텍스트 인스턴스들에 대해 위치 정보를 추출
      for text_instance in text_instances:
        x0, y0, x1, y1 = text_instance  # 바운딩 박스 좌표

        # 상대적인 위치로 계산 (픽셀을 페이지 크기로 나누어 상대값 계산)
        rel_x0 = x0 / page_width
        rel_y0 = y0 / page_height
        rel_x1 = x1 / page_width
        rel_y1 = y1 / page_height

        # 바운딩 박스 높이 계산 (상대값 기준)
        box_height = rel_y1 - rel_y0
        epsilon = box_height / 4

        # y 값을 기준으로 그룹화
        group_found = False
        for y_key in grouped_positions:
          if abs(rel_y0 - y_key) <= epsilon:
            grouped_positions[y_key].append((rel_x0, rel_x1, rel_y0, rel_y1))
            group_found = True
            break

        if not group_found:
          grouped_positions[rel_y0] = [(rel_x0, rel_x1, rel_y0, rel_y1)]

      # 그룹화된 바운딩 박스를 하나의 큰 박스로 묶기
      for y_key, group in grouped_positions.items():
        # 하나의 그룹에서 x0, x1의 최솟값과 최댓값을 구하기
        min_x0 = min([x[0] for x in group])  # 최소 x0 값
        max_x1 = max([x[1] for x in group])  # 최대 x1 값

        # 하나의 그룹에서 y0, y1의 최솟값과 최댓값을 구하기
        min_y0 = min([x[2] for x in group])  # 최소 y0 값
        max_y1 = max([x[3] for x in group])  # 최대 y1 값

        # 상대적인 값에 100을 곱해줍니다
        min_x0 *= 100
        min_y0 *= 100
        max_x1 *= 100
        max_y1 *= 100

        width = max_x1 - min_x0
        height = max_y1 - min_y0

        # 바운딩 박스를 생성 (최소값과 최대값을 사용)
        page_positions.append({
          "page": page_num + 1,
          "bbox": (min_x0, min_y0, width, height)  # 상대적 x, y, 너비, 높이
        })

    # 페이지별로 위치를 딕셔너리에 추가
    if page_positions:
      all_positions[page_num + 1] = page_positions

  return all_positions


async def extract_positions_by_page(all_positions: dict[int, List[dict]]) -> \
    List[List]:
  positions = [[], []]
  first_page = None

  # `rag_result.clause_data`에 두 개만 저장
  for page_num, positions_in_page in all_positions.items():
    # 첫 번째 문장이 시작되는 페이지를 찾으면 첫 번째 위치에 저장
    if first_page is None:
      first_page = page_num
      positions[0].extend(p['bbox'] for p in positions_in_page)
    else:
      # 첫 번째 문장이 시작된 후, 페이지가 변경되면 두 번째 위치에 저장
      if page_num != first_page:
        positions[1].extend(p['bbox'] for p in positions_in_page)

  return positions
