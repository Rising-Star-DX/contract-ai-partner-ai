import asyncio
import logging
from asyncio import Semaphore
from typing import List, Optional, Any

import fitz
from qdrant_client import models, AsyncQdrantClient
from qdrant_client.http.models import QueryResponse

from app.blueprints.agreement.agreement_exception import AgreementException
from app.clients.qdrant_client import get_qdrant_client
from app.common.constants import ARTICLE_CLAUSE_SEPARATOR, \
  CLAUSE_TEXT_SEPARATOR, MAX_RETRIES
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode
from app.containers.service_container import embedding_service, prompt_service
from app.schemas.analysis_response import RagResult, SearchResult
from app.schemas.document_request import DocumentRequest
from app.services.standard.vector_store import ensure_qdrant_collection

SEARCH_COUNT = 3
LLM_REQUIRED_KEYS = {"clause_content", "correctedText", "proofText",
                     "violation_score"}


async def vectorize_and_calculate_similarity(
    combined_chunks: List[RagResult], document_request: DocumentRequest,
    byte_type_pdf: fitz.Document) -> List[RagResult]:
  qd_client = get_qdrant_client()
  await ensure_qdrant_collection(qd_client, document_request.categoryName)

  semaphore = asyncio.Semaphore(5)
  tasks = []
  for chunk in combined_chunks:
    tasks.append(process_clause(qd_client, chunk, chunk.incorrect_text,
                                document_request.categoryName, semaphore,
                                byte_type_pdf))
  # 모든 임베딩 및 유사도 검색 태스크를 병렬로 실행
  results = await asyncio.gather(*tasks)
  return [result for result in results if result is not None]


async def process_clause(qd_client: AsyncQdrantClient, rag_result: RagResult,
    clause_content: str, collection_name: str, semaphore,
    byte_type_pdf: fitz.Document) -> Optional[RagResult]:
  parts = clause_content.split(ARTICLE_CLAUSE_SEPARATOR, 1)

  article_title = ""
  if len(parts) == 2:
    article_title = parts[0].strip()
    article_content = parts[1].strip()
  else:
    article_content = parts[0].strip()

  embedding = \
    await embedding_service.embed_text(article_title + " " + article_content)
  search_results = \
    await search_qdrant(semaphore, collection_name, embedding, qd_client)
  clause_results = await gather_search_results(search_results)
  corrected_result = \
    await generate_clause_correction(article_content, clause_results)

  if not corrected_result:
    return None

  try:
    score = float(corrected_result["violation_score"])
  except (KeyError, ValueError, TypeError):
    logging.warning(f"violation_score 추출 실패")
    return None

  if score >= 0.90:
    # 원문 텍스트에 대한 위치 정보 찾기
    all_positions = await find_text_positions(clause_content, byte_type_pdf)

    # 페이지를 기준으로 position을 나누어 저장할 리스트
    positions = [[], []]

    first_page = None  # 첫 번째 문장이 시작되는 페이지를 추적

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

    # `rag_result.clause_data`에 위치 정보 저장
    rag_result.accuracy = float(corrected_result["violation_score"])
    rag_result.incorrect_text = (
      article_content
      .replace(CLAUSE_TEXT_SEPARATOR, "")
      .replace("\n", "")
      .replace("", '"')
    )
    rag_result.corrected_text = corrected_result["correctedText"]
    rag_result.proof_text = corrected_result["proofText"]

    rag_result.clause_data[0].position = positions[0]

    # 문장이 다음페이지로 넘어가는 경우에만 [1] 에 저장
    if positions[1]:
      rag_result.clause_data[1].position = positions[1]

    return rag_result

  else:
    # accuracy가 0.5 이하일 경우 빈 객체 반환
    return None


async def search_qdrant(semaphore: Semaphore, collection_name: str,
    embedding: List[float],
    qd_client: AsyncQdrantClient) -> QueryResponse:
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      async with semaphore:
        search_results = await qd_client.query_points(
            collection_name=collection_name,
            query=embedding,
            # query_filter=models.Filter(
            #     must=[models.FieldCondition(
            #         key="category",
            #         match=models.MatchValue(value=category_name)
            #     )]
            # ),
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=SEARCH_COUNT
        )
      break
    except Exception as e:
      if attempt == MAX_RETRIES:
        raise CommonException(ErrorCode.QDRANT_SEARCH_FAILED)
      logging.warning(
          f"query_points: Qdrant Search 재요청 발생 {attempt}/{MAX_RETRIES} {e}")
      await asyncio.sleep(1)

  if search_results is None:
    raise AgreementException(ErrorCode.NO_POINTS_FOUND)

  return search_results


async def gather_search_results(search_results: QueryResponse) -> List[
  SearchResult]:
  clause_results = []
  for match in search_results.points:
    payload_data = match.payload or {}

    if not isinstance(payload_data, dict):
      logging.warning(f"search_results dict 타입 반환 안됨: {type(payload_data)}")
      continue

    clause_results.append(
        SearchResult(
            proof_text=payload_data.get("proof_text", ""),
            incorrect_text=payload_data.get("incorrect_text", ""),
            corrected_text=payload_data.get("corrected_text", "")
        ))

  return clause_results


async def generate_clause_correction(article_content: str,
    clause_results: List[SearchResult]) -> Optional[dict[str, Any]]:
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      result = await prompt_service.correct_contract(
          clause_content=article_content,
          proof_text=[item.proof_text for item in clause_results],  # 기준 문서들
          incorrect_text=[item.incorrect_text for item in clause_results],
          corrected_text=[item.corrected_text for item in clause_results]
      )
      if isinstance(result, dict) and LLM_REQUIRED_KEYS.issubset(result.keys()):
        return result
      logging.warning(f"[generate_clause_correction]: llm 응답 필수 키 누락됨")

    except Exception as e:
      if attempt == MAX_RETRIES:
        raise AgreementException(ErrorCode.REVIEW_FAIL)
      logging.warning(
          f"query_points: 계약서 LLM 재요청 발생 {attempt}/{MAX_RETRIES} {e}")
      await asyncio.sleep(0.5 * attempt)

  return None


async def find_text_positions(clause_content: str,
    pdf_document: fitz.Document) -> dict[int, List[dict]]:
  all_positions = {}  # 페이지별로 위치 정보를 저장할 딕셔너리

  # +를 기준으로 문장을 나누고 뒤에 있는 부분만 사용
  clause_content_parts = clause_content.split('+', 1)
  if len(clause_content_parts) > 1:
    clause_content = clause_content_parts[1].strip()  # `+` 뒤의 내용만 사용

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

        # y 값을 기준으로 그룹화
        if rel_y0 not in grouped_positions:
          grouped_positions[rel_y0] = []

        grouped_positions[rel_y0].append((rel_x0, rel_x1, rel_y0, rel_y1))

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
