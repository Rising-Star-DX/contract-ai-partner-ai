import asyncio
import re
from typing import List, Tuple

from app.common.constants import CLAUSE_TEXT_SEPARATOR, ARTICLE_CHUNK_PATTERN, \
  NUMBER_HEADER_PATTERN
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode
from app.common.file_type import FileType
from app.schemas.analysis_response import RagResult, ClauseData
from app.schemas.chunk_schema import ClauseChunk
from app.schemas.chunk_schema import Document
from app.schemas.chunk_schema import DocumentChunk, DocumentMetadata
from app.schemas.document_request import DocumentRequest
from app.services.agreement.ocr_service import extract_ocr, \
  vectorize_and_calculate_similarity_ocr
from app.services.agreement.vectorize_similarity import \
  vectorize_and_calculate_similarity
from app.services.common.chunking_service import \
  chunk_by_article_and_clause_with_page, semantic_chunk_with_overlap, \
  chunk_by_paragraph
from app.services.common.pdf_service import preprocess_pdf


def ocr_service(document_request: DocumentRequest):
  full_text, all_texts_with_bounding_boxes = extract_ocr(document_request.url)

  documents: List[Document] = [
    Document(page_content=full_text, metadata=DocumentMetadata(page=1))]

  document_chunks = chunk_agreement_documents(documents)
  combined_chunks = combine_chunks_by_clause_number(document_chunks)

  # 입력값이 다르기에 함수가 분리되어야 함
  chunks = asyncio.run(
      vectorize_and_calculate_similarity_ocr(combined_chunks, document_request,
                                             all_texts_with_bounding_boxes))

  return chunks, len(combined_chunks), len(documents)


def pdf_agreement_service(document_request: DocumentRequest) -> Tuple[
  List[RagResult], int, int]:
  documents, fitz_document = preprocess_pdf(document_request)
  document_chunks = chunk_agreement_documents(documents)
  combined_chunks = combine_chunks_by_clause_number(document_chunks)
  chunks = asyncio.run(
      vectorize_and_calculate_similarity(combined_chunks, document_request,
                                         fitz_document))

  return chunks, len(combined_chunks), len(documents)


def extract_file_type(url: str) -> FileType:
  try:
    ext = url.split(".")[-1].strip().upper()
    return FileType(ext)
  except Exception:
    raise CommonException(ErrorCode.UNSUPPORTED_FILE_TYPE)


def chunk_standard_texts(documents: List[Document], category: str,
    page_batch_size: int = 50) -> List[ClauseChunk]:
  all_clauses = []

  for start in range(0, len(documents), page_batch_size):
    batch_docs = documents[start:start + page_batch_size]
    extracted_text = "\n".join(doc.page_content for doc in batch_docs)

    article_chunks = semantic_chunk_with_overlap(extracted_text, similarity_threshold=0.3)
    all_clauses.extend(article_chunks)

  return all_clauses


def chunk_agreement_documents(documents: List[Document]) -> List[DocumentChunk]:
  if re.findall(ARTICLE_CHUNK_PATTERN, documents[0].page_content,
                flags=re.DOTALL):
    chunks = chunk_by_article_and_clause_with_page(documents,
                                                   ARTICLE_CHUNK_PATTERN)
  elif re.findall(NUMBER_HEADER_PATTERN, documents[0].page_content,
                  flags=re.DOTALL):
    chunks = chunk_by_article_and_clause_with_page(documents,
                                                   NUMBER_HEADER_PATTERN)
  else:
    chunks = chunk_by_paragraph(documents)

  # keep_text, _ = chunks[0].clause_content.split("1.", 1)
  # chunks[0].clause_content = keep_text
  # keep_text, _ = chunks[-2].clause_content.split("날짜 :", 1)
  # chunks[-2].clause_content = keep_text.strip()
  # del chunks[-1]

  if not chunks:
    raise CommonException(ErrorCode.CHUNKING_FAIL)
  return chunks


def combine_chunks_by_clause_number(document_chunks: List[DocumentChunk]) -> \
    List[RagResult]:
  combined_chunks: List[RagResult] = []
  clause_map: dict[str, RagResult] = {}

  for doc in document_chunks:
    rag_result = clause_map.setdefault(doc.clause_number, RagResult())

    if not doc.clause_content.strip():
      continue

    if rag_result.incorrect_text:
      rag_result.incorrect_text += (
          CLAUSE_TEXT_SEPARATOR + doc.clause_content)
    else:
      rag_result.incorrect_text = doc.clause_content
      combined_chunks.append(rag_result)

    rag_result.clause_data.append(ClauseData(
        order_index=doc.order_index,
        page=doc.page
    ))

  return combined_chunks


def normalize_spacing(text: str) -> str:
  text = text.replace('\n', '[[[NEWLINE]]]')
  text = re.sub(r'\s{5,}', '\n', text)
  text = text.replace('[[[NEWLINE]]]', '\n')
  return text
