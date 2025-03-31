from typing import List, Tuple

import fitz

from app.blueprints.agreement.agreement_exception import AgreementException
from app.common.constants import CLAUSE_TEXT_SEPARATOR
from app.common.exception.custom_exception import BaseCustomException
from app.common.exception.error_code import ErrorCode
from app.common.file_type import FileType
from app.schemas.analysis_response import RagResult, ClauseData
from app.schemas.chunk_schema import ArticleChunk, DocumentChunk
from app.schemas.chunk_schema import Document
from app.schemas.document_request import DocumentRequest
from app.services.agreement.img_service import process_img
from app.services.common.chunking_service import \
  chunk_by_article_and_clause_with_page, chunk_by_article_and_clause
from app.services.common.pdf_service import convert_to_bytes_io, \
  extract_documents_from_pdf_io, byte_data
from app.services.common.s3_service import s3_get_object


def preprocess_data(document_request: DocumentRequest) -> Tuple[
  List[Document], fitz.Document]:

  documents: List[Document] = []
  byte_type_pdf = None

  if document_request.type in (FileType.PNG, FileType.JPG, FileType.JPEG):
    extracted_text = process_img(document_request)
  elif document_request.type == FileType.PDF:
    s3_stream = s3_get_object(document_request.url)
    pdf_bytes_io = convert_to_bytes_io(s3_stream)
    byte_type_pdf = byte_data(pdf_bytes_io)
    documents = extract_documents_from_pdf_io(pdf_bytes_io)
  else:
    raise BaseCustomException(ErrorCode.UNSUPPORTED_FILE_TYPE)

  if len(documents) == 0:
    raise BaseCustomException(ErrorCode.NO_TEXTS_EXTRACTED)
  return documents, byte_type_pdf


def chunk_standard_texts(extracted_text: str) -> List[ArticleChunk]:
  chunks = chunk_by_article_and_clause(extracted_text)
  if len(chunks) == 0:
    raise BaseCustomException(ErrorCode.CHUNKING_FAIL)
  return chunks


def chunk_agreement_documents(documents: List[Document]) -> List[DocumentChunk]:
  chunks = chunk_by_article_and_clause_with_page(documents)
  if len(chunks) == 0:
    raise BaseCustomException(ErrorCode.CHUNKING_FAIL)
  return chunks


def combine_chunks_by_clause_number(document_chunks: List[DocumentChunk]) -> \
List[RagResult]:
  combined_chunks: List[RagResult] = []
  clause_map: dict[str, RagResult] = {}

  for doc in document_chunks:
    rag_result = clause_map.setdefault(doc.clause_number, RagResult())

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