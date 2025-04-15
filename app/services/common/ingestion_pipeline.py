import re
from typing import List, Tuple

import fitz

from app.common.constants import CLAUSE_TEXT_SEPARATOR
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode
from app.common.file_type import FileType
from app.schemas.analysis_response import RagResult, ClauseData, OCRRagResult, OCRClauseData
from app.schemas.chunk_schema import DocumentChunk, OCRDocumentChunk
from app.schemas.chunk_schema import Document, OCRDocument
from app.schemas.document_request import DocumentRequest
from app.services.agreement.ocr_service import \
  chunk_by_article_and_clause_without_page_ocr
from app.services.common.chunking_service import \
  chunk_by_article_and_clause_with_page, semantic_chunk
from app.services.common.pdf_service import convert_to_bytes_io, \
  parse_pdf_to_documents, extract_fitz_document_from_pdf_io, \
  parse_pdf_to_documents_ocr
from app.services.common.s3_service import s3_get_object, \
  generate_pre_signed_url
from app.services.agreement.vision_service import extract_ocr

def preprocess_data(document_request: DocumentRequest) -> Tuple[
  List[Document], fitz.Document]:

  documents: List[Document] = []
  fitz_document = None

  file_type = extract_file_type(document_request.url)
  if file_type in (FileType.PNG, FileType.JPG, FileType.JPEG):
    pre_signed_url = generate_pre_signed_url(document_request.url)
  elif file_type == FileType.PDF:
    s3_stream = s3_get_object(document_request.url)
    pdf_bytes_io = convert_to_bytes_io(s3_stream)
    fitz_document = extract_fitz_document_from_pdf_io(pdf_bytes_io)
    documents = parse_pdf_to_documents(fitz_document)
  else:
    raise CommonException(ErrorCode.UNSUPPORTED_FILE_TYPE)

  if len(documents) == 0:
    raise CommonException(ErrorCode.NO_TEXTS_EXTRACTED)
  return documents, fitz_document


def preprocess_data_ocr(document_request: DocumentRequest) -> Tuple[
  List[OCRDocument], List[dict]]:

  documents: List[OCRDocument] = []

  height = 0
  width = 0
  all_texts_with_bounding_boxes = None

  file_type = extract_file_type(document_request.url)
  if file_type in (FileType.PNG, FileType.JPG, FileType.JPEG):
    full_text, all_texts_with_bounding_boxes = extract_ocr(document_request.url)

    documents.append(OCRDocument(content=full_text))

  elif file_type == FileType.PDF:
    s3_stream = s3_get_object(document_request.url)
    pdf_bytes_io = convert_to_bytes_io(s3_stream)
    fitz_document = extract_fitz_document_from_pdf_io(pdf_bytes_io)
    documents = parse_pdf_to_documents_ocr(fitz_document)
  else:
    raise CommonException(ErrorCode.UNSUPPORTED_FILE_TYPE)

  if len(documents) == 0:
    raise CommonException(ErrorCode.NO_TEXTS_EXTRACTED)
  return documents, all_texts_with_bounding_boxes



def extract_file_type(url: str) -> FileType:
  try:
    ext = url.split(".")[-1].strip().upper()
    return FileType(ext)
  except Exception:
    raise CommonException(ErrorCode.UNSUPPORTED_FILE_TYPE)


def chunk_standard_texts(extracted_text: str) -> List[str]:
  chunks =  semantic_chunk(
      extracted_text,
      similarity_threshold=0.6
  )
  if len(chunks) == 0:
    raise CommonException(ErrorCode.CHUNKING_FAIL)
  return chunks


def chunk_agreement_documents(documents: List[Document]) -> List[DocumentChunk]:
  chunks = chunk_by_article_and_clause_with_page(documents)
  keep_text, _ = chunks[0].clause_content.split("1.", 1)
  chunks[0].clause_content = keep_text
  keep_text, _ = chunks[-2].clause_content.split("날짜 :", 1)
  chunks[-2].clause_content = keep_text.strip()
  del chunks[-1]

  if len(chunks) == 0:
    raise CommonException(ErrorCode.CHUNKING_FAIL)
  return chunks


def chunk_agreement_documents_ocr(documents: List[OCRDocument]) -> List[OCRDocumentChunk]:
  chunks = chunk_by_article_and_clause_without_page_ocr(documents)
  # keep_text, _ = chunks[0].clause_content.split("1.", 1)
  # chunks[0].clause_content = keep_text
  # keep_text, _ = chunks[-2].clause_content.split("날짜 :", 1)
  # chunks[-2].clause_content = keep_text.strip()
  # del chunks[-1]

  if len(chunks) == 0:
    raise CommonException(ErrorCode.CHUNKING_FAIL)
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


def combine_chunks_by_clause_number_ocr(document_chunks: List[OCRDocumentChunk]) -> \
List[OCRRagResult]:
  combined_chunks: List[OCRRagResult] = []

  for doc in document_chunks:
    combined_chunks.append(
      OCRRagResult(
          incorrect_text=doc.clause_content,
          clause_data = [OCRClauseData()]
      )
    )

  return combined_chunks


def normalize_spacing(text: str) -> str:
    text = text.replace('\n', '[[[NEWLINE]]]')
    text = re.sub(r'\s{5,}', '\n', text)
    text = text.replace('[[[NEWLINE]]]', '\n')
    return text
