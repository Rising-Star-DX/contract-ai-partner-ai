from typing import List

from app.schemas.chunk_schema import ArticleChunk
from app.schemas.document_request import DocumentRequest
from app.services.common.chunking_service import chunk_by_article_and_clause
from app.services.common.pdf_service import convert_to_bytes_io, \
  extract_text_from_pdf_io
from app.services.common.s3_service import s3_get_object


def preprocess_data(document_request: DocumentRequest) -> List[ArticleChunk]:
  # 1️⃣ s3에서 문서(pdf) 가져오기 (메모리 내)
  s3_stream = s3_get_object(document_request.url)

  # 2️⃣ PDF 스트림으로 변환
  pdf_bytes_io = convert_to_bytes_io(s3_stream)

  # 3️⃣ PDF에서 텍스트 추출
  extracted_text = extract_text_from_pdf_io(pdf_bytes_io)

  # 4️⃣ 텍스트 청킹
  chunks = chunk_by_article_and_clause(extracted_text)

  return chunks
