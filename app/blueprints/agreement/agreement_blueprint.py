import asyncio
import logging
from http import HTTPStatus
from typing import List

from flask import Blueprint, request
from pydantic import ValidationError

from app.blueprints.agreement.agreement_exception import AgreementException
from app.common.constants import Constants
from app.common.exception.custom_exception import BaseCustomException
from app.common.exception.error_code import ErrorCode
from app.common.file_type import FileType
from app.schemas.analysis_response import AnalysisResponse
from app.schemas.document import Document
from app.schemas.document_request import DocumentRequest
from app.schemas.success_code import SuccessCode
from app.schemas.success_response import SuccessResponse
from app.services.agreement.img_service import process_img
from app.services.agreement.vectorize_similarity import \
  vectorize_and_calculate_similarity
from app.services.common.ingestion_pipeline import preprocess_data, chunk_agreement_documents
import time

agreements = Blueprint('agreements', __name__, url_prefix="/flask/agreements")


@agreements.route('/analysis', methods=['POST'])
def process_agreements_pdf_from_s3():
  try:
    json_data = request.get_json()
    if json_data is None:
      raise BaseCustomException(ErrorCode.INVALID_JSON_FORMAT)

    document_request = DocumentRequest(**json_data)
  except ValidationError:
    raise BaseCustomException(ErrorCode.FIELD_MISSING)

  documents: List[Document] = []
  if document_request.type in (FileType.PNG, FileType.JPG, FileType.JPEG):
    extracted_text = process_img(document_request)
  elif document_request.type == FileType.PDF:
    documents = preprocess_data(document_request)
  else:
    raise AgreementException(ErrorCode.UNSUPPORTED_FILE_TYPE)

  if len(documents) == 0:
    raise AgreementException(ErrorCode.NO_TEXTS_EXTRACTED)

  document_chunks = chunk_agreement_documents(documents)

  # 5️⃣ 벡터화 + 유사도 비교 (리턴값 추가)
  start_time = time.time()
  result = asyncio.run(vectorize_and_calculate_similarity(extracted_text, chunks, document_request))
  end_time = time.time()
  logging.info(f"Time vectorize and prompt texts: {end_time - start_time:.4f} seconds")

  response = AnalysisResponse(
      total_page = len(documents),
      summary_content="",
      chunks=[]
  )

  return SuccessResponse(SuccessCode.REVIEW_SUCCESS, response).of(), HTTPStatus.OK
