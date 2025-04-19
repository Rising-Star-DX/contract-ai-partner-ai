import asyncio
import logging
import time
from http import HTTPStatus

from flask import Blueprint, request
from pydantic import ValidationError

from app.blueprints.agreement.agreement_exception import AgreementException
from app.common.constants import SUCCESS
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode
from app.schemas.analysis_response import StandardResponse
from app.schemas.document_request import DocumentRequest
from app.schemas.success_code import SuccessCode
from app.schemas.success_response import SuccessResponse
from app.services.common.ingestion_pipeline import preprocess_data, \
  chunk_standard_texts, normalize_spacing
from app.services.standard.vector_delete import delete_by_standard_id
from app.services.standard.vector_store.vector_processor import \
  vectorize_and_save

standards = Blueprint('standards', __name__, url_prefix="/flask/standards")


@standards.route('/analysis', methods=['POST'])
def process_standards_pdf_from_s3():
  try:
    json_data = request.get_json()
    if json_data is None:
      raise CommonException(ErrorCode.INVALID_JSON_FORMAT)

    document_request = DocumentRequest(**json_data)
  except ValidationError:
    raise CommonException(ErrorCode.FIELD_MISSING)

  documents, _ = preprocess_data(document_request)
  chunks = chunk_standard_texts(documents, document_request.categoryName)

  # 5️⃣ 벡터화 + Qdrant 저장
  start_time = time.time()
  asyncio.run(vectorize_and_save(chunks, document_request))
  end_time = time.time()
  logging.info(f"vectorize_and_save 소요 시간: {end_time - start_time}")

  contents = [normalize_spacing(doc.page_content) for doc in documents]
  return SuccessResponse(SuccessCode.ANALYSIS_COMPLETE,
                         StandardResponse(result=SUCCESS, contents=contents)).of(), HTTPStatus.OK


@standards.route('<categoryName>/<standardId>', methods=["DELETE"])
def delete_standard(categoryName: str, standardId: str):
  if not categoryName.strip() or not standardId.strip():
    raise AgreementException(ErrorCode.INVALID_URL_PARAMETER)

  if not standardId.isdigit():
    raise AgreementException(ErrorCode.CANNOT_CONVERT_TO_NUM)

  success_code = (
    asyncio.run(delete_by_standard_id(int(standardId), categoryName)))
  return SuccessResponse(success_code, SUCCESS).of(), HTTPStatus.OK
