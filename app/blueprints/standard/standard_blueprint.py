from flask import Blueprint, request

from app.blueprints.standard.standard_exception import StandardException
from app.common.exception.custom_exception import BaseCustomException
from app.schemas.pdf_request import PDFRequest
from app.schemas.success_code import SuccessCode
from app.schemas.success_response import SuccessResponse
from app.services.standard import processor
from pydantic import ValidationError

standard = Blueprint('standard', __name__, url_prefix="/flask/standard")

@standard.route('', methods=['POST'])
def process_pdf_from_s3():

  try:
    pdf_request = PDFRequest(**request.get_json())
  except ValidationError as e:
    raise e

  try:
    response, status_code = processor.process_pdf(pdf_request)
  except (StandardException, BaseCustomException) as e:
    raise e
  except Exception as e:
    raise e

  return SuccessResponse(SuccessCode.UPLOAD_SUCCESS, response).of(), status_code
