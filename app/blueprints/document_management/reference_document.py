import io

from flask import Blueprint, request, jsonify, send_file
from app.services import s3_service
from app.services import pdf_service

bp = Blueprint('/reference_document', __name__, url_prefix="/flask/reference-document")

@bp.route('', methods=['POST'])
def process_pdf_from_s3():
  data = request.json
  s3_path = data.get("s3Path") # 없으면 None을 반환한다.

  if not s3_path:
    return jsonify({"error": "Missing s3Path"}), 400

  try:
    # 1️⃣ s3에서 문서(pdf) 가져오기 (메모리 내)
    s3_stream = s3_service.s3_get_object(s3_path)

    if s3_stream is None:
      raise FileNotFoundError(f"S3에서 파일을 찾을 수 없습니다: {s3_path}")

    pdf_bytes_io = io.BytesIO(s3_stream.read())
    pdf_bytes_io.seek(0)

    # 2️⃣ PDF에서 텍스트 추출
    extracted_text = pdf_service.extract_text_from_pdf(pdf_bytes_io)
    print(extracted_text) # 확인용

    # 3️⃣ 텍스트를 청킹하여 분할 추가해야 함
    # chunk

    # 파일이 제대로 받아와져 있음을 확인할 용도..
    return send_file(
        pdf_bytes_io,
        mimetype='application/pdf',
        as_attachment=True,
        download_name="downloaded_document.pdf"  # 원하는 파일명 지정 가능
    )

  except ValueError as e:
    return jsonify({"error": str(e)}), 400
  except Exception as e:
    return jsonify({"error": "S3 download failed", "details": str(e)}), 500
