from enum import Enum
from http import HTTPStatus


class ErrorCode(Enum):
  # global 에러
  REQUEST_UNMATCH = (HTTPStatus.BAD_REQUEST, "G001", "요청 json 매핑 불일치")
  URL_NOT_FOUND = (HTTPStatus.BAD_REQUEST, "G002", "존재하지 않는 url 요청")

  # standard 관련 에러
  UPLOAD_FAIL = (HTTPStatus.INTERNAL_SERVER_ERROR, "S001", "기준 문서 저장 과정 중 에러 발생")
  DELETE_FAIL = (HTTPStatus.INTERNAL_SERVER_ERROR, "S002", "삭제 과정 중 에러 발생")
  CHUNKING_FAIL = (HTTPStatus.INTERNAL_SERVER_ERROR, "S003", "청킹된 데이터 없음")
  PROMPT_MAX_TRIAL_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "S004", "프롬프트 응답 재요청 3회 시도 진행 결과 json 불일치")
  COLLECTION_NOT_FOUND = (HTTPStatus.BAD_REQUEST, "S005", "요청 카테고리에 해당하는 컬렉션 없음")

  # common 에러
  DATA_TYPE_NOT_MATCH = (HTTPStatus.BAD_REQUEST, "C001", "데이터의 형식이 맞지 않음")
  FILE_LOAD_FAILED = (HTTPStatus.BAD_REQUEST, "C002", "S3에서 파일을 가져 오지 못함")
  S3_CLIENT_ERROR = (HTTPStatus.INTERNAL_SERVER_ERROR, "C003", "S3 연결 도중 에러 발생")
  CONVERT_TO_IO_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C004", "스트림을 io 바이트로 변환하는데 에러 발생")
  INNER_DATA_ERROR = (HTTPStatus.INTERNAL_SERVER_ERROR, "C005", "페이지가 없거나 내부 구조가 깨져 있음")
  FILE_FORMAT_INVALID = (HTTPStatus.INTERNAL_SERVER_ERROR, "C006", "파일이 깨졌거나 지원 불가")
  S3_STREAM_READ_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C007", "S3 스트림을 읽는 중 에러 발생")
  INVALID_JSON_FORMAT = (HTTPStatus.BAD_REQUEST, "C008", "입력 형태가 JSON 형식에 맞지 않음")
  FIELD_MISSING = (HTTPStatus.BAD_REQUEST, "C009", "JSON에 필수 필드가 누락됨")
  INVALID_URL_PARAMETER = (HTTPStatus.BAD_REQUEST, "C010", "url 경로에 필수 항목이 누락됨")
  CANNOT_CONVERT_TO_NUM = (HTTPStatus.BAD_REQUEST, "C011", "URL 매개변수를 숫자로 변환할 수 없음")
  EMBEDDING_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C012", "임베딩 실패")
  NO_POINTS_GENERATED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C013", "생성된 포인트 없음")
  QDRANT_CONNECTION_TIMEOUT = (HTTPStatus.INTERNAL_SERVER_ERROR, "C014", "Qdrant 연결 타임 아웃")
  QDRANT_NOT_STARTED = (HTTPStatus.NOT_FOUND, "C015", "Qdrant 실행 여부 확인 필요")
  UNSUPPORTED_FILE_TYPE = (HTTPStatus.BAD_REQUEST, "C016", "지원되지 않는 타입의 파일")
  NO_TEXTS_EXTRACTED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C017", "해당 파일에서 추출된 텍스트 없음")
  PDF_LOAD_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "C018", "PDF 로딩 실패")

  # agreement 관련 에러
  REVIEW_FAIL = (HTTPStatus.INTERNAL_SERVER_ERROR, "A001", "AI 검토 보고서 생성 작업 중 에러 발생")
  INVALID_S3_PATH = (HTTPStatus.BAD_REQUEST, "A002", "openai 파싱에 적합하지 않은 s3 경로")
  NO_SEARCH_RESULT = (HTTPStatus.INTERNAL_SERVER_ERROR, "A003", "Qdrant 내 검색된 데이터 없음")
  QDRANT_SEARCH_FAILED = (HTTPStatus.INTERNAL_SERVER_ERROR, "A004", "Qdrant 검색 결과 반환 실패")
  NO_POINTS_FOUND = (HTTPStatus.INTERNAL_SERVER_ERROR, "A005", "Qdrant 내 조회 가능한 포인트 없음")

  def __init__(self, status: HTTPStatus, code: str, message: str):
    self.status = status
    self.code = code
    self.message = message
