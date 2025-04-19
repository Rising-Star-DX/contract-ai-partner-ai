from datetime import datetime

from app.containers.service_container import prompt_service
from app.models.vector import VectorPayload, WordPayload
from app.schemas.chunk_schema import ClauseChunk
from app.services.standard.vector_store.llm_retry import retry_llm_call

STANDARD_LLM_REQUIRED_KEYS = {"incorrect_text", "corrected_text"}
STANDARD_WORD_REQUIRED_KEYS = {"keyword", "meaning_difference"}


async def make_clause_payload(prompt_client, article, pdf_request,
    semaphore) -> VectorPayload | None:
  async with semaphore:
    result = await retry_llm_call(
        prompt_service.make_correction_data,
        prompt_client, article,
        required_keys=STANDARD_LLM_REQUIRED_KEYS
    )
    if not result:
      return None

  return VectorPayload(
      standard_id=pdf_request.id,
      incorrect_text=result.get("incorrect_text") or "",
      proof_text=article.clause_content,
      corrected_text=result.get("corrected_text") or "",
      created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  )


async def make_word_payload(prompt_client, article: ClauseChunk, pdf_request,
    semaphore) -> WordPayload | None:
  async with semaphore:
    result = await retry_llm_call(
        prompt_service.extract_keywords,
        prompt_client, article.clause_content,
        required_keys=STANDARD_WORD_REQUIRED_KEYS
    )
    if not result:
      return None

  return WordPayload(
      standard_id=pdf_request.id,
      definition=article.clause_content,
      term=article.clause_number,
      meaning_difference=result.get("meaning_difference") or "",
      keywords=result.get("keyword") or [],
      created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  )