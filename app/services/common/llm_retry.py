import asyncio
import logging
from typing import Callable, Coroutine, Any

from app.common.constants import MAX_RETRIES, LLM_TIMEOUT
from app.common.exception.custom_exception import CommonException
from app.common.exception.error_code import ErrorCode


async def retry_llm_call(
    func: Callable[..., Coroutine[Any, Any, dict]],
    *args,
    required_keys: set | None = None) -> dict | None:
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      result = await asyncio.wait_for(func(*args), timeout=LLM_TIMEOUT)
      if isinstance(result, dict) or required_keys.issubset(result.keys()):
        return result
      logging.warning(
          "[retry_llm_call]: llm 응답 필수 키 누락 / dict 구조 아님")

    except asyncio.TimeoutError:
      logging.warning(
          f"[retry_llm_call]: Timeout {attempt}/{MAX_RETRIES}")
      if attempt == MAX_RETRIES:
        raise CommonException(ErrorCode.LLM_RESPONSE_TIMEOUT)

    except Exception as e:
      logging.warning(
          f"[retry_llm_call]: 재요청 발생 {attempt}/{MAX_RETRIES} {e}")
      if attempt == MAX_RETRIES:
        raise CommonException(ErrorCode.STANDARD_REVIEW_FAIL)

    await asyncio.sleep(0.5 * attempt)

  raise CommonException(ErrorCode.PROMPT_MAX_TRIAL_FAILED)
