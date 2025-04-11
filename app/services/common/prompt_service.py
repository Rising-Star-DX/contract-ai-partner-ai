import json
import logging
import re
from typing import List, Any, Optional

import httpx
from openai import AsyncOpenAI

from app.clients.openai_clients import sync_openai_client, prompt_async_client, \
  openai_client


def clean_markdown_block(response_text: str) -> str:
  response_text_cleaned = response_text

  if response_text_cleaned.startswith(
      "```json") and response_text_cleaned.endswith("```"):
    return response_text_cleaned[7:-3].strip()
  elif response_text_cleaned.startswith(
      "```") and response_text_cleaned.endswith("```"):
    return response_text_cleaned[3:-3].strip()
  else:
    return response_text_cleaned


class PromptService:
  def __init__(self, deployment_name):
    self.deployment_name = deployment_name

  async def make_correction_data(self, clause_content: str) -> Any | None:
    # async with httpx.AsyncClient() as httpx_client:
    #   async with AsyncOpenAI(http_client=httpx_client) as client:
    response = await prompt_async_client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "user",
            "content": f"""
              입력 문장을 검토하여 '계약 체결자에게 불리하게 작용할 수 있는 위배 문장'을 생성하고, 이를 공정하게 수정한 교정 문장을 제시하는 전문가야.
              - 생성된 불공정 문장은 단순한 원문 복사나 요약이 아니라, **원문을 읽는 사람이 특정 방식으로 오해하거나 불리하게 해석할 가능성이 있는 문장으로 추정하여 구성**할 것
              - 즉, 실제로 문서에 존재하지 않더라도, **문맥을 곡해하거나 핵심 조건을 생략함으로써 발생할 수 있는 해석상의 불리함**을 적극적으로 반영할 것

              다음 조건을 반드시 지켜:
              - 원문에 명확한 위배 문장이 없어 보여도, **해석 가능성이나 맥락에 근거해 위배 소지가 있는 문장을 추정해서 생성할 것**
              - **절대 원문 그대로 반환하지 말 것**
              - **맞춤법, 어휘 표현 개선은 하지 말고, 오직 불공정성/위배 가능성에만 초점 둘 것**
      
              예시 출력 형식:
              {{
                "incorrect_text": "업무를 인계받지 못한 공무원은 아무런 책임이 없다.",
                "corrected_text": "업무를 인계받지 못한 공무원도, 특별한 사유에 해당하지 않는 경우에는 인계 지연에 따른 책임을 부담할 수 있다. 이는 '책임 없음'으로 오해될 수 있는 표현을 방지하기 위함이다."
              }}
      
              원문:
              \"\"\"{clause_content}\"\"\"
      
              불공정 판단 기준:
              - 특정 당사자의 권리를 과도하게 제한하거나
              - 의무를 일방에게만 지우거나
              - 해석 여지로 인해 불리하게 적용될 가능성이 있으며
              - 효력 발생 조건이 불명확하거나 불공정한 경우
      
              지금 문장을 분석해 위 기준에 따라 불리할 수 있는 위배 문장을 생성하고, 공정하게 수정해서 JSON으로 반환해.
              """

          }
        ],
        temperature=0.8,
        max_tokens=1000,
        top_p=1
    )

    raw_response_text = response.choices[
      0].message.content if response.choices else ''
    cleaned_text = clean_markdown_block(raw_response_text)

    try:
      parsed_response = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
      logging.error(
          f"[PromptService]: jsonDecodeError: {e} | raw response: {cleaned_text}")
      return None

    return parsed_response

  async def correct_contract(self, clause_content: str, proof_text: List[str],
      incorrect_text: List[str], corrected_text: List[str]) -> Optional[
    dict[str, Any]]:

    clause_content = clause_content.replace("\n", " ")
    clause_content = clause_content.replace("+", "")
    clause_content = clause_content.replace("!!!", "")

    # ✅ JSON 형식으로 변환할 데이터
    input_data = {
      "clause_content": clause_content,
      "proof_text": proof_text,
      "incorrect_text": incorrect_text,
      "corrected_text": corrected_text
    }

    # async with httpx.AsyncClient() as httpx_client:
    #   async with AsyncOpenAI(http_client=httpx_client) as client:
    response = await prompt_async_client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "developer",
            "content":
              f"""
                너는 한국에서 계약서 및 법률 문서를 검토하는 최고의 변호사야.
                계약서에서 법률 위반 가능성이 있는 부분을 정확히 찾아내고,
                그 부분을 교정할 때 법적인 근거를 설명해야 해.
              """
          },
          {
            "role": "user",
            "content":
              f"""
                입력 데이터를 참고해서
                계약서 문장에서 부당한 문구가 있는지 찾아 수정해주세요.
                특히, 법적인 요구 사항에 맞지 않는 부분, 
                노동자에게 불리한 문장을 교정하고 그 이유를 proofText 에 설명해 주세요.
                입력 데이터 변수명을 proofText 에 포함시키면 안됩니다.
                근로자와의 협의를 덜 고려해 주세요.                    

            틀린 확률이 높아보인다면 violation_score를 높게 반환해 주세요.
            문법적인 측면이 아닌 내용적인 측면에서 교정해 주세요.

            [입력 데이터 설명]
            - clause_content: 계약서 문장
            - proof_text: 법률 문서의 문장 목록
            - incorrect_text: 법률 위반할 가능성이 있는 예시 문장 
            - corrected_text: 법률 위반 가능성이 있는 예시 문장을 올바르게 수정한 문장 목록

            [입력 데이터]
            {json.dumps(input_data, ensure_ascii=False, indent=2)}

            [출력 형식]
            {{
                "clause_content": 계약서 원문
                "correctedText": "계약서의 문장을 올바르게 교정한 문장",
                "proofText": 입력데이터를 참조해 잘못된 포인트와 이유"
                "violation_score": "문장이 틀리거나 법률을 위배할 확률 "
            }}

            json 바깥에는 아무것도 반환하지 마세요
            violation_score는 0~1 범위의 소수점 셋째자리까지 반환해 주세요. 

          """
        }
      ],
      temperature=0.1,
      max_tokens=512,
    )

    response_text = response.choices[0].message.content
    response_text_cleaned = response_text.strip()

    if response_text_cleaned.startswith(
        "```json") and response_text_cleaned.endswith("```"):
      pure_json = response_text_cleaned[7:-3].strip()
    elif response_text_cleaned.startswith(
        "```") and response_text_cleaned.endswith("```"):
      pure_json = response_text_cleaned[3:-3].strip()
    else:
      pure_json = response_text_cleaned

    try:
      parsed_response = json.loads(pure_json)
    except json.JSONDecodeError:
      logging.error(
          f"[PromptService] 응답이 JSON 형식이 아님:\n{response_text_cleaned}"
      )
      return None

    return parsed_response
