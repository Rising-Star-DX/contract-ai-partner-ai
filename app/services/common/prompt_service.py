import json
import logging
from typing import List, Any, Optional

from openai import AsyncAzureOpenAI

import re

from app.schemas.analysis_response import SearchResult


def clean_incorrect_part(text: str) -> str:
  particles = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '보다', '로', '과',
               '와']

  for particle in particles:
    # 마침표, 느낌표, 물음표 다음엔 조사 결합을 시도하지 않도록 예외 처리
    # 단어 + 공백 + 조사 이고, 그 앞에 구두점이 **없을 때만** 붙이기
    pattern = rf'(?<![\.!?])([가-힣])\s+{particle}(?=[\s\.,\)\"\'\”\’\]]|$)'
    replacement = rf'\1{particle}'
    text = re.sub(pattern, replacement, text)

  # 2칸 이상 공백은 1칸으로 줄이기
  text = re.sub(r'\s{2,}', ' ', text)

  return text


def clean_markdown_block(response_text: str) -> dict | None:
  response_text_cleaned = response_text

  if response_text_cleaned.startswith(
      "```json") and response_text_cleaned.endswith("```"):
    response_text_cleaned = response_text_cleaned[7:-3].strip()
  elif response_text_cleaned.startswith(
      "```") and response_text_cleaned.endswith("```"):
    response_text_cleaned = response_text_cleaned[3:-3].strip()

  try:
    parsed_response = json.loads(response_text_cleaned)
    if "incorrectPart" in parsed_response:
      parsed_response["incorrectPart"] = clean_incorrect_part(
          parsed_response["incorrectPart"])

    return parsed_response

  except json.JSONDecodeError as e:
    logging.error(
      f"[PromptService]: jsonDecodeError: {e} | raw response: {response_text_cleaned}")
    return None

class PromptService:
  def __init__(self, deployment_name):
    self.deployment_name = deployment_name

  async def make_correction_data(self, prompt_client: AsyncAzureOpenAI,
      clause_content: str) -> Any | None:
    response = await prompt_client.chat.completions.create(
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

    response_text = response.choices[0].message.content
    return clean_markdown_block(response_text)


  async def extract_keywords(self, prompt_client: AsyncAzureOpenAI,
      clause_content: str) -> Any | None:
    response = await prompt_client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "user",
            "content": f"""
              너는 법률 문서를 분석하고, **해석의 위험성과 핵심 개념을 추출**하는 전문가야.
        
              아래 문장을 기반으로 다음 두 가지 정보를 추출해:
              1. **meaning_difference**: 비전문가와 전문가 사이에 해석 차이가 발생할 수 있는 예시 상황을 한 문장으로 설명해줘.  
                 - 예시는 현실 계약에서 이 문장이 어떻게 오해될 수 있는지를 설명해야 해  
                 - 반드시 '비전문가는 ~ / 전문가는 ~' 식으로 해석 차이를 비교해서 써줘
        
              2. **keyword**: 문장의 핵심 개념 또는 법률적 쟁점을 최대 3개까지 추출해줘  
                 - 키워드는 원문에서 핵심이 되는 단어 또는 유사한 의미를 갖는 단어로 작성해줘  
                 - 반드시 Python 리스트 형태로 제공해
        
              다음과 같은 JSON 형식으로만 응답해줘. 그 외의 설명은 절대 하지 마.
        
              예시 출력 형식:
              {{
                "meaning_difference": "비전문가는 '협의'를 법적 구속력이 있는 절차로 해석할 수 있지만, 전문가는 단순한 의견 교환으로 본다. 이 문장은 계약 해지 요건과 관련된 해석 차이를 유발할 수 있다.",
                "keyword": ["협의", "계약 해지", "구속력"]
              }}

              원문:
              \"\"\"{clause_content}\"\"\"
              """

          }
        ],
        temperature=0.8,
        max_tokens=512,
        top_p=1
    )

    response_text = response.choices[0].message.content
    return clean_markdown_block(response_text)

  async def correct_contract(self, prompt_client: AsyncAzureOpenAI,
      clause_content: str) -> Optional[
    dict[str, Any]]:

    clause_content = clause_content.replace("\n", " ")
    clause_content = clause_content.replace("+", "")
    clause_content = clause_content.replace("!!!", " ")

    input_data = {
      "clause_content": clause_content
    }

    response = await prompt_client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "user",
            "content":
              f"""
            입력 데이터를 참고해서 계약서 문장에서 부당한 문구가 있는지 찾아 수정해주세요.

            
            [입력 데이터]
            {json.dumps(input_data, ensure_ascii=False, indent=2)}
            
            [출력 형식]
            출력은 dict 형태이며, value 값은 반드시 문자열(string) 형태로 출력할 것:
            {{
              "correctedText": "계약서의 문장을 올바르게 교정한 문장",
              "proofText": "입력 데이터를 참조해 잘못된 포인트와 그 이유",
              "violation_score": "0.000 ~ 1.000 사이의 소수점 셋째 자리까지의 문자열"
              "incorrectPart": clause_content에서 문제가 되는 부분 길이는 최대 단어 5개까지 똑같이 반환해주세요.
          """
          }
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    response_text = response.choices[0].message.content
    return clean_markdown_block(response_text)