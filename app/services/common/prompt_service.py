import json
import logging
from typing import List, Any, Optional

from openai import AsyncAzureOpenAI

from app.schemas.analysis_response import SearchResult


def clean_markdown_block(response_text: str) -> str | None:
  response_text_cleaned = response_text

  if response_text_cleaned.startswith(
      "```json") and response_text_cleaned.endswith("```"):
    return response_text_cleaned[7:-3].strip()
  elif response_text_cleaned.startswith(
      "```") and response_text_cleaned.endswith("```"):
    return response_text_cleaned[3:-3].strip()

  try:
    parsed_response = json.loads(response_text_cleaned)
  except json.JSONDecodeError as e:
    logging.error(
        f"[PromptService]: jsonDecodeError: {e} | raw response: {response_text_cleaned}")
    return None

  return parsed_response


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
      search_results: List[SearchResult], clause_content: str) -> Optional[
    dict[str, Any]]:

    input_data = {
      "clause_content": clause_content,
      "proof_text": [item.proof_text for item in search_results],
      "incorrect_text": [item.incorrect_text for item in search_results],
      "corrected_text": [item.corrected_text for item in search_results],
      "term": [x for item in search_results for x in[item.term] + item.keywords],
      "meaning_difference": [item.meaning_difference for item in search_results],
      "definition": [item.definition for item in search_results],
    }

    response = await prompt_client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "developer",
            "content":
              f"""
                너는 한국에서 계약서 및 법률 문서를 검토하는 최고의 변호사야.
                계약서에서 법률 위반 가능성이 있는 부분을 정확히 찾아내고,
                그 부분을 교정할 때 법적인 근거를 설명해야 해.
                특히 계약서 내 용어 사용이 오해를 일으킬 수 있는 경우, 
                관련 법률 용어의 정의와 해석 차이를 기준으로 다시 설명해 줘야 해.
              """
          },
          {
            "role": "user",
            "content":
              f"""
            입력 데이터를 참고해서 계약서 문장에서 부당한 문구가 있는지 찾아 수정해주세요.

            [특히 고려해야 할 사항]
            - 계약서 문장이 **법적 요건에 맞지 않거나**, **근로자에게 일방적으로 불리한 조건**을 담고 있다면 반드시 교정이 필요합니다.
            - 계약서 문장에 등장하는 특정 단어가 법률용어(`term`)와 같거나 유사할 경우, 해당 단어의 **정의(`definition`)를 기준**으로 문장이 잘못 쓰였는지 검토해 주세요.
            - 비전문가와 전문가 사이에 **해석 차이의 여지가 있다면 (`meaning_difference`)**, 그 위험성을 `proofText`에 반드시 설명해 주세요.
            - 문법적 오류보다는 **내용의 법적 타당성**에 집중해 주세요.
            - `proofText`에는 어떤 입력 변수명도 그대로 포함시키지 마세요.
            - 계약서 문장의 위배 확률이 높아 보인다면 `violation_score`를 높게 반환해 주세요.
    
            [입력 데이터 설명]
            - clause_content: 계약서 문장
            - proof_text: 법률 문서의 문장 목록
            - incorrect_text: 법률 위반할 가능성이 있는 예시 문장 
            - corrected_text: 법률 위반 가능성이 있는 예시 문장을 올바르게 수정한 문장 목록
            - term: 계약서 문장에서 등장하거나 유사한 법률 용어
            - meaning_difference: 이 용어에 대해 비전문가와 전문가의 해석 차이가 발생할 수 있는 경우 설명
            - definition: 해당 법률 용어의 사전적 또는 법적 정의

            [출력 형식]
            각 항목은 반드시 문자열(string) 형태로 출력할 것:
            {{
              "correctedText": "계약서의 문장을 올바르게 교정한 문장",
              "proofText": "입력 데이터를 참조해 잘못된 포인트와 그 이유",
              "violation_score": "계약서 문장의 위배 확률로, 0.000 ~ 1.000 사이의 소수점 셋째 자리까지 정확한 문자열"
            }}
            
            주의 사항:
            - JSON 외의 다른 텍스트는 절대 출력하지 마세요.
            - `violation_score`는 반드시 **의미 있는 소수점 셋째 자리까지** 생성하되, 단순히 '0.750', '0.500'과 같은 반복 패턴이 아닌, **상황에 맞게 다양하고 구체적인 값**을 반환하세요.

            [입력 데이터]
            {json.dumps(input_data, ensure_ascii=False, indent=2)}
          """
          }
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    response_text = response.choices[0].message.content
    return clean_markdown_block(response_text)

  async def original_correct_contract(self, prompt_client: AsyncAzureOpenAI,
      clause_content: str, proof_text: List[str],
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

    response = await prompt_client.chat.completions.create(
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
            각 항목은 반드시 문자열(string) 형태로 출력할 것:
            {{
              "correctedText": "계약서의 문장을 올바르게 교정한 문장",
              "proofText": "입력 데이터를 참조해 잘못된 포인트와 그 이유",
              "violation_score": "0.000 ~ 1.000 사이의 소수점 셋째 자리까지 정확한 문자열 (예: '0.742', '0.687', '0.913')"
            }}

            주의 사항:
            - JSON 외의 다른 텍스트는 절대 출력하지 마세요.
            - `violation_score`는 반드시 **의미 있는 소수점 셋째 자리까지** 생성해 주세요.
            - 단순히 '0.750', '0.500'과 같이 반복되는 패턴이 아닌, **상황에 맞게 다양하고 구체적인 값**을 반환하세요.

          """
          }
        ],
        temperature=0.1,
        max_tokens=1024,
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
