import json
import logging
import re
from typing import List, Any, Optional


class PromptService:
  def __init__(self, client, deployment_name):
    self.client = client
    self.deployment_name = deployment_name

  async def make_correction_data(self, clause_content: str) -> Any | None:
    response = await self.client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "user",
            "content": f"""
              다음 지시문에 맞게 반환 해줘
              반드시 JSON 코드 블록 (```json ...) 을 사용하지 말고, 그냥 JSON 객체만 반환해
              JSON 문자열 내 줄바꿈(\n)이 포함되지 않도록 한 줄로 작성해줘

              문서 원문:
              \"\"\"
              {clause_content}
              \"\"\"

              [생성할 JSON 형식]
              {{
                "incorrect_text": "문서 원문을 기준으로 법률을 위반할 수 있는 예시 문장",
                "corrected_text": "위의 문장을 교정한 문장"
              }}

              조건:
              - 위반 문장과 교정 문장은 서로 논리적으로 연결되어야 함.
              - 결과는 반드시 JSON만 반환. 설명, 추가 텍스트 없이.
            """
          }
        ],
        temperature=0.5,
        max_tokens=512,
        top_p=1
    )

    response_text = response.choices[0].message.content
    response_text_cleaned = re.sub(r'(?<!\\)\n', ' ', response_text).strip()
    try:
      parsed_response = json.loads(response_text_cleaned)
    except json.JSONDecodeError:
      logging.error(f"[PromptService]: jsonDecodeError response_text {response_text_cleaned}")
      return None

    return parsed_response


  async def correct_contract(self, clause_content: str, proof_text: List[str],
      incorrect_text: List[str], corrected_text: List[str]):
    # ✅ JSON 형식으로 변환할 데이터
    input_data = {
      "clause_content": clause_content,
      "proof_text": proof_text,
      "incorrect_text": incorrect_text,
      "corrected_text": corrected_text
    }

    response = await self.client.chat.completions.create(
        model=self.deployment_name,
        messages=[
          {
            "role": "user",
            "content": f"""
                    예시 위배 문장과 예시 위배 교정 문장을 참고해서 
                    입력받은 계약서 중 틀린 문장을 기준 문서(법률 조항)과 비교하여 교정해줘
                    입력받은 텍스트에서 '\n' 엔터키는 지우고 작업 수행해.
                    그리고 참고한 자료를 기반으로 위배된 확률을 계산해줘
                    틀린 확률이 높다면 accuracy 를 높여줘
                    accuracy 0~1 범위의 float 형태로 출력해줘
                    반드시 JSON 코드 블록 (```json ...) 을 사용하지 말고, 그냥 JSON 객체만 반환해.
                    맞춤법, 띄어쓰기를 수정하지 말고 계약서 내용에서 틀린걸 수정해줘
                    교정 전 후 값이 일치하거나 의미차이가 없다면 데이터 출력 하지 말아줘

                    [입력 데이터 설명]
                    - clause_content: 사용자가 입력한 계약서의 문장 (수정해야 하는 문장)
                    - proof_text: 기준이 되는 법률 문서의 문장 목록 (계약서와 비교할 법률 조항들)
                    - incorrect_text: 법률을 위반할 가능성이 있는 문장 예시 목록
                    - corrected_text: 법률 위반 가능성이 있는 예시 문장을 올바르게 수정한 문장 목록

                    [입력 데이터]
                    {json.dumps(input_data, ensure_ascii=False, indent=2)}

                    [출력 형식]
                    {{
                        "incorrect_text": 계약서 원문
                        "corrected_text": "계약서의 문장을 올바르게 교정한 문장",
                        "proof_text": 입력데이터를 참조해 잘못된 포인트와 이유"
                        "accuracy": "위배된 비율, 신뢰도, 소수점 셋째자리까지 반환"
                    }}

                    [조건]
                    - 위반 문장과 교정 문장은 서로 논리적으로 연결되어야 함,
                    - 결과는 반드시 JSON 형식으로 반환해
                """
          }
        ],
        temperature=0.5,
        top_p=1
    )

    response_text = response.choices[0].message.content

    # ✅ JSON 변환 시도
    try:
      parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
      logging.error(f"❌ OpenAI 응답이 JSON 형식이 아님: {response_text}")
      return None  # JSON 변환 실패 시 None 반환

    return parsed_response
