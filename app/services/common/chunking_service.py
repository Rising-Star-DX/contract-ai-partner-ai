import os
import re
from typing import List
from typing import Optional, Tuple

import nltk
import numpy as np
import tiktoken
from nltk import find
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from app.blueprints.standard.standard_exception import StandardException
from app.common.constants import ARTICLE_CHUNK_PATTERN, ARTICLE_HEADER_PATTERN, \
  ARTICLE_CLAUSE_SEPARATOR, ARTICLE_HEADER_PARSE_PATTERN, CLAUSE_HEADER_PATTERN
from app.common.exception.error_code import ErrorCode
from app.containers.service_container import embedding_service
from app.schemas.chunk_schema import ArticleChunk, ClauseChunk, DocumentChunk
from app.schemas.chunk_schema import Document


MIN_CLAUSE_BODY_LENGTH = 20
def semantic_chunk(extracted_text: str, similarity_threshold: float = 0.9,
    max_tokens: int = 250, visualize: bool = False) -> \
    List[str]:
  sentences = split_into_sentences(extracted_text)
  if not sentences:
    raise StandardException(ErrorCode.CHUNKING_FAIL)

  sentences = [s for s in sentences if len(s.strip()) > MIN_CLAUSE_BODY_LENGTH]
  embeddings = embedding_service.get_embeddings(sentences)

  if visualize:
    try:
        visualize_embeddings_3d(embeddings, sentences)
    except Exception as e:
        print(f"[시각화 오류] {e}")

  chunks = []
  current_chunk = [sentences[0]]
  prev_embedding = embeddings[0]

  for i in range(1, len(sentences)):
    similarity = cosine_similarity([prev_embedding], [embeddings[i]])[0][0]
    tentative_chunk = " ".join(current_chunk + [sentences[i]])
    token_len = count_tokens(tentative_chunk)

    if similarity < similarity_threshold or token_len > max_tokens:
      chunk_text = " ".join(current_chunk)
      chunks.append(chunk_text)
      current_chunk = [sentences[i]]
    else:
      current_chunk.append(sentences[i])

    prev_embedding = embeddings[i]

  chunk_text = " ".join(current_chunk)
  if len(chunk_text.strip()) >= MIN_CLAUSE_BODY_LENGTH:
    chunks.append(chunk_text)

  return chunks


def visualize_embeddings_3d(embeddings: List[List[float]], sentences: List[str]):
  plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'AppleGothic'
  plt.rcParams['axes.unicode_minus'] = False

  tsne = TSNE(n_components=3, random_state=0, perplexity=5)
  reduced = tsne.fit_transform(np.array(embeddings))

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')

  for i, text in enumerate(sentences):
    x, y, z = reduced[i]
    ax.scatter(x, y, z)
    ax.text(x, y, z, text[:30] + "...", size=9)
  ax.set_title("Semantic Embedding 결과")
  plt.tight_layout()
  plt.show()


def ensure_punkt():
  try:
    find('tokenizers/punkt')
  except LookupError:
    nltk.download('punkt')


def split_into_sentences(extracted_text: str):
  ensure_punkt()
  return nltk.sent_tokenize(extracted_text)


def count_tokens(text: str) -> int:
  encoding = tiktoken.encoding_for_model(
      os.getenv("AZURE_PROMPT_OPENAI_DEPLOYMENT_NAME"))
  return len(encoding.encode(text))


def split_text_by_pattern(text: str, pattern: str) -> List[str]:
  return re.split(pattern, text)


def chunk_by_article_and_clause(extracted_text: str) -> List[ArticleChunk]:
  article_pattern = r'\n(제\s*\d+조(?:\([^)]+\))?)'  # 조(Article) 기준 정규식

  chunks = split_text_by_pattern(extracted_text, article_pattern)
  result: List[ArticleChunk] = []

  for i in range(1, len(chunks), 2):
    article_title = chunks[i].strip()
    article_body = chunks[i + 1].strip() if i + 1 < len(chunks) else ""

    clauses = []

    # ⭐ 항이 ① 또는 1. 로만 시작한다는 전제 (따라서 기준문서도 이에 맞는 문서만 필요)
    first_clause_match = re.search(r'(①|1\.)', article_body)
    if first_clause_match is None:
      result.append(
          ArticleChunk(article_title=article_title + article_body, clauses=[]))
      continue

    match_idx = first_clause_match.start()
    article_title += ' ' + article_body[:match_idx]
    if first_clause_match:
      clause_pattern = r'([\n\s]*[①-⑨])' if first_clause_match.group(
          1) == '①' else r'(\n\s*\d+\.)'
      clause_chunks = split_text_by_pattern("\n" + article_body[match_idx:],
                                            clause_pattern)

      for j in range(1, len(clause_chunks), 2):
        clause_title = clause_chunks[j].strip()
        clause_body = clause_chunks[j + 1].strip() if j + 1 < len(
            clause_chunks) else ""

        if len(clause_body) >= MIN_CLAUSE_BODY_LENGTH:
          clauses.append(
              ClauseChunk(clause_number=clause_title,
                          clause_content=clause_body))
      result.append(ArticleChunk(article_title=article_title, clauses=clauses))

  return result


def chunk_by_article_and_clause_with_page(documents: List[Document]) -> List[
  DocumentChunk]:
  chunks: List[DocumentChunk] = []

  for doc in documents:
    page = doc.metadata.page
    page_text = doc.page_content
    order_index = 1

    preamble_exists = check_if_preamble_exists_except_first_page(page,
                                                                 page_text)
    if preamble_exists:
      order_index, chunks = (
        chunk_preamble_content(page_text, chunks, page, order_index))

    matches = re.findall(ARTICLE_CHUNK_PATTERN, page_text, flags=re.DOTALL)
    for header, body in matches:
      header_match = re.match(ARTICLE_HEADER_PARSE_PATTERN, header)
      if not header_match:
        continue

      article_number = header_match.group(1)
      article_title = header_match.group(2) or header_match.group(3)
      article_body = body.strip()

      first_clause_match = re.search(CLAUSE_HEADER_PATTERN, article_body)
      if first_clause_match and article_body.startswith(
          first_clause_match.group(1)):

        clause_chunks = (
          split_by_clause_header_pattern(
              first_clause_match.group(1), "\n" + article_body))

        for j in range(1, len(clause_chunks), 2):
          clause_number = clause_chunks[j].strip()
          if clause_number.endswith("."):
            clause_number = clause_number[:-1]

          clause_content = clause_chunks[j + 1].strip() if j + 1 < len(
              clause_chunks) else ""

          if len(clause_content) >= MIN_CLAUSE_BODY_LENGTH:
            chunks.append(DocumentChunk(
                clause_content=f"{article_title}{ARTICLE_CLAUSE_SEPARATOR}\n{clause_content}",
                page=page,
                order_index=order_index,
                clause_number=f"제{article_number}조 {clause_number}항"
            ))
            order_index += 1
      else:
        if len(article_body) >= MIN_CLAUSE_BODY_LENGTH:
          chunks.append(DocumentChunk(
              clause_content=f"{article_title}{ARTICLE_CLAUSE_SEPARATOR}\n{article_body}",
              page=page,
              order_index=order_index,
              clause_number=f"제{article_number}조 1항"
          ))
          order_index += 1

  return chunks


def check_if_preamble_exists_except_first_page(page: int,
    page_text: str) -> bool:
  return page != 1 and not is_page_text_starting_with_article_heading(
      ARTICLE_HEADER_PATTERN, page_text
  )


def is_page_text_starting_with_article_heading(heading: str,
    page_text: str) -> bool:
  lines = page_text.strip().splitlines()
  content_lines = [line for line in lines if not line.strip().startswith("페이지")]
  return bool(re.match(heading, content_lines[0])) if content_lines else False


def chunk_preamble_content(page_text: str, chunks: List[DocumentChunk],
    page: int, order_index: int) -> Tuple[int, List[DocumentChunk]]:
  first_article_match = (
    re.search(ARTICLE_HEADER_PATTERN, page_text, flags=re.MULTILINE))

  preamble = page_text[
             :first_article_match.start()] if first_article_match else page_text
  return append_preamble(chunks, preamble, page, order_index)


def append_preamble(result: List[DocumentChunk], preamble: str, page: int,
    order_index: int) -> Tuple[int, List[DocumentChunk]]:
  if not result:
    return order_index, result

  pattern = get_clause_pattern(result[-1].clause_number)

  if not pattern:
    result.append(DocumentChunk(
        clause_content=preamble,
        page=page,
        order_index=order_index,
        clause_number=result[-1].clause_number
    ))
    return order_index + 1, result

  clause_chunks = split_text_by_pattern(preamble, pattern)
  lines = clause_chunks[0].strip().splitlines()
  content_lines = [line for line in lines if not line.strip().startswith("페이지")]

  result.append(DocumentChunk(
      clause_content="\n".join(content_lines),
      page=page,
      order_index=order_index,
      clause_number=result[-1].clause_number
  ))
  order_index += 1

  for j in range(1, len(clause_chunks), 2):
    clause_number = clause_chunks[j].strip()
    clause_content = clause_chunks[j + 1].strip() if j + 1 < len(
        clause_chunks) else ""

    if len(clause_content) >= MIN_CLAUSE_BODY_LENGTH:
      prev_clause_prefix = result[-1].clause_number.split(" ")[0]
      result.append(DocumentChunk(
          clause_content=clause_content,
          page=page,
          order_index=order_index,
          clause_number=f"{prev_clause_prefix} {clause_number}항"
      ))
      order_index += 1

  return order_index, result


def split_by_clause_header_pattern(clause_header: str, article_body: str) \
    -> List[str]:
  clause_pattern = ""

  if clause_header == '①':
    clause_pattern = r'([\n\s]*[①-⑨])'
  elif clause_header == '1.':
    clause_pattern = r'(\n\s*\d+\.)'
  elif clause_header == '(1)':
    clause_pattern = r'(\n\s*\(\d+\))'

  return split_text_by_pattern("\n" + article_body, clause_pattern)


def get_clause_pattern(clause_number: str) -> Optional[str]:
  parts = clause_number.split(" ")
  if len(parts) < 1:
    return None

  pattern = parts[1].strip()
  if re.match(r'[①-⑨]', pattern):
    return r'([\n\s]*[①-⑨])'
  elif re.match(r'\d+\.', pattern):
    return r'(\n\s*\d+\.)'
  return None
