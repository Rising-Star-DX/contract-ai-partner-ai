import re
from typing import List
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk import find
from sklearn.manifold import TSNE

from app.blueprints.agreement.agreement_exception import AgreementException
from app.blueprints.standard.standard_exception import StandardException
from app.clients.openai_clients import get_embedding_sync_client
from app.common.constants import ARTICLE_CHUNK_PATTERN, \
  ARTICLE_CLAUSE_SEPARATOR, CLAUSE_HEADER_PATTERN, NUMBER_HEADER_PATTERN
from app.common.exception.error_code import ErrorCode
from app.containers.service_container import embedding_service
from app.schemas.chunk_schema import ClauseChunk, DocumentChunk
from app.schemas.chunk_schema import Document

MIN_CLAUSE_BODY_LENGTH = 10


def semantic_chunk_with_overlap(extracted_text: str,
    similarity_threshold: float = 0.88, max_tokens: int = 250,
    overlap: int = 1, visualize: bool = False) -> List[ClauseChunk]:
  sentences = split_into_sentences(extracted_text)
  sentences = [s for s in sentences if len(s.strip()) > MIN_CLAUSE_BODY_LENGTH]
  if not sentences:
    raise StandardException(ErrorCode.CHUNKING_FAIL)

  with get_embedding_sync_client() as embedding_client:
    embeddings = embedding_service.batch_sync_embed_texts(embedding_client,
                                                          sentences)

  chunks = []
  current_chunk = [sentences[0]]
  prev_embedding = embeddings[0]

  for i in range(1, len(sentences)):
    similarity = cosine(prev_embedding, embeddings[i])
    tentative_chunk = current_chunk + [sentences[i]]
    token_len = count_tokens(" ".join(tentative_chunk))

    if similarity < similarity_threshold or token_len > max_tokens:
      chunks.append(ClauseChunk(clause_content=" ".join(current_chunk)))
      current_chunk = current_chunk[-overlap:] + [sentences[i]]
    else:
      current_chunk.append(sentences[i])

    prev_embedding = embeddings[i]

  if current_chunk:
    chunks.append(ClauseChunk(clause_content=" ".join(current_chunk)))

  if visualize:
    try:
      visualize_embeddings_3d(embeddings, sentences, chunks)
    except Exception as e:
      print(f"[시각화 오류] {e}")

  if not chunks:
    raise StandardException(ErrorCode.CHUNKING_FAIL)

  return chunks


def cosine(a: List[float], b: List[float]) -> float:
  a = np.array(a).flatten()
  b = np.array(b).flatten()
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def append_chunk_if_valid(chunks: List[ClauseChunk], current_chunk: List[str]):
  chunk_text = " ".join(current_chunk)
  if len(chunk_text.strip()) >= MIN_CLAUSE_BODY_LENGTH:
    chunks.append(ClauseChunk(clause_content=chunk_text))


def visualize_embeddings_3d(embeddings: List[List[float]], sentences: List[str],
    chunks: List[ClauseChunk]):
  for idx, chunk in enumerate(chunks):
    print(
      f"[청크 {idx}] 길이: {len(chunk.clause_content)} / 토큰 수: {count_tokens(chunk.clause_content)}")

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
  plt.savefig("semantic_embedding_result")


def ensure_punkt():
  try:
    find('tokenizers/punkt')
  except LookupError:
    nltk.download('punkt')


def split_into_sentences(extracted_text: str):
  ensure_punkt()
  return nltk.sent_tokenize(extracted_text)


def count_tokens(text: str) -> int:
  encoding = tiktoken.encoding_for_model("gpt-4o-mini")
  return len(encoding.encode(text))


def split_text_by_pattern(text: str, pattern: str) -> List[str]:
  return re.split(pattern, text)


def chunk_by_article_and_clause_with_page(documents: List[Document],
    pattern: str) -> List[
  DocumentChunk]:
  chunks: List[DocumentChunk] = []

  for doc in documents:
    page = doc.metadata.page
    page_text = doc.page_content
    order_index = 1

    preamble_exists = check_if_preamble_exists_except_first_page(pattern,
                                                                 page_text)
    if preamble_exists:
      order_index, chunks = (
        chunk_preamble_content(pattern, page_text, chunks, page, order_index))

    matches = re.findall(pattern, page_text, flags=re.DOTALL)
    for header, body in matches:
      header_match = None

      if pattern == NUMBER_HEADER_PATTERN:
        header_match = parse_number_header(header)
      elif pattern == ARTICLE_CHUNK_PATTERN:
        header_match = parse_article_header(header)

      if not header_match:
        continue

      article_number, article_title = header_match
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


def parse_article_header(header: str) -> Tuple[int, str]:
  clean_header = header.replace(" ", "")

  if not clean_header.startswith("제") or "조" not in clean_header:
    raise AgreementException(ErrorCode.NOT_SUPPORTED_FORMAT)

  try:
    num_part = clean_header.split("조")[0].replace("제", "")
    title_part = clean_header.split("조")[1]
    title_without_parentheses = title_part.strip("【】()[]")
    return int(num_part), title_without_parentheses

  except Exception:
    raise AgreementException(ErrorCode.NOT_SUPPORTED_FORMAT)


def parse_number_header(header: str) -> Tuple[int, str]:
  clean_header = header.replace(" ", "")

  try:
    num_part = clean_header.split(".")[0]
    title_part = clean_header.split(".")[1]
    title_without_parentheses = title_part.strip("【】()[]")
    return int(num_part), title_without_parentheses

  except Exception:
    raise AgreementException(ErrorCode.NOT_SUPPORTED_FORMAT)


def check_if_preamble_exists_except_first_page(pattern: str,
    page_text: str) -> bool:
  return not is_page_text_starting_with_article_heading(pattern, page_text)


def is_page_text_starting_with_article_heading(heading: str,
    page_text: str) -> bool:
  lines = page_text.strip().splitlines()
  content_lines = [line for line in lines if not line.strip().startswith("페이지")]
  return bool(re.match(heading, content_lines[0])) if content_lines else False


def chunk_preamble_content(pattern: str, page_text: str,
    chunks: List[DocumentChunk], page: int, order_index: int) -> \
    Tuple[int, List[DocumentChunk]]:
  first_article_match = (
    re.search(pattern, page_text, flags=re.MULTILINE))

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


def chunk_by_paragraph(documents: List[Document]) -> List[DocumentChunk]:
  chunks = []
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=300,
      chunk_overlap=50,
      separators=["\n\n", "."]
  )

  for doc in documents:
    divided_text = text_splitter.split_text(doc.page_content)

    for idx, content in enumerate(divided_text, start=1):
      chunks.append(
          DocumentChunk(
              page=doc.metadata.page,
              clause_content=content,
              order_index=idx,
              clause_number=str(len(chunks) + 1)
          )
      )

  return chunks
