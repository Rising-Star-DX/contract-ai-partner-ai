from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClauseChunk:
  clause_number: str
  clause_content: str

@dataclass
class ArticleChunk:
  article_title: str
  clauses: List[ClauseChunk]

@dataclass
class DocumentChunk:
  clause_content: str = ''
  page: int = 0
  order_index: int = 0
  clause_number: Optional[str] = None

@dataclass
class DocumentMetadata:
  page: int

@dataclass
class Document:
  page_content: str
  metadata: DocumentMetadata

@dataclass
class OCRDocument:
  content: str

