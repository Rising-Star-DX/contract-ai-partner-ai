from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClauseData:
  order_index: int = 0
  page: int = 0
  position: List[List[float]] = field(default_factory=list)

@dataclass
class SearchResult:
  proof_text: str
  incorrect_text: str
  corrected_text: str
  term: str
  meaning_difference: str
  definition: str
  keywords: str

@dataclass
class RagResult:
  incorrect_text: str = ''
  corrected_text: Optional[str] = None
  proof_text: Optional[str] = None
  accuracy: Optional[float] = None
  clause_data: List[ClauseData] = field(default_factory=list)

@dataclass
class AnalysisResponse:
  total_page: int = 0
  chunks: List[RagResult] = field(default_factory=list)
  total_chunks: int = 0

@dataclass
class StandardResponse:
  result: str
  contents: List[str] = field(default_factory=list)

