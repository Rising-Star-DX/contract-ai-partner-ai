from dataclasses import dataclass, field
from typing import List


@dataclass
class VectorPayload:
  standard_id: int
  incorrect_text: str | None
  proof_text: str | None
  corrected_text: str | None
  created_at: str

  def to_dict(self) -> dict:
    return {
      "standard_id": self.standard_id or "",
      "incorrect_text": self.incorrect_text or "",
      "proof_text": self.proof_text or "",
      "corrected_text": self.corrected_text or "",
      "created_at": self.created_at or ""
    }

  def embedding_input(self) -> str:
    return self.proof_text


@dataclass
class WordPayload:
  standard_id: int
  definition: str
  term: str
  meaning_difference: str
  created_at: str
  keywords: List[str] = field(default_factory=list)

  def to_dict(self) -> dict:
    return {
      "standard_id": self.standard_id or "",
      "definition": self.definition or "",
      "term": self.term or "",
      "meaning_difference": self.meaning_difference or "",
      "keywords": self.keywords or [],
      "created_at": self.created_at or ""
    }

  def embedding_input(self) -> str:
    return f"{self.term} {self.definition} {self.meaning_difference}"

