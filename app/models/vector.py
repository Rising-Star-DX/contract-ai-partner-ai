from dataclasses import dataclass


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

