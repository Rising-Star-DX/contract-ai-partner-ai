from dataclasses import dataclass, asdict


@dataclass
class VectorPayload:
  standard_id: float
  category: str
  incorrect_text: str
  proof_text: str
  corrected_text: str

  def to_dict(self):
    return asdict(self)

