"""Game dataclass and detection manifest serialization."""

from dataclasses import dataclass


@dataclass
class Game:
    num: int
    start: int
    end: int

    @property
    def pages_human(self):
        return f"{self.start + 1}-{self.end + 1}"

    def to_dict(self):
        return {"game_num": self.num,
                "start_page": self.start,
                "end_page": self.end,
                "pages_human": self.pages_human}

    @classmethod
    def from_dict(cls, d):
        return Game(d["game_num"],d["start_page"],d["end_page"])
