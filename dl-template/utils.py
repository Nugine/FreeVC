from typing import List, Any
from pathlib import Path

import json

"""
https://github.com/Nugine/dl-template
"""


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    with open(str(path), "w", encoding="utf8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(str(path), "r", encoding="utf8") as f:
        return json.load(f)


def read_txt_lines(path: str | Path) -> List[str]:
    ans = []
    with open(str(path), "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                ans.append(line)
    return ans
