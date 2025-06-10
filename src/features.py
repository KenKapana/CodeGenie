import re
from collections import Counter
from typing import List


def extract_skill_frequency(texts: List[str], skills: List[str]) -> dict[str, int]:
    counts = Counter()
    pattern = re.compile(r"\b(\w+)\b")
    for text in texts:
        words = pattern.findall(text.lower())
        for skill in skills:
            counts[skill.lower()] += words.count(skill.lower())
    return dict(counts)