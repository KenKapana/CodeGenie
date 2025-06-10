def best_framework(dev_time: str) -> str:
    mapping = {
        "short": "Flask",
        "medium": "Django",
        "long": "FastAPI + React"
    }
    return mapping.get(dev_time, "Flask")