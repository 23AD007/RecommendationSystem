import re


def extract_attributes(text: str) -> dict:
    features = {}

    patterns = {
        "fragility_score": r"fragility[:\s]+([\d.]+)",
        "sustainability_priority": r"sustainability[:\s]+([\d.]+)",
        "durability_requirement": r"durability[:\s]+([\d.]+)",
        "max_packaging_cost": r"max packaging cost[:\s]+([\d.]+)",
        "material_cost": r"material cost[:\s]+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            features[key] = float(match.group(1))

    return features
