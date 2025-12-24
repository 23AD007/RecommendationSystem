import re


def extract_attributes(text: str) -> dict:
    text = text.lower()

    return {
        "eco_pressure": 0.2 if "plastic" in text else 0.5,
        "cost_efficiency": 0.8 if "low cost" in text else 0.6,
        "durability_pressure": 0.3 if "durable" in text else 0.6,

        # ✅ ADD THESE
        "innovation_level": 0.7 if "biodegradable" in text else 0.4,

        # NLP cannot truly "recommend" → default / placeholder
        "recommended": False
    }

if __name__ == "__main__":
    sample_text = """
    Product Category: Electronics
    Fragility: 0.7
    Sustainability: 0.8
    Durability: 0.75
    Max Packaging Cost: 120
    Material Cost: 60
    Innovation Level: Improved
    """

    features = extract_attributes(sample_text)
    print("Extracted Features:")
    for k, v in features.items():
        print(f"{k}: {v}")
