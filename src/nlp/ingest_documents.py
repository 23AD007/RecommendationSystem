from sqlalchemy import create_engine
from urllib.parse import quote_plus
from pdfminer.high_level import extract_text
import pandas as pd
import os

# -----------------------------
# Database connection
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", "ads@1234"))  # SAFE
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# NLP helpers
# -----------------------------
def extract_attributes(text):
    """
    Simple rule-based NLP extraction
    """
    text = text.lower()

    def find_score(keyword):
        import re
        match = re.search(rf"{keyword}\s*[:=]\s*(\d+(\.\d+)?)", text)
        return float(match.group(1)) if match else None

    return {
        "product_category": "electronics" if "electronics" in text else None,
        "fragility_score": find_score("fragility"),
        "sustainability_priority": find_score("sustainability"),
        "durability_requirement": find_score("durability"),
        "max_packaging_cost": find_score("cost"),
        "material_cost": find_score("material cost"),
        "innovation_level": 2  # default: Improved
    }


def estimate_confidence(features):
    filled = sum(v is not None for v in features.values())
    return filled / len(features)


# -----------------------------
# Main ingestion logic
# -----------------------------
def ingest_document(pdf_path):
    text = extract_text(pdf_path)

    features = extract_attributes(text)
    confidence = estimate_confidence(features)

    record = {
        "source_document": os.path.basename(pdf_path),
        "extracted_text": text[:2000],
        **features,
        "confidence_score": confidence,
        "extraction_method": "rule_based_nlp"
    }

    df = pd.DataFrame([record])

    df.to_sql(
        "nlp_extracted_data",
        engine,
        if_exists="append",
        index=False
    )

    print("Document ingested successfully")
    print("Confidence score:", round(confidence, 2))


if __name__ == "__main__":
    ingest_document("data/documents/supplier_report_2025.pdf")
