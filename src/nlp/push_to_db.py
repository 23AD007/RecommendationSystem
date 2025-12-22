from sqlalchemy import create_engine
from urllib.parse import quote_plus
from pdfminer.high_level import extract_text
import os
import pandas as pd

from src.nlp.extract_features import extract_attributes
from src.nlp.extract_text import extract_text_from_pdf


def ingest_document(pdf_path: str):
    text = extract_text(pdf_path)
    features = extract_attributes(text)

    confidence = len(features) / 5.0

    record = {
        "source_document": os.path.basename(pdf_path),
        "extracted_text": text[:2000],
        **features,
        "confidence_score": confidence,
        "extraction_method": "rule_based"
    }

    engine = create_engine(
        f"postgresql://postgres:{quote_plus(os.getenv('DB_PASSWORD'))}@localhost:5432/ecopackai"
    )

    pd.DataFrame([record]).to_sql(
        "nlp_extracted_data",
        engine,
        if_exists="append",
        index=False
    )

    print("Document ingested successfully")
    print("Confidence score:", confidence)


if __name__ == "__main__":
    ingest_document("data/documents/supplier_report_2025.pdf")
