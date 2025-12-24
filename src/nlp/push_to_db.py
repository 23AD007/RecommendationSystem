import pandas as pd
import mlflow
from datetime import datetime

from src.ecopackdb.db_connect import get_engine
from src.nlp.extract_text import extract_text
from src.nlp.extract_features import extract_attributes
from src.utils.schema_validation import validate_schema, SchemaValidationError


def ingest_document(pdf_path: str):
    mlflow.set_experiment("NLP_Schema_Validation")

    with mlflow.start_run(run_name="nlp_ingestion"):
        try:
            # 1. Extract text
            text = extract_text(pdf_path)

            # 2. Extract features (dict)
            features = extract_attributes(text)

            # ✅ FIX: dict → DataFrame BEFORE validation
            df = pd.DataFrame([features])

            # 3. Validate schema (expects DataFrame)
            validate_schema(df)

            # 4. Add metadata (to DataFrame, not dict)
            df["data_source"] = "nlp"
            df["ingested_at"] = datetime.utcnow()

            # 5. Insert into DB
            engine = get_engine()
            df.to_sql(
                "nlp_extracted_data",
                engine,
                if_exists="append",
                index=False
            )

            # 6. Log success
            mlflow.log_param("status", "success")
            mlflow.log_param("source_file", pdf_path)

            print("✅ NLP document validated and ingested successfully")

        except SchemaValidationError as e:
            # Schema failure logging
            mlflow.log_param("status", "schema_failed")
            mlflow.log_param("source_file", pdf_path)

            mlflow.log_text(
                str(e),
                artifact_file="schema_validation_error.txt"
            )

            print("❌ Schema validation failed")
            print(str(e))


if __name__ == "__main__":
    ingest_document("data/documents/supplier_report_2025.pdf")
