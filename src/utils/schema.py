from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ColumnSchema:
    dtype: Any
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None


TRAINING_DATA_SCHEMA: Dict[str, ColumnSchema] = {
    # "product_category": ColumnSchema(str, nullable=False),
    "fragility_score": ColumnSchema(float, min_value=0.0, max_value=1.0),
    "sustainability_priority": ColumnSchema(float, min_value=0.0, max_value=1.0),
    "durability_requirement": ColumnSchema(float, min_value=0.0, max_value=1.0),

    "max_packaging_cost": ColumnSchema(float, min_value=0),
    "material_cost": ColumnSchema(float, min_value=0),

    "innovation_level": ColumnSchema(int, min_value=0, max_value=5),

    "overall_sustainability_score": ColumnSchema(
        float, nullable=False, min_value=0.0, max_value=1.0
    )
}
