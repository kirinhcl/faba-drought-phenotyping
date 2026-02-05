"""DAG (Drought onset days After Germination) class definitions.

Maps the 13 unique DAG values to class indices for classification.
"""

from __future__ import annotations

DAG_VALUES: list[int] = [13, 14, 17, 19, 20, 21, 24, 27, 28, 29, 31, 33, 35, 38]

NUM_DAG_CLASSES: int = len(DAG_VALUES)

DAG_TO_CLASS: dict[int, int] = {dag: idx for idx, dag in enumerate(DAG_VALUES)}

CLASS_TO_DAG: dict[int, int] = {idx: dag for idx, dag in enumerate(DAG_VALUES)}


def dag_to_class(dag_value: float) -> int:
    """Convert DAG value to class index. Returns -1 for invalid/missing values."""
    dag_int = int(round(dag_value))
    return DAG_TO_CLASS.get(dag_int, -1)


def class_to_dag(class_idx: int) -> int:
    """Convert class index to DAG value."""
    return CLASS_TO_DAG[class_idx]
