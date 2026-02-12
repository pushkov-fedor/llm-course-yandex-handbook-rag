import json
import os

from models.dataset_item import DatasetItem


def read_eval_dataset(answer_only: bool = True) -> list[DatasetItem]:
    path = os.path.join(os.path.dirname(__file__), "../eval_dataset.jsonl")
    dataset_items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                dataset_items.append(DatasetItem(**data))

    if answer_only:
        dataset_items = [item for item in dataset_items if item.should_answer]

    return dataset_items