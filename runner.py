from pathlib import Path
from typing import Dict, List, Type

from .auto_annotate.base import BaseAnnotator, ImageFolderDataset, TaskType, VLMModel
from .auto_annotate.classification import ClassificationAnnotator
from .auto_annotate.detection import DetectionAnnotator
# from .auto_annotate.segmentation import SegmentationAnnotator

ANNOTATOR_REGISTRY: Dict[TaskType, Type[BaseAnnotator]] = {
    TaskType.CLASSIFICATION: ClassificationAnnotator,
    TaskType.DETECTION: DetectionAnnotator,
    # TaskType.SEGMENTATION: SegmentationAnnotator,
}

def auto_annotate(
    task: TaskType,
    dataset: ImageFolderDataset,
    model: VLMModel,
    instruction: str,
    class_names: List[str],
    output_path: Path,
    batch_size: int = 4,
    limit: int | None = None,
) -> None:
    annotator_cls = ANNOTATOR_REGISTRY[task]

    annotator = annotator_cls(
        dataset=dataset,
        model=model,
        instruction=instruction,
        output_path=output_path,
        batch_size=batch_size,
        class_names=class_names,
    )
    annotator.run(limit=limit)