from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Iterator, Sequence
from PIL import Image
import json

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


@dataclass
class Sample:
    """
    Один элемент датасета для авторазметки.
    """
    image: Image.Image
    image_id: str
    meta: Dict[str, Any]


class ImageFolderDataset:
    """
    Датасет, который просто обходит папку с изображениями (рекурсивно)
    и выдает сэмплы для авторазметки.
    """

    def __init__(
        self,
        root: str | Path,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        limit: Optional[int] = None,
    ):
        self.root = Path(root)
        self.exts = tuple(e.lower() for e in exts)

        if not self.root.exists():
            raise FileNotFoundError(f"Path does not exist: {self.root}")

        # собираем список файлов
        self.paths: List[Path] = [
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.exts
        ]

        self.paths.sort()

        if limit is not None:
            self.paths = self.paths[:limit]

        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Sample]:
        for path in self.paths:
            image = Image.open(path).convert("RGB")

            rel = path.relative_to(self.root)
            image_id = str(rel)

            yield Sample(
                image=image,
                image_id=image_id,
                meta={
                    "abs_path": str(path),
                    "rel_path": str(rel),
                    "root": str(self.root),
                },
            )

class ModelFamily(str, Enum):
    QWEN = "qwen"
    GEMMA = "gemma"
    GLM = "glm"

class VLMModel(ABC):
    """
    Абстракция над моделью (vLLM / любая VLM).
    """
    def __init__(self, model_family: ModelFamily) -> None:
        self.model_family = model_family

    @abstractmethod
    def predict_batch(
        self,
        images: List[Image.Image],
        instruction: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список предсказаний в task-специфичном формате, но
        с одним контрактом: JSON-сериализуемый dict.
        """
        ...

 
class BaseAnnotator(ABC):
    """
    Базовый класс для авторазметки.
    """

    def __init__(
        self,
        dataset: ImageFolderDataset,
        model: VLMModel,
        instruction: str,
        output_path: Path,
        batch_size: int = 4,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.instruction = instruction
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        ...

    @abstractmethod
    def _build_model_inputs(
        self,
        batch: List[Sample],
    ) -> Dict[str, Any]:
        """
        Собрать входы для модели из батча сэмплов (промпты, тексты и т.п.).
        """
        ...

    @abstractmethod
    def _postprocess_predictions(
        self,
        batch: List[Sample],
        raw_preds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Преобразовать сырые предсказания модели в итоговый формат для записи.
        """
        ...

    def run(self, limit: Optional[int] = None) -> None:
        """
        Итерируется по датасету, гоняет через модель и пишет predictions.jsonl.
        """

        n_processed = 0
        buffer: List[Sample] = []

        with self.output_path.open("w", encoding="utf-8") as f:
            for sample in self.dataset:
                buffer.append(sample)

                if len(buffer) == self.batch_size:
                    self._process_batch(buffer, f)
                    n_processed += len(buffer)
                    buffer = []

                    if limit is not None and n_processed >= limit:
                        break

            # хвост
            if buffer and (limit is None or n_processed < limit):
                self._process_batch(buffer, f)

    def _process_batch(self, batch: List[Sample], file_handle) -> None:

        model_inputs = self._build_model_inputs(batch)
        raw_preds = self.model.predict_batch(
            model_inputs["images"],
            instruction=self.instruction,
            **{k: v for k, v in model_inputs.items() if k != "images"},
        )
        processed = self._postprocess_predictions(batch, raw_preds)

        for item in processed:
            file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

