from typing import Any, Dict, List, Optional

from .base import BaseAnnotator, Sample, TaskType, VLMModel, ModelFamily
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor
import torch
from PIL import Image
import json
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import contextlib
import torch

from .model_adapters.qwen import prepare_inputs_for_vllm

def make_detection_schema(class_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "description": "List of detected objects",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "description": (
                                "Bounding box in [x1, y1, x2, y2] "
                                "image coordinates"
                            ),
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "label": {
                            "type": "string",
                            "enum": class_names,
                            "description": "Object class label"
                        },
                    },
                    "required": ["bbox", "label"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["detections"],
        "additionalProperties": False
    }



def construct_messages(image, prompt):
    return [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}
            ],
        }
    ]

class DetectionModel(VLMModel):
    def __init__(
        self,
        *args,
        hf_name: str,
        max_model_len: int = 1024,
        **kwargs
    ) -> None:
        """
        :param hf_name: имя модели HuggingFace, которое понимает vLLM
        :param max_tokens: максимум токенов в ответе
        :param temperature: температура генерации
        :param top_p: top-p сэмплинг
        :param stop: список стоп-строк (опционально)
        :param prompt_template: шаблон промпта, можно переопределить
        """
        super().__init__(*args, **kwargs)
        self.hf_name = hf_name

        self.llm = LLM(
            model=hf_name,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=max_model_len,
            gpu_memory_utilization=0.90,
            seed=0
        )

        self.processor = AutoProcessor.from_pretrained(
            hf_name
        )
        
    def kill(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    # ---- реализация абстрактного метода VLMModel ----

    def predict_batch(
        self,
        images: List[Image.Image],
        instruction: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Ожидаем, что нам передадут:
        - images: список PIL.Image
        - class_names: список строк с именами классов
        - instruction: инструкцию по разметке

        Возвращаем список dict:
        {
        "detections": [
                {
                "bbox": [x1, y1, x2, y2],
                "label": string
                }
            ]
        }
        """
        class_names: List[str] = kwargs.get("class_names")
        if not class_names:
            raise ValueError("predict_batch requires 'class_names' in kwargs")

        structured_output = StructuredOutputsParams(
            json=make_detection_schema(class_names)
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            structured_outputs=structured_output
        )

        messages = [construct_messages(image, instruction) for image in images]

        match self.model_family:
            case ModelFamily.QWEN:
                inputs = [prepare_inputs_for_vllm(message, self.processor) for message in messages]
            case ModelFamily.GEMMA:
                raise NotImplementedError()
            case ModelFamily.GLM:
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unknown model family: {self.model_family}")

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        results: List[Dict[str, Any]] = []

        for out in outputs:
            # vLLM возвращает объект с полем .outputs (list), берём первый вариант
            text = out.outputs[0].text.strip()

            result = json.loads(text)
            detections = result.get("detections", [])

            results.append(
                {
                    "detections": detections
                }
            )

        return results



class DetectionAnnotator(BaseAnnotator):
    @property
    def task_type(self) -> TaskType:
        return TaskType.DETECTION

    def __init__(
        self,
        *args,
        class_names: List[str],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_names = class_names

    def _build_model_inputs(
        self,
        batch: List[Sample],
    ) -> Dict[str, Any]:
        images = [s.image for s in batch]

        return {
            "images": images,
            "class_names": self.class_names,
        }

    def _postprocess_predictions(
        self,
        batch: List[Sample],
        raw_preds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ожидаем формат от модели:
        {
        "detections": [
                {
                "bbox": [x1, y1, x2, y2],
                "label": string
                }
            ]
        }
        """
        assert len(batch) == len(raw_preds)
        out: List[Dict[str, Any]] = []

        for sample, pred in zip(batch, raw_preds):
            out.append(
                {
                    "image_id": sample.image_id,
                    "task": "detection",
                    "prediction": {
                        "detections": pred.get("detections")
                    },
                    "raw": {
                        "model_output": pred,
                        "meta": sample.meta,
                    },
                }
            )
        return out