from typing import Any, Dict, List, Optional

from .base import BaseAnnotator, Sample, TaskType, VLMModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
import json


def make_classification_schema(class_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": class_names, 
                "description": "Predicted class label. Must be one of enum values."
            }
        },
        "required": ["label"],
        "additionalProperties": False
    }

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data
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

class ClassificationModel(VLMModel):
    def __init__(
        self,
        hf_name: str
    ) -> None:
        """
        :param hf_name: имя модели HuggingFace, которое понимает vLLM
        :param max_tokens: максимум токенов в ответе
        :param temperature: температура генерации
        :param top_p: top-p сэмплинг
        :param stop: список стоп-строк (опционально)
        :param prompt_template: шаблон промпта, можно переопределить
        """
        self.hf_name = hf_name

        self.llm = LLM(
            model=hf_name,
            mm_encoder_tp_mode="data",
            enable_prefix_caching=True,
            # tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=1024,
            # gpu_memory_utilization=0.95,
            enforce_eager=False,
            seed=0
        )

        self.processor = AutoProcessor.from_pretrained(
            hf_name
        )

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
          "label": str
        }
        """
        class_names: List[str] = kwargs.get("class_names")
        if not class_names:
            raise ValueError("predict_batch requires 'class_names' in kwargs")

        structured_output = StructuredOutputsParams(
            json=make_classification_schema(class_names)
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            structured_outputs=structured_output
        )

        messages = [construct_messages(image, instruction) for image in images]
        inputs = [prepare_inputs_for_vllm(message, self.processor) for message in messages]

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        results: List[Dict[str, Any]] = []

        for out in outputs:
            # vLLM возвращает объект с полем .outputs (list), берём первый вариант
            text = out.outputs[0].text.strip()

            result = json.loads(text)
            label = result.get("label", "")

            results.append(
                {
                    "label": label
                }
            )

        return results



class ClassificationAnnotator(BaseAnnotator):
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

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
          "label": str
        }
        """
        assert len(batch) == len(raw_preds)
        out: List[Dict[str, Any]] = []

        for sample, pred in zip(batch, raw_preds):
            out.append(
                {
                    "image_id": sample.image_id,
                    "task": "classification",
                    "prediction": {
                        "label": pred.get("label")
                    },
                    "raw": {
                        "model_output": pred,
                        "meta": sample.meta,
                    },
                }
            )
        return out