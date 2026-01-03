# vlm-markup-anything

Небольшой пайплайн для авторазметки картинок с помощью VLM (через vLLM) для задач классификации, детекции, сегментации

На вход:  
- папка с изображениями (без разметки)

На выход:  
- `predictions.jsonl` — по строке на каждую картинку, с предсказанным классом (`cat` / `dog`, или любые другие)

В примере ниже используется **Qwen3-VL** через **vLLM** для задачи классификации.

---

## Требования

Минимальный рабочий набор библиотек (тестировалось с такими версиями):

```txt
vllm==0.11.0
pillow==11.0.0
torch==2.8.0
qwen-vl-utils==0.0.14
transformers==4.57.1
```

## Подготовка датасета

Ожидается простая папка с картинками, например:

```
datasets/
└── cats-vs-dogs/
    ├── img_0001.jpg
    ├── img_0002.png
    ├── ...
```

## Пример использования

Перед использованием: 

1) git clone в папку vlm_markup_anything
2) устанавливаем зависимости
3) создаем main.py по типу такого и запускаем, прописывам пути до файлов, выбираем модель.

```python
from pathlib import Path
from vlm_markup_anything.auto_annotate.base import TaskType, ImageFolderDataset
from vlm_markup_anything.auto_annotate.classification import ClassificationModel
from vlm_markup_anything.runner import auto_annotate

instruction = """You need to predict, who is on the picture: cat or dog?"""

if __name__ == "__main__":

    dataset = ImageFolderDataset(root="datasets/cats-vs-dogs")
    model = ClassificationModel(hf_name="Qwen/Qwen3-VL-4B-Instruct-FP8")

    auto_annotate(
        task=TaskType.CLASSIFICATION,
        dataset=dataset,
        model=model,
        instruction=instruction,
        class_names=["cat", "dog"],
        output_path=Path("outputs/cats-vs-dogs/predictions.jsonl"),
        batch_size=4,
    )
```

## Результат

```json
{"image_id": "cat0.png", "task": "classification", "prediction": {"label": "cat"}, "raw": {"model_output": {"label": "cat"}, "meta": {"abs_path": "datasets/cats-vs-dogs/cat0.png", "rel_path": "cat0.png", "root": "datasets/cats-vs-dogs"}}}
{"image_id": "dog0.png", "task": "classification", "prediction": {"label": "dog"}, "raw": {"model_output": {"label": "dog"}, "meta": {"abs_path": "datasets/cats-vs-dogs/dog0.png", "rel_path": "dog0.png", "root": "datasets/cats-vs-dogs"}}}
...
```

