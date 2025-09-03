# Cassify - Классификация изображений одежды

Проект для классификации изображений одежды с использованием ResNet50 и PyTorch.

## Структура проекта

```
cassify/
├── bottoms/                    # Папка с изображениями одежды
│   ├── acne_studios/
│   ├── carhartt/
│   ├── kapital/
│   ├── levis/
│   ├── rick_owens/
│   └── saint_laurent/
├── checkpoints/                # Сохранённые модели
│   └── best.pt
├── dataset.py                  # Класс для работы с данными
├── model.py                    # Архитектура модели ResNet50
├── train.py                    # Скрипт обучения
├── validate.py                 # Скрипт валидации
├── train_out.csv              # Train данные
└── test_out.csv               # Test данные
```

## Установка зависимостей

```bash
pip install torch torchvision pandas scikit-learn tqdm pillow numpy
```

## Использование

### 1. Обучение модели

```bash
# Обучение с train и val данными
python train.py --train_csv train_out.csv --val_csv test_out.csv --epochs 20 --images_root bottoms

# Обучение только с train данными (без валидации)
python train.py --train_csv train_out.csv --epochs 20 --images_root bottoms

# Обучение с дополнительными параметрами
python train.py --train_csv train_out.csv --val_csv test_out.csv \
    --epochs 50 --batch_size 64 --lr 0.001 --freeze_backbone
```

**Параметры обучения:**
- `--train_csv` - путь к train CSV файлу (по умолчанию: train_out.csv)
- `--val_csv` - путь к val CSV файлу (опционально)
- `--images_root` - корневая папка с изображениями (по умолчанию: .)
- `--epochs` - количество эпох (по умолчанию: 10)
- `--batch_size` - размер батча (по умолчанию: 32)
- `--lr` - скорость обучения (по умолчанию: 3e-4)
- `--freeze_backbone` - заморозить backbone ResNet
- `--early_stop` - ранняя остановка после N эпох без улучшения

### 2. Валидация модели

```bash
# Базовая валидация
python validate.py --checkpoint checkpoints/best.pt --csv test_out.csv --images_root bottoms

# Валидация с настройками
python validate.py --checkpoint checkpoints/best.pt --csv test_out.csv \
    --batch_size 64 --device cuda
```

**Параметры валидации:**
- `--checkpoint` - путь к чекпоинту модели (обязательно)
- `--csv` - путь к CSV файлу с данными (обязательно)
- `--images_root` - корневая папка с изображениями (по умолчанию: .)
- `--batch_size` - размер батча (по умолчанию: 32)
- `--device` - устройство: auto/cuda/cpu (по умолчанию: auto)

## Формат данных

CSV файлы должны содержать колонки:
- `image_path` - относительный путь к изображению
- `target` - метка класса (0, 1)

Пример:
```csv
image_path,target
bottoms/levis/bottoms/listproduct/image1.jpg,0
bottoms/carhartt/bottoms/listproduct/image2.jpg,1
```

## Метрики валидации

Скрипт `validate.py` выводит полный набор метрик для бинарной классификации:
- **Accuracy** - общая точность
- **Precision** - точность
- **Recall** - полнота
- **F1-score** - гармоническое среднее
- **AUC** - площадь под ROC кривой
- **Confusion Matrix** - матрица ошибок
- **Specificity/Sensitivity** - специфичность и чувствительность

## Примеры запуска

```bash
# Полный цикл: обучение + валидация
python train.py --train_csv train_out.csv --val_csv test_out.csv --epochs 20
python validate.py --checkpoint checkpoints/best.pt --csv test_out.csv

# Быстрое тестирование
python train.py --epochs 5 --batch_size 16
python validate.py --checkpoint checkpoints/best.pt --csv test_out.csv --batch_size 16
```

## Требования

- Python 3.7+
- PyTorch 1.8+
- CUDA (опционально, для GPU)
- 8GB+ RAM (рекомендуется)
