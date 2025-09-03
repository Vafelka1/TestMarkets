import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

from dataset import ImageCsvDataset
from model import build_resnet50_v2


def parse_args():
    parser = argparse.ArgumentParser(description='Валидация модели для бинарной классификации')
    parser.add_argument('--checkpoint', type=str, required=True, help='Путь к чекпоинту модели (.pt файл)')
    parser.add_argument('--csv', type=str, required=True, help='Путь к CSV файлу с данными для валидации')
    parser.add_argument('--images_root', type=str, default='.', help='Корневая папка с изображениями')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--num_workers', type=int, default=2, help='Количество воркеров для DataLoader')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (cuda/cpu/auto)')
    return parser.parse_args()


def get_transforms():
    """Получает трансформации для валидации"""
    from torchvision import transforms
    
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return val_tfms


def load_model(checkpoint_path: str, device: torch.device):
    """Загружает модель из чекпоинта"""
    print(f"Загрузка модели из {checkpoint_path}...")
    
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Определяем количество классов из чекпоинта или используем 2 для бинарной классификации
    num_classes = 2  # Бинарная классификация
    
    # Создаём модель
    model = build_resnet50_v2(num_classes=num_classes, pretrained=False)
    model.to(device)
    
    # Загружаем веса
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"Модель загружена. Эпоха: {checkpoint.get('epoch', 'N/A')}, Val acc: {checkpoint.get('val_acc', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("Модель загружена (старый формат чекпоинта)")
    
    model.eval()
    return model


def evaluate_model(model: nn.Module, dataset, device: torch.device, batch_size: int = 32, num_workers: int = 2):
    """Оценивает модель и возвращает предсказания и метрики"""
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("Выполнение валидации...")
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Валидация"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Получаем предсказания
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)


def print_metrics(y_true, y_pred, y_prob):
    """Печатает все метрики для бинарной классификации"""
    print("\n" + "="*60)
    print("МЕТРИКИ БИНАРНОЙ КЛАССИФИКАЦИИ")
    print("="*60)
    
    # Основные метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # AUC (если есть вероятности для класса 1)
    if len(y_prob.shape) > 1 and y_prob.shape[1] >= 2:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            print(f"AUC:       {auc:.4f}")
        except ValueError:
            print("AUC:       N/A (недостаточно данных)")
    else:
        print("AUC:       N/A (нет вероятностей)")
    
    # Confusion Matrix
    print("\n" + "-"*40)
    print("CONFUSION MATRIX")
    print("-"*40)
    cm = confusion_matrix(y_true, y_pred)
    print(f"                 Predicted")
    print(f"Actual    0     1")
    print(f"    0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"    1   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Детальная статистика
    print("\n" + "-"*40)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА")
    print("-"*40)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    # Дополнительные метрики
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSpecificity (TNR): {specificity:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    
    # Classification Report
    print("\n" + "-"*40)
    print("CLASSIFICATION REPORT")
    print("-"*40)
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], zero_division=0))


def main():
    args = parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Устройство: {device}")
    print(f"Чекпоинт: {args.checkpoint}")
    print(f"CSV: {args.csv}")
    print(f"Images root: {args.images_root}")
    
    # Проверяем существование файлов
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Чекпоинт не найден: {args.checkpoint}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV файл не найден: {args.csv}")
    
    # Загружаем модель
    model = load_model(args.checkpoint, device)
    
    # Создаём датасет
    print("Создание датасета...")
    val_transform = get_transforms()
    dataset = ImageCsvDataset(args.csv, images_root=args.images_root, transform=val_transform)
    print(f"Загружено {len(dataset)} образцов")
    
    # Оцениваем модель
    predictions, targets, probabilities = evaluate_model(
        model, dataset, device, args.batch_size, args.num_workers
    )
    
    # Печатаем метрики
    print_metrics(targets, predictions, probabilities)
    
    print("\n" + "="*60)
    print("ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    print("="*60)


if __name__ == '__main__':
    main()
