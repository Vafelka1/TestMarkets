
import os
from typing import Optional, Tuple, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ImageCsvDataset(Dataset):

	def __init__(self, csv_path: str, images_root: Optional[str] = None, transform=None):
		self.dataframe = pd.read_csv(csv_path)
		if 'image_path' not in self.dataframe.columns or 'target' not in self.dataframe.columns:
			raise ValueError("CSV должен содержать колонки 'image_path' и 'target'")
		self.images_root = images_root
		self.transform = transform

	def __len__(self) -> int:
		return len(self.dataframe)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		row = self.dataframe.iloc[index]
		img_rel_path: str = str(row['image_path'])
		label = int(row['target'])
		# Нормализация путей: избегаем дублирования корня и смешанных слэшей
		img_path = self._resolve_image_path(img_rel_path)
		with Image.open(img_path) as img:
			img = img.convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
		return img, torch.tensor(label, dtype=torch.long)

	def _resolve_image_path(self, img_rel_path: str) -> str:
		rel = img_rel_path.replace('\\', '/')
		candidate = os.path.normpath(rel)
		root = os.path.normpath(self.images_root) if self.images_root else None
		candidates = []
		# 1) root + candidate
		if root:
			candidates.append(os.path.normpath(os.path.join(root, candidate)))
		# 2) candidate как есть (от CWD)
		candidates.append(candidate)
		# 3) если в пути есть 'bottoms', пробуем подпуть начиная с 'bottoms'
		parts = rel.split('/')
		if 'bottoms' in parts:
			idx = parts.index('bottoms')
			sub = os.path.normpath(os.path.join(*parts[idx:]))
			if root:
				candidates.append(os.path.normpath(os.path.join(root, sub)))
			candidates.append(sub)
		# Возвращаем первый существующий
		for p in candidates:
			if os.path.exists(p):
				return p
		# Фоллбек: первый кандидат
		print(f"WARNING: Файл не найден. Исходный путь: {img_rel_path}, пробовали: {candidates}")
		return candidates[0]


def create_dataloaders(
	train_csv_path: str,
	val_csv_path: Optional[str] = None,
	images_root: Optional[str] = None,
	batch_size: int = 32,
	num_workers: int = 2,
	val_size: float = 0.2,
	random_state: int = 42,
	train_transform=None,
	val_transform=None,
) -> Tuple[DataLoader, DataLoader, List[int]]:
	"""Создаёт DataLoader-ы из train и val CSV файлов.

	Возвращает (train_loader, val_loader, class_counts)
	"""
	train_df = pd.read_csv(train_csv_path)
	if 'image_path' not in train_df.columns or 'target' not in train_df.columns:
		raise ValueError("Train CSV должен содержать колонки 'image_path' и 'target'")

	# Создаём train датасет
	train_dataset = ImageCsvDataset(train_csv_path, images_root=images_root, transform=train_transform)
	
	# Создаём val датасет если указан путь
	if val_csv_path is not None:
		val_df = pd.read_csv(val_csv_path)
		if 'image_path' not in val_df.columns or 'target' not in val_df.columns:
			raise ValueError("Val CSV должен содержать колонки 'image_path' и 'target'")
		val_dataset = ImageCsvDataset(val_csv_path, images_root=images_root, transform=val_transform)
	else:
		# Если val_csv не указан, создаём пустой датасет
		val_dataset = None

	class_counts = train_df['target'].value_counts().sort_index().tolist()

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	
	if val_dataset is not None:
		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=torch.cuda.is_available(),
		)
	else:
		val_loader = None

	return train_loader, val_loader, class_counts
