import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_resnet50_v2


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_csv', type=str, default='train_out.csv', help='Путь к train CSV файлу')
	parser.add_argument('--val_csv', type=str, help='Путь к val CSV файлу (опционально)')
	parser.add_argument('--images_root', type=str, default='.')
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--output', type=str, default='checkpoints')
	parser.add_argument('--freeze_backbone', action='store_true')
	parser.add_argument('--early_stop', type=int, default=0, help='0 выключено; N>0 — остановка после N эпох без улучшения val-accuracy')
	return parser.parse_args()


def get_transforms(img_size: int = 224):
	train_tfms = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	val_tfms = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return train_tfms, val_tfms


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
	model.eval()
	correct = 0
	total = 0
	with torch.inference_mode():
		for images, targets in tqdm(loader, desc="Валидация", leave=False):
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			logits = model(images)
			preds = logits.argmax(dim=1)
			correct += (preds == targets).sum().item()
			total += targets.numel()
	return correct / max(total, 1)


def main():
	args = parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Устройство: {device}")
	print(f"Train CSV: {args.train_csv}, Val CSV: {args.val_csv}, images_root: {args.images_root}")

	print("Создание трансформаций...")
	train_tfms, val_tfms = get_transforms(224)
	
	print("Создание DataLoader-ов...")
	train_loader, val_loader, class_counts = create_dataloaders(
		train_csv_path=args.train_csv,
		val_csv_path=args.val_csv,
		images_root=args.images_root,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		train_transform=train_tfms,
		val_transform=val_tfms,
	)
	num_classes = len(class_counts)
	val_batches = len(val_loader) if val_loader is not None else 0
	print(f"Классов: {num_classes}, train батчей: {len(train_loader)}, val батчей: {val_batches}")

	print("Создание модели...")
	model = build_resnet50_v2(num_classes=num_classes, pretrained=True)
	model.to(device)

	if args.freeze_backbone:
		print("Заморозка backbone...")
		for name, param in model.named_parameters():
			if 'fc' not in name and 'classifier' not in name:
				param.requires_grad = False

	print("Настройка оптимизатора...")
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

	best_acc = 0.0
	no_improve_epochs = 0
	Path(args.output).mkdir(parents=True, exist_ok=True)
	print(f"Начало обучения на {args.epochs} эпох, early_stop={args.early_stop}")

	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0
		train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
		for images, targets in train_pbar:
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)
			logits = model(images)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * images.size(0)
			
			# Обновляем прогресс-бар с текущим loss
			train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

		train_loss = running_loss / len(train_loader.dataset)
		
		if val_loader is not None:
			val_acc = evaluate(model, val_loader, device)
			print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")
			
			if val_acc > best_acc:
				best_acc = val_acc
				no_improve_epochs = 0
				ckpt_path = os.path.join(args.output, 'best.pt')
				torch.save({'model_state': model.state_dict(), 'val_acc': best_acc, 'epoch': epoch}, ckpt_path)
				print(f"Saved best model to {ckpt_path}")
			else:
				no_improve_epochs += 1
				if args.early_stop > 0 and no_improve_epochs >= args.early_stop:
					print(f"Early stopping: нет улучшений {no_improve_epochs} эпох подряд")
					break
		else:
			print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_acc: N/A (нет val данных)")
			# Сохраняем модель без валидации
			ckpt_path = os.path.join(args.output, 'best.pt')
			torch.save({'model_state': model.state_dict(), 'val_acc': 0.0, 'epoch': epoch}, ckpt_path)
		
		scheduler.step()

	if val_loader is not None:
		print(f"Best val acc: {best_acc:.4f}")
	else:
		print("Обучение завершено без валидации")


if __name__ == '__main__':
	main()


