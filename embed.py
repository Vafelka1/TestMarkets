import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageCsvDataset
from model import build_resnet50_for_embeddings


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, default='out_df.csv')
	parser.add_argument('--images_root', type=str, default='.')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--img_size', type=int, default=224)
	parser.add_argument('--output', type=str, default=None, help='Куда сохранить: .csv | .parquet | .npy | .pt. По умолчанию перезапись CSV')
	parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', help='Путь к обученному чекпоинту')
	return parser.parse_args()


def get_transform(img_size: int = 224):
	return transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def _try_load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> bool:
	if not ckpt_path or not os.path.exists(ckpt_path):
		print(f"Checkpoint не найден: {ckpt_path}")
		return False
	ckpt = torch.load(ckpt_path, map_location='cpu')
	state = ckpt.get('model_state', ckpt)
	missing, unexpected = model.load_state_dict(state, strict=False)
	print(f"Загрузили веса: missing={len(missing)}, unexpected={len(unexpected)}")
	return True


def _build_model_for_embeddings(checkpoint: str, device: torch.device) -> torch.nn.Module:
	# Модель, которая сразу возвращает эмбеддинг (2048) при forward
	model = build_resnet50_for_embeddings(pretrained=True).to(device)
	_ = _try_load_checkpoint_into_model(model, checkpoint)  # strict=False внутри
	model.eval()
	return model


def _extract_flatten_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
	# Для модели из build_resnet50_for_embeddings forward сразу возвращает 2048.
	with torch.no_grad():
		out = model(images)
		return out if out.dim() == 2 else out.view(out.size(0), -1)


def main():
	args = parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(f"Device: {device}")
	print(f"CSV: {args.csv}, images_root: {args.images_root}")

	df = pd.read_csv(args.csv)
	if 'image_path' not in df.columns:
		raise ValueError("CSV должен содержать колонку 'image_path'")

	transform = get_transform(args.img_size)
	dataset = ImageCsvDataset(args.csv, images_root=args.images_root, transform=transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

	# Собираем модель и грузим обученный чекпоинт, затем извлекаем фичи из предпоследнего слоя
	model = _build_model_for_embeddings(args.checkpoint, device)
	model.eval()

	embeddings = []
	with torch.inference_mode():
		for images, _ in tqdm(loader, desc='Embeddings'):
			images = images.to(device, non_blocking=True)
			feats = _extract_flatten_features(model, images)
			embeddings.extend(feats.cpu().numpy())

	# Финализация и сохранение
	df = df.reset_index(drop=True)
	if len(df) != len(embeddings):
		raise RuntimeError(f"Размеры не совпадают: df={len(df)} vs embeds={len(embeddings)}")

	embeds_arr = np.stack(embeddings)
	print(f"Embeddings shape: {embeds_arr.shape}")

	output = args.output or args.csv
	Path(os.path.dirname(output) or '.').mkdir(parents=True, exist_ok=True)

	lower = output.lower()
	if lower.endswith('.npy') or lower.endswith('.npz'):
		# Сохраняем отдельно: index (N,1) и embeds (N,2048) в .npz
		idx = df.index.values.reshape(-1, 1).astype(np.int64)
		out_path = output[:-4] + '.npz' if lower.endswith('.npy') else output
		np.savez(out_path, index=idx, embeds=embeds_arr.astype(np.float32))
		print(f"Saved npz with separate arrays to {out_path}: index {idx.shape}, embeds {embeds_arr.shape}")
	elif lower.endswith('.pt'):
		# Отдельно: dict с двумя тензорами
		idx = torch.from_numpy(df.index.values.reshape(-1, 1).astype(np.int64))
		emb = torch.from_numpy(embeds_arr.astype(np.float32))
		torch.save({'index': idx, 'embeds': emb}, output)
		print(f"Saved PT dict to {output}: index {tuple(idx.shape)}, embeds {tuple(emb.shape)}")
	elif lower.endswith('.parquet'):
		try:
			import json
			df_out = df.copy()
			df_out['embed'] = [embeds_arr[i].tolist() for i in range(len(df_out))]
			df_out.to_parquet(output, index=False)
			print(f"Saved DataFrame with embeddings to {output}")
		except Exception as e:
			print(f"Parquet save failed ({e}). Falling back to CSV JSON-encoded.")
			import json
			df_out = df.copy()
			df_out['embed'] = [json.dumps(embeds_arr[i].tolist()) for i in range(len(df_out))]
			csv_fallback = output[:-8] + '.csv'
			df_out.to_csv(csv_fallback, index=False)
			print(f"Saved CSV to {csv_fallback}")
	else:
		# CSV: сериализуем список как JSON-строку
		import json
		df_out = df.copy()
		df_out['embed'] = [json.dumps(embeds_arr[i].tolist()) for i in range(len(df_out))]
		df_out.to_csv(output, index=False)
		print(f"Saved CSV with JSON-encoded embeddings to {output}")


if __name__ == '__main__':
	main()


