import torch.nn as nn


def build_resnet50_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
	"""Создаёт ResNet50 v2 (через timm, с резервом на torchvision)."""
	try:
		import timm
		model = timm.create_model(
			"resnetv2_50", pretrained=pretrained, num_classes=num_classes
		)
		return model
	except Exception:
		# Резерв через torchvision: используем классическую resnet50
		from torchvision.models import resnet50, ResNet50_Weights
		weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
		model = resnet50(weights=weights)
		# Меняем финальный классификатор под число классов
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)
		return model



def build_resnet50_for_embeddings(pretrained: bool = True) -> nn.Module:
	"""Модель, сразу возвращающая эмбеддинги (2048) при обычном forward.

	- timm: resnetv2_50 с num_classes=0 (выход уже pooled+flatten)
	- torchvision: resnet50 с fc=Identity (выход — 2048 после avgpool)
	"""
	try:
		import timm
		return timm.create_model("resnetv2_50", pretrained=pretrained, num_classes=0)
	except Exception:
		from torchvision.models import resnet50, ResNet50_Weights
		weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
		m = resnet50(weights=weights)
		m.fc = nn.Identity()
		return m

