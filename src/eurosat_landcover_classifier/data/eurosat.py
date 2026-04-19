from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


@dataclass
class DatasetBundle:
    class_names: List[str]
    train_samples: List[Sample]
    val_samples: List[Sample]
    test_samples: List[Sample]
    mean: np.ndarray
    std: np.ndarray
    image_shape: Tuple[int, int, int]

    @property
    def input_dim(self) -> int:
        height, width, channels = self.image_shape
        return height * width * channels


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def scan_eurosat(data_dir: str) -> Tuple[List[str], List[Sample], Tuple[int, int, int]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    class_dirs = sorted([item for item in root.iterdir() if item.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under {data_dir}")

    class_names = [item.name for item in class_dirs]
    samples: List[Sample] = []
    first_image_shape = None

    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted([path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
        for image_path in image_paths:
            samples.append(Sample(str(image_path), label))
        if first_image_shape is None and image_paths:
            first_image_shape = load_image(str(image_paths[0])).shape

    if first_image_shape is None:
        raise ValueError("Dataset contains no readable images.")

    return class_names, samples, first_image_shape


def stratified_split(
    samples: Sequence[Sample],
    num_classes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    buckets: List[List[Sample]] = [[] for _ in range(num_classes)]
    for sample in samples:
        buckets[sample.label].append(sample)

    rng = np.random.default_rng(seed)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for bucket in buckets:
        bucket = list(bucket)
        rng.shuffle(bucket)
        num_samples = len(bucket)
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val
        if min(num_train, num_val, num_test) <= 0:
            raise ValueError("Split ratios produced an empty split for at least one class.")

        train_samples.extend(bucket[:num_train])
        val_samples.extend(bucket[num_train : num_train + num_val])
        test_samples.extend(bucket[num_train + num_val :])

    return train_samples, val_samples, test_samples


def compute_channel_stats(samples: Sequence[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for sample in samples:
        image = load_image(sample.path)
        pixels = image.reshape(-1, 3).astype(np.float64)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels * pixels).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - mean ** 2
    std = np.sqrt(np.maximum(variance, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)


def prepare_eurosat(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    class_names, samples, image_shape = scan_eurosat(data_dir)
    train_samples, val_samples, test_samples = stratified_split(
        samples=samples,
        num_classes=len(class_names),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    mean, std = compute_channel_stats(train_samples)
    return DatasetBundle(
        class_names=class_names,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        mean=mean,
        std=std,
        image_shape=image_shape,
    )


class DataLoader:
    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        shuffle: bool = False,
        flatten: bool = True,
        return_images: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = list(samples)
        self.batch_size = batch_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
        self.shuffle = shuffle
        self.flatten = flatten
        self.return_images = return_images
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            images = []
            labels = []

            for index in batch_indices:
                sample = self.samples[index]
                images.append(load_image(sample.path))
                labels.append(sample.label)

            batch = np.stack(images, axis=0).astype(np.float32)
            raw_images = (batch * 255.0).clip(0, 255).astype(np.uint8)
            batch = (batch - self.mean) / self.std

            if self.flatten:
                batch = batch.reshape(batch.shape[0], -1)

            labels_array = np.asarray(labels, dtype=np.int64)
            batch = batch.astype(np.float32)

            if self.return_images:
                yield batch, labels_array, raw_images
            else:
                yield batch, labels_array
