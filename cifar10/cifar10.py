# coding=utf-8
"""CIFAR-100 dataset."""

import os
from etils import epath
import numpy as np
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CIFAR_IMAGE_SIZE = 32
_CIFAR_IMAGE_SHAPE = (_CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE, 3)


class Cifar10(tfds.core.GeneratorBasedBuilder):
    """CIFAR-100 dataset."""

    VERSION = tfds.core.Version("3.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="CIFAR-100 dataset with 100 classes.",
            features=tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "image": tfds.features.Image(shape=_CIFAR_IMAGE_SHAPE),
                "coarse_label": tfds.features.ClassLabel(num_classes=20),
                "label": tfds.features.ClassLabel(num_classes=100),
            }),
            supervised_keys=("image", "label"),
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
        )

    def _split_generators(self, dl_manager):
        cifar_path = dl_manager.download_and_extract(
            "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
        )
        cifar_path = os.path.join(cifar_path, "cifar-100-binary")

        # Load label names
        coarse_path = os.path.join(cifar_path, "coarse_label_names.txt")
        fine_path = os.path.join(cifar_path, "fine_label_names.txt")
        
        with epath.Path(coarse_path).open() as f:
            coarse_names = [name.strip() for name in f.read().split("\n") if name.strip()]
        with epath.Path(fine_path).open() as f:
            fine_names = [name.strip() for name in f.read().split("\n") if name.strip()]
        
        self.info.features["coarse_label"].names = coarse_names
        self.info.features["label"].names = fine_names

        return {
            tfds.Split.TRAIN: self._generate_examples(
                "train_", os.path.join(cifar_path, "train.bin")
            ),
            tfds.Split.TEST: self._generate_examples(
                "test_", os.path.join(cifar_path, "test.bin")
            ),
        }

    def _generate_examples(self, split_prefix, filepath):
        with tf.io.gfile.GFile(filepath, "rb") as f:
            data = f.read()
        
        offset = 0
        index = 0
        record_size = 2 + 3072  # 2 label bytes + 3072 image bytes
        
        while offset + record_size <= len(data):
            coarse_label = data[offset]
            fine_label = data[offset + 1]
            offset += 2
            
            img = (
                np.frombuffer(data, dtype=np.uint8, count=3072, offset=offset)
                .reshape((3, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE))
                .transpose((1, 2, 0))
            )
            offset += 3072
            
            yield index, {
                "id": f"{split_prefix}{index:05d}",
                "image": img,
                "coarse_label": int(coarse_label),
                "label": int(fine_label),
            }
            index += 1
