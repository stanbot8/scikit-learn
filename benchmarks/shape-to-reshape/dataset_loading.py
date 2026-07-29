"""ASV benchmarks for the changed dataset-loading paths."""

from __future__ import annotations

import tempfile
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets._lfw import _fetch_lfw_pairs


class DatasetLoading:
    def setup(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        faces = root / "faces"
        person_a = faces / "Person_A"
        person_b = faces / "Person_B"
        person_a.mkdir(parents=True)
        person_b.mkdir()
        for person in (person_a, person_b):
            (person / "1.jpg").touch()
            (person / "2.jpg").touch()
        self.index = root / "pairs.txt"
        self.index.write_text(
            "2\nPerson_A\t1\t2\nPerson_A\t1\tPerson_B\t1\n",
            encoding="utf-8",
        )
        self.faces = faces
        self.image_data = np.arange(4 * 62 * 47, dtype=np.float32).reshape(
            4, 62, 47
        )
        self.loader = patch(
            "sklearn.datasets._lfw._load_imgs",
            side_effect=self._load_images,
        )
        self.loader.start()

    def _load_images(self, *args, **kwargs):
        self.loaded_image_data = self.image_data.copy()
        return self.loaded_image_data

    def teardown(self):
        self.loader.stop()
        self.temporary.cleanup()

    def time_load_digits(self):
        load_digits()

    def track_load_digits_checksum(self):
        digits = load_digits()
        return zlib.crc32(digits.images.tobytes(), zlib.crc32(digits.target.tobytes()))

    def track_load_digits_shares_memory(self):
        digits = load_digits()
        return float(np.shares_memory(digits.data, digits.images))

    def time_fetch_lfw_pairs(self):
        _fetch_lfw_pairs(self.index, self.faces)

    def track_fetch_lfw_pairs_checksum(self):
        pairs, target, _ = _fetch_lfw_pairs(self.index, self.faces)
        return zlib.crc32(pairs.tobytes(), zlib.crc32(target.tobytes()))

    def track_fetch_lfw_pairs_shares_memory(self):
        pairs, _, _ = _fetch_lfw_pairs(self.index, self.faces)
        return float(np.shares_memory(self.loaded_image_data, pairs))
