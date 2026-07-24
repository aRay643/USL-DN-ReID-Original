import os
import os.path as osp
import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple, Union


TrainItem = Tuple[str, int]
EvalItem = Tuple[str, int, int]


class SYSUMM01DatasetManager:
    """Read SYSU-MM01 directly from its official directory layout.

    The current project trains two independent domains with ``(path, pid)``
    records, while the SYSU evaluation protocol additionally needs camera ids.
    ``modal=1`` is visible light and ``modal=2`` is infrared throughout the
    SYSU-specific training and evaluation path.
    """

    dataset_dir = "SYSU-MM01"
    visible_cams = ("cam1", "cam2", "cam4", "cam5")
    infrared_cams = ("cam3", "cam6")
    indoor_gallery_cams = ("cam1", "cam2")
    camera_ids = {f"cam{camera_id}": camera_id for camera_id in range(1, 7)}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Union[str, Sequence[str]]):
        if isinstance(root, (list, tuple)):
            root = root[0]
        root = osp.expanduser(str(root))
        direct_root = osp.join(root, "exp")
        self.data_root = root if osp.isdir(direct_root) else osp.join(root, self.dataset_dir)
        self.exp_dir = osp.join(self.data_root, "exp")

        if not osp.isdir(self.data_root):
            raise RuntimeError(f"SYSU-MM01 is not available at: {self.data_root}")
        if not osp.isdir(self.exp_dir):
            raise RuntimeError(f"SYSU-MM01 split directory is missing: {self.exp_dir}")

        train_ids = sorted(set(self._load_ids("train_id") + self._load_ids("val_id")))
        test_ids = sorted(set(self._load_ids("test_id")))
        train_pid2label = {pid: label for label, pid in enumerate(train_ids)}

        self.train_visible = self._collect_train_images(
            train_ids, self.visible_cams, train_pid2label
        )
        self.train_infrared = self._collect_train_images(
            train_ids, self.infrared_cams, train_pid2label
        )

        self.eval_query_infrared = self._collect_eval_images(test_ids, self.infrared_cams)
        self.eval_gallery_visible_all = self._collect_eval_images(test_ids, self.visible_cams)
        self.eval_gallery_visible_indoor = self._collect_eval_images(
            test_ids, self.indoor_gallery_cams
        )

        self.num_ids_train_visible = self._count_ids(self.train_visible)
        self.num_ids_train_infrared = self._count_ids(self.train_infrared)
        self.num_ids_test = len(test_ids)
        self.statistics()

    def _load_ids(self, split_name: str) -> List[int]:
        split_path = osp.join(self.exp_dir, f"{split_name}.txt")
        if not osp.isfile(split_path):
            raise FileNotFoundError(f"SYSU-MM01 split file is missing: {split_path}")
        with open(split_path, "r", encoding="utf-8", errors="ignore") as split_file:
            ids = [int(token) for token in re.findall(r"\d+", split_file.read())]
        if not ids:
            raise ValueError(f"No person ids found in SYSU-MM01 split: {split_path}")
        return ids

    def _iter_images(self, pid: int, cameras: Iterable[str]):
        pid_dir_name = f"{pid:04d}"
        for camera in cameras:
            image_dir = osp.join(self.data_root, camera, pid_dir_name)
            if not osp.isdir(image_dir):
                continue
            for image_name in sorted(os.listdir(image_dir)):
                image_path = osp.join(image_dir, image_name)
                if not osp.isfile(image_path):
                    continue
                if osp.splitext(image_name)[1].lower() not in self.image_extensions:
                    continue
                yield image_path, camera

    def _collect_train_images(
        self,
        ids: Iterable[int],
        cameras: Iterable[str],
        pid2label: Dict[int, int],
    ) -> List[TrainItem]:
        dataset: List[TrainItem] = []
        for pid in ids:
            for image_path, _ in self._iter_images(pid, cameras):
                dataset.append((image_path, pid2label[pid]))
        return dataset

    def _collect_eval_images(
        self, ids: Iterable[int], cameras: Iterable[str]
    ) -> List[EvalItem]:
        dataset: List[EvalItem] = []
        for pid in ids:
            for image_path, camera in self._iter_images(pid, cameras):
                dataset.append((image_path, pid, self.camera_ids[camera]))
        return dataset

    def build_eval_sets(self, mode: str = "all", trial: int = 0):
        """Build one official IR-query/RGB-gallery single-shot trial."""
        mode = str(mode).lower()
        if mode == "all":
            gallery_pool = self.eval_gallery_visible_all
        elif mode == "indoor":
            gallery_pool = self.eval_gallery_visible_indoor
        else:
            raise ValueError(
                f"SYSU-MM01 evaluation mode must be 'all' or 'indoor', got: {mode}"
            )

        grouped: Dict[Tuple[int, int], List[EvalItem]] = {}
        for item in gallery_pool:
            _, pid, camera_id = item
            grouped.setdefault((pid, camera_id), []).append(item)

        rng = random.Random(int(trial))
        gallery = [rng.choice(sorted(grouped[key])) for key in sorted(grouped)]
        return list(self.eval_query_infrared), gallery

    @staticmethod
    def _count_ids(dataset) -> int:
        return len({item[1] for item in dataset})

    def statistics(self):
        print(f"Dataset {self.dataset_dir} statistics:")
        print("  ----------------------------------------")
        print("  subset           | # ids | # images")
        print("  ----------------------------------------")
        print(
            "  train_visible    | {:5d} | {:8d}".format(
                self.num_ids_train_visible, len(self.train_visible)
            )
        )
        print(
            "  train_infrared   | {:5d} | {:8d}".format(
                self.num_ids_train_infrared, len(self.train_infrared)
            )
        )
        print("  ----------------------------------------")
        print(
            "  query_infrared   | {:5d} | {:8d}".format(
                self._count_ids(self.eval_query_infrared), len(self.eval_query_infrared)
            )
        )
        print(
            "  gallery_visible  | {:5d} | {:8d}".format(
                self._count_ids(self.eval_gallery_visible_all),
                len(self.eval_gallery_visible_all),
            )
        )
        print("  ----------------------------------------")
