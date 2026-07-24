import os
import tempfile
import unittest

import numpy as np

from datasets.SYSU_MM01 import SYSUMM01DatasetManager
from utils.sysu_metrics import eval_sysu


class SYSUDatasetManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = os.path.join(self.temp_dir.name, "SYSU-MM01")
        exp_dir = os.path.join(self.dataset_root, "exp")
        os.makedirs(exp_dir)
        self._write_split(exp_dir, "train_id.txt", "1")
        self._write_split(exp_dir, "val_id.txt", "2")
        self._write_split(exp_dir, "test_id.txt", "3,4")

        for pid in (1, 2):
            for camera_id in range(1, 7):
                self._make_images(camera_id, pid, count=1)
        for pid in (3, 4):
            for camera_id in range(1, 7):
                self._make_images(camera_id, pid, count=4)

        self.dataset = SYSUMM01DatasetManager(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write_split(exp_dir, filename, content):
        with open(os.path.join(exp_dir, filename), "w", encoding="utf-8") as output:
            output.write(content)

    def _make_images(self, camera_id, pid, count):
        image_dir = os.path.join(
            self.dataset_root, f"cam{camera_id}", f"{pid:04d}"
        )
        os.makedirs(image_dir, exist_ok=True)
        for index in range(count):
            with open(
                os.path.join(image_dir, f"{index:04d}.jpg"), "wb"
            ) as image_file:
                image_file.write(b"test")

    def test_training_uses_train_and_val_with_modality_cameras(self):
        self.assertEqual(len(self.dataset.train_visible), 8)
        self.assertEqual(len(self.dataset.train_infrared), 4)
        self.assertEqual({pid for _, pid in self.dataset.train_visible}, {0, 1})
        self.assertEqual(
            {os.path.basename(os.path.dirname(os.path.dirname(path)))
             for path, _ in self.dataset.train_visible},
            {"cam1", "cam2", "cam4", "cam5"},
        )
        self.assertEqual(
            {os.path.basename(os.path.dirname(os.path.dirname(path)))
             for path, _ in self.dataset.train_infrared},
            {"cam3", "cam6"},
        )

    def test_all_and_indoor_single_shot_trials(self):
        query, gallery_all = self.dataset.build_eval_sets("all", trial=0)
        _, gallery_all_repeat = self.dataset.build_eval_sets("all", trial=0)
        _, gallery_indoor = self.dataset.build_eval_sets("indoor", trial=0)

        self.assertEqual(gallery_all, gallery_all_repeat)
        self.assertEqual(len(query), 16)
        self.assertEqual(len(gallery_all), 8)
        self.assertEqual(len(gallery_indoor), 4)
        self.assertEqual({camera for _, _, camera in query}, {3, 6})
        self.assertEqual({camera for _, _, camera in gallery_all}, {1, 2, 4, 5})
        self.assertEqual({camera for _, _, camera in gallery_indoor}, {1, 2})

        trial_galleries = {
            tuple(item[0] for item in self.dataset.build_eval_sets("all", trial)[1])
            for trial in range(10)
        }
        self.assertGreater(len(trial_galleries), 1)


class SYSUMetricTest(unittest.TestCase):
    def test_cam3_query_filters_cam2_gallery(self):
        distmat = np.asarray([[0.0, 0.1, 0.2]], dtype=np.float32)
        cmc, mean_ap, minp = eval_sysu(
            distmat,
            q_pids=np.asarray([1]),
            g_pids=np.asarray([1, 2, 1]),
            q_camids=np.asarray([3]),
            g_camids=np.asarray([2, 1, 1]),
            max_rank=3,
        )
        np.testing.assert_array_equal(cmc, np.asarray([0.0, 1.0, 1.0]))
        self.assertAlmostEqual(mean_ap, 0.5)
        self.assertAlmostEqual(minp, 0.5)

    def test_cmc_deduplicates_ranked_identities(self):
        distmat = np.asarray([[0.0, 0.1, 0.2]], dtype=np.float32)
        cmc, mean_ap, minp = eval_sysu(
            distmat,
            q_pids=np.asarray([1]),
            g_pids=np.asarray([2, 2, 1]),
            q_camids=np.asarray([6]),
            g_camids=np.asarray([1, 4, 5]),
            max_rank=3,
        )
        np.testing.assert_array_equal(cmc, np.asarray([0.0, 1.0, 1.0]))
        self.assertAlmostEqual(mean_ap, 1.0 / 3.0)
        self.assertAlmostEqual(minp, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
