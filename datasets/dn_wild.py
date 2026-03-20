from pathlib import Path
from typing import List, Tuple, Union
import os.path as osp

def _parse_pid_from_rel(rel: str) -> int:
    """
    从类似 '00834/000.jpg' 解析 PID=834
    """
    pid_str = Path(rel).parts[0]
    try:
        return int(pid_str)
    except ValueError:
        return -1

class DNwildDatasetManager:
    """
    DN-wild 数据集管理器
    一次性加载四个 split：train_day / train_night / test_day / test_night
    每个 split 存为列表: [(绝对路径, pid), ...]
    """

    dataset_dir = 'dnwild'
    def __init__(self, root: Union[str, Path]):
        self.data_root = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.data_root, "train_test_split")

        assert osp.exists(self.data_root), f"The dataset path does not exist: {self.data_root}"
        assert osp.exists(self.split_dir), f"The split folder does not exist: {self.split_dir}"

        # 四个列表
        self.train_day:  List[Tuple[Path, int]] = []
        self.train_night: List[Tuple[Path, int]] = []
        self.query_day: List[Tuple[Path, int]] = []
        self.query_night: List[Tuple[Path, int]] = []
        self.test_day:  List[Tuple[Path, int]] = []
        self.test_night: List[Tuple[Path, int]] = []

        # 加载数据
        self._load_split("day",   "train", self.train_day)
        self._load_split("night", "train", self.train_night)
        self._load_split("day", "query", self.query_day)
        self._load_split("night", "query", self.query_night)
        self._load_split("day",   "test",  self.test_day)
        self._load_split("night", "test",  self.test_night)

        # 统计 id 数量
        self.num_ids_train_day = self._count_ids(self.train_day)
        self.num_ids_train_night = self._count_ids(self.train_night)
        self.num_ids_query_day = self._count_ids(self.query_day)
        self.num_ids_query_night = self._count_ids(self.query_night)
        self.num_ids_test_day = self._count_ids(self.test_day)
        self.num_ids_test_night = self._count_ids(self.test_night)

        self.statistics()

    def _load_split(self, period: str, split: str, store: List[Tuple[Path, int]]):
        """
        period: 'day' 或 'night'
        split: 'train' 或 'test'
        store: 存储列表
        """
        list_file = osp.join(self.split_dir, f"{period}_{split}.txt")
        root_dir = osp.join(self.data_root, period)
        assert osp.exists(list_file), f"No such file: {list_file}"
        assert osp.exists(root_dir), f"can't find root_dir: {root_dir}"

        with open(list_file, "r") as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                abspath = osp.join(root_dir, rel)
                if not osp.exists(abspath):
                    print(f"Warning: The file does not exist {abspath}")
                    continue
                pid = _parse_pid_from_rel(rel)
                store.append((abspath, pid))

    def _count_ids(self, data: List[Tuple[Path, int]]) -> int:
        """统计唯一 pid 数量"""
        pids = set(pid for _, pid in data if pid != -1)
        return len(pids)

    def statistics(self):

        print('Dataset {} statistics:'.format(self.dataset_dir))
        print('  ------------------------------')
        print('  subset   | # ids | # images')
        print('  ------------------------------')
        print('  train_day    | {:5d} | {:8d}'.format(self.num_ids_train_day, len(self.train_day)))
        print('  train_night  | {:5d} | {:8d}'.format(self.num_ids_train_night, len(self.train_night)))
        print('  ------------------------------')
        print('  query_day     | {:5d} | {:8d}'.format(self.num_ids_query_day, len(self.query_day)))
        print('  query_night   | {:5d} | {:8d}'.format(self.num_ids_query_night, len(self.query_night)))
        print('  ------------------------------')
        print('  test_day     | {:5d} | {:8d}'.format(self.num_ids_test_day, len(self.test_day)))
        print('  test_night   | {:5d} | {:8d}'.format(self.num_ids_test_night, len(self.test_night)))