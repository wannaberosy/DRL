"""HotpotQA 数据加载器"""
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import urllib.request
import urllib.error


class HotpotDataLoader:
    """加载 HotpotQA 数据集"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，如果为None则使用默认路径
        """
        if data_dir is None:
            # 默认使用项目根目录下的 data 文件夹
            base_path = Path(__file__).parent.parent
            data_dir = base_path / "data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        self.data = []
        self.full_data = []  # 完整格式的数据（包含context）
    
    def download_dataset(self, url: str, output_file: str) -> Path:
        """
        从URL下载数据集
        
        Args:
            url: 数据集URL
            output_file: 输出文件路径（可以是相对路径或绝对路径）
            
        Returns:
            下载的文件路径
        """
        # 如果是相对路径，则保存到data_dir
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = self.data_dir / output_file
        
        # 如果文件已存在，跳过下载
        if output_path.exists():
            print(f"文件已存在: {output_path}")
            return output_path
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"正在从 {url} 下载数据集...")
        print(f"保存到: {output_path}")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"下载完成: {output_path}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"下载失败: {e}")
        
        return output_path
    
    def load_full_dataset(self, data_file: Optional[str] = None, 
                         url: Optional[str] = None) -> List[Dict]:
        """
        加载完整格式的数据集（包含context和supporting_facts）
        
        Args:
            data_file: 本地文件路径（可以是相对路径或绝对路径）
            url: 数据集URL（如果data_file不存在，则从URL下载）
            
        Returns:
            完整格式的数据列表
        """
        if data_file is None:
            # 使用默认路径
            data_file = "hotpot_train_v1.1.json"
        
        # 处理文件路径
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = self.data_dir / data_file
        
        # 如果文件不存在且提供了URL，尝试下载
        if not data_path.exists() and url:
            print(f"文件不存在，将从URL下载: {url}")
            data_path = self.download_dataset(url, data_file)
        elif not data_path.exists():
            # 如果文件不存在且没有提供URL，尝试使用默认URL
            default_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
            print(f"文件不存在，将从默认URL下载: {default_url}")
            data_path = self.download_dataset(default_url, data_file)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        print(f"正在加载完整数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.full_data = json.load(f)
        
        print(f"成功加载 {len(self.full_data)} 个样本")
        return self.full_data
    
    def load_split(self, split: str = "dev") -> List[Tuple[str, str]]:
        """
        加载指定数据集分割（简化格式，只包含question和answer）
        
        Args:
            split: 数据集分割，'train', 'dev', 或 'test'
            
        Returns:
            (question, answer) 元组列表
        """
        split_files = {
            "train": "hotpot_train_v1.1_simplified.json",
            "dev": "hotpot_dev_v1_simplified.json",
            "test": "hotpot_test_v1_simplified.json",
        }
        
        if split not in split_files:
            raise ValueError(f"Unknown split: {split}. Must be one of {list(split_files.keys())}")
        
        file_path = self.data_dir / split_files[split]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取问题和答案
        questions_answers = []
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            if question and answer:
                questions_answers.append((question, answer))
        
        self.data = questions_answers
        return questions_answers
    
    def get_problems(self, num_problems: int, split: str = "dev", start_idx: int = 0, 
                     random_sample: bool = False, random_seed: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        获取指定数量的问题（简化格式）
        
        Args:
            num_problems: 需要的问题数量
            split: 数据集分割
            start_idx: 起始索引（如果random_sample=False）
            random_sample: 是否随机采样（如果True，则忽略start_idx）
            random_seed: 随机种子（用于可复现性）
            
        Returns:
            (question, answer) 元组列表
        """
        if not self.data:
            self.load_split(split)
        
        if random_sample:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            # 随机采样
            if num_problems >= len(self.data):
                return self.data.copy()
            return random.sample(self.data, num_problems)
        else:
            # 顺序加载
            end_idx = min(start_idx + num_problems, len(self.data))
            return self.data[start_idx:end_idx]
    
    def get_full_problems(self, num_problems: int, start_idx: int = 0, 
                         data_file: Optional[str] = None,
                         url: Optional[str] = None,
                         random_sample: bool = False, random_seed: Optional[int] = None) -> List[Dict]:
        """
        获取指定数量的完整格式问题（包含context）
        
        Args:
            num_problems: 需要的问题数量
            start_idx: 起始索引（如果random_sample=False）
            data_file: 数据文件路径
            url: 数据集URL（如果文件不存在）
            random_sample: 是否随机采样（如果True，则忽略start_idx）
            random_seed: 随机种子（用于可复现性）
            
        Returns:
            完整格式的数据列表
        """
        if not self.full_data:
            self.load_full_dataset(data_file=data_file, url=url)
        
        if random_sample:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            # 随机采样
            if num_problems >= len(self.full_data):
                return self.full_data.copy()
            return random.sample(self.full_data, num_problems)
        else:
            # 顺序加载
            end_idx = min(start_idx + num_problems, len(self.full_data))
            return self.full_data[start_idx:end_idx]
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data) if self.data else len(self.full_data)

