"""SQuAD 数据加载器"""
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re


class SQuADDataLoader:
    """加载 SQuAD 数据集"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，如果为None则使用默认路径
        """
        if data_dir is None:
            base_path = Path(__file__).parent.parent
            data_dir = base_path / "data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
        self.full_data = []
    
    def load_full_dataset(self, data_file: Optional[str] = None) -> List[Dict]:
        """
        加载完整格式的 SQuAD 数据集并转换为 HotpotQA 兼容格式
        
        Args:
            data_file: 本地文件路径（可以是相对路径或绝对路径）
            
        Returns:
            转换为 HotpotQA 格式的数据列表
        """
        if data_file is None:
            data_file = "dev-v2.0.json"
        
        # 处理文件路径
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = self.data_dir / data_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"SQuAD 数据文件不存在: {data_path}")
        
        print(f"正在加载 SQuAD 数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        # SQuAD 格式: {"version": "...", "data": [...]}
        if isinstance(squad_data, dict) and "data" in squad_data:
            squad_items = squad_data["data"]
        elif isinstance(squad_data, list):
            squad_items = squad_data
        else:
            raise ValueError(f"未知的 SQuAD 数据格式: {type(squad_data)}")
        
        # 转换为 HotpotQA 兼容格式
        converted_data = []
        for article in squad_items:
            title = article.get('title', '')
            paragraphs = article.get('paragraphs', [])
            
            for para in paragraphs:
                context = para.get('context', '')
                qas = para.get('qas', [])
                
                # 将段落分割为句子
                sentences = [s.strip() + '.' for s in context.split('. ') if s.strip()]
                
                for qa in qas:
                    question = qa.get('question', '')
                    
                    # 获取答案（SQuAD v2.0 可能有 plausible_answers）
                    answers = qa.get('answers', [])
                    if not answers:
                        answers = qa.get('plausible_answers', [])
                    
                    if not answers:
                        continue
                    
                    # 使用第一个答案
                    answer = answers[0].get('text', '')
                    
                    if question and answer and sentences:
                        converted_data.append({
                            'question': question,
                            'answer': answer,
                            'context': [{
                                'title': title or 'Context',
                                'sentences': sentences
                            }]
                        })
        
        self.full_data = converted_data
        print(f"成功加载 {len(converted_data)} 个 SQuAD 样本（已转换为 HotpotQA 格式）")
        return converted_data
    
    def load_split(self, split: str = "dev") -> List[Tuple[str, str]]:
        """
        加载指定数据集分割（简化格式，只包含question和answer）
        
        Args:
            split: 数据集分割（SQuAD 通常只有 dev）
            
        Returns:
            (question, answer) 元组列表
        """
        # 加载完整数据
        full_data = self.load_full_dataset()
        
        # 提取问题和答案
        questions_answers = []
        for item in full_data:
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
            random_sample: 是否随机采样
            random_seed: 随机种子
        """
        if not self.data:
            self.load_split(split)
        
        if random_sample:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            if num_problems >= len(self.data):
                return self.data.copy()
            return random.sample(self.data, num_problems)
        else:
            end_idx = min(start_idx + num_problems, len(self.data))
            return self.data[start_idx:end_idx]










