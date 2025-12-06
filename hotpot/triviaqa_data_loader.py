"""TriviaQA 数据加载器"""
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os


class TriviaQADataLoader:
    """加载 TriviaQA RC 数据集"""
    
    def __init__(self, data_dir: Optional[str] = None, evidence_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，如果为None则使用默认路径
            evidence_dir: 证据文档目录路径（包含wikipedia目录）
        """
        if data_dir is None:
            base_path = Path(__file__).parent.parent
            data_dir = base_path / "data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置证据目录（默认在数据目录的父目录下查找）
        if evidence_dir is None:
            # 尝试在常见位置查找
            possible_paths = [
                Path("E:/xlg/triviaqa-rc/evidence/wikipedia"),  # 用户提供的路径
                self.data_dir.parent / "triviaqa-rc" / "evidence" / "wikipedia",
                self.data_dir / "triviaqa-rc" / "evidence" / "wikipedia",
                Path(__file__).parent.parent.parent / "triviaqa-rc" / "evidence" / "wikipedia",
            ]
            for path in possible_paths:
                if path.exists():
                    self.evidence_dir = path
                    print(f"找到 TriviaQA evidence 目录: {path}")
                    break
            else:
                self.evidence_dir = None
                print("警告: 未找到 TriviaQA evidence 目录，将无法加载文档内容")
                print("提示: 请确保 TriviaQA evidence 目录存在，或通过 evidence_dir 参数指定")
        else:
            self.evidence_dir = Path(evidence_dir)
        
        self.data = []
        self.full_data = []
    
    def load_full_dataset(self, data_file: Optional[str] = None) -> List[Dict]:
        """
        加载完整格式的 TriviaQA RC 数据集并转换为 HotpotQA 兼容格式
        
        Args:
            data_file: 本地文件路径（可以是相对路径或绝对路径）
            
        Returns:
            转换为 HotpotQA 格式的数据列表
        """
        if data_file is None:
            data_file = "verified-wikipedia-dev.json"
        
        # 处理文件路径
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = self.data_dir / data_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"TriviaQA 数据文件不存在: {data_path}")
        
        print(f"正在加载 TriviaQA 数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            triviaqa_data = json.load(f)
        
        # TriviaQA 格式: {"Data": [...]}
        if isinstance(triviaqa_data, dict) and "Data" in triviaqa_data:
            triviaqa_items = triviaqa_data["Data"]
        elif isinstance(triviaqa_data, list):
            triviaqa_items = triviaqa_data
        else:
            raise ValueError(f"未知的 TriviaQA 数据格式: {type(triviaqa_data)}")
        
        # 转换为 HotpotQA 兼容格式
        converted_data = []
        for item in triviaqa_items:
            question = item.get('Question', '')
            answer_obj = item.get('Answer', {})
            answer = answer_obj.get('Value', '') or answer_obj.get('NormalizedValue', '')
            
            # 提取上下文文档
            context = []
            entity_pages = item.get('EntityPages', [])
            
            for page in entity_pages:
                title = page.get('Title', '')
                filename = page.get('Filename', '')
                
                if not title:
                    continue
                
                # 尝试加载文档内容
                sentences = []
                if filename and self.evidence_dir:
                    doc_path = self.evidence_dir / filename
                    if doc_path.exists():
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # 将文档内容分割为句子
                                sentences = [s.strip() + '.' for s in content.split('. ') if s.strip()]
                        except Exception as e:
                            print(f"警告: 无法读取文档 {doc_path}: {e}")
                            # 如果没有文档，至少保留标题
                            sentences = [f"Information about {title}."]
                    else:
                        sentences = [f"Information about {title}."]
                else:
                    sentences = [f"Information about {title}."]
                
                if sentences:
                    context.append({
                        'title': title,
                        'sentences': sentences
                    })
            
            if question and answer and context:
                converted_data.append({
                    'question': question,
                    'answer': answer,
                    'context': context
                })
        
        self.full_data = converted_data
        print(f"成功加载 {len(converted_data)} 个 TriviaQA 样本（已转换为 HotpotQA 格式）")
        return converted_data
    
    def load_split(self, split: str = "dev") -> List[Tuple[str, str]]:
        """
        加载指定数据集分割（简化格式，只包含question和answer）
        
        Args:
            split: 数据集分割（TriviaQA 通常只有 dev）
            
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

