"""基于HotpotQA数据集的环境实现，不使用Wikipedia搜索"""
import gym
from typing import Optional, Dict, List
from pathlib import Path
import json
import re
import string


def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


class DatasetWikiEnv(gym.Env):
    """
    基于HotpotQA数据集的环境，从数据集中提取信息而不是搜索Wikipedia
    
    数据集格式：
    {
        "question": "...",
        "answer": "...",
        "context": [
            {
                "title": "Entity Name",
                "sentences": ["sentence1", "sentence2", ...]
            },
            ...
        ],
        "supporting_facts": [
            {"title": "...", "sent_idx": 0},
            ...
        ]
    }
    """
    
    def __init__(self, dataset_data: Optional[List[Dict]] = None, data_file: Optional[str] = None):
        """
        初始化环境
        
        Args:
            dataset_data: 已加载的数据集数据（列表）
            data_file: 数据集文件路径（如果dataset_data为None，则从文件加载）
        """
        super().__init__()
        
        # 加载数据集
        if dataset_data is not None:
            self.dataset = dataset_data
        elif data_file:
            self._load_dataset(data_file)
        else:
            raise ValueError("必须提供 dataset_data 或 data_file 之一")
        
        # 当前问题的索引和上下文
        self.current_idx = None
        self.current_context: Dict[str, List[str]] = {}  # title -> sentences
        self.current_question = None
        self.current_answer = None
        
        # 环境状态
        self.page = None  # 当前页面内容
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0
    
    def _load_dataset(self, data_file: str):
        """从文件加载数据集"""
        file_path = Path(data_file)
        if not file_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {data_file}")
        
        print(f"正在加载数据集: {data_file}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        print(f"成功加载 {len(self.dataset)} 个样本")
    
    def _get_obs(self):
        return self.obs
    
    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}
    
    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """重置环境，设置当前问题"""
        # **关键修复：完全清理所有状态，避免状态污染**
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.current_idx = None
        self.current_question = None
        self.current_answer = None
        self.current_context = {}  # 先清空
        self.result_titles = None  # 清理搜索结果标题
        
        # 设置当前问题
        if idx is None:
            import random
            if seed is not None:
                random.seed(seed)
            self.current_idx = random.randint(0, len(self.dataset) - 1)
        else:
            # 确保 idx 在有效范围内
            if idx < 0 or idx >= len(self.dataset):
                raise ValueError(f"问题索引 {idx} 超出范围 [0, {len(self.dataset)-1}]")
            self.current_idx = idx
        
        # 获取当前问题的数据
        current_data = self.dataset[self.current_idx]
        self.current_question = current_data.get('question', '')
        self.current_answer = current_data.get('answer', '')
        
        # **关键修复：重新构建上下文字典，确保完全清理旧数据**
        self.current_context = {}
        context_list = current_data.get('context', [])
        
        if not context_list:
            print(f"警告: 问题 {self.current_idx} 没有 context 数据")
        
        for idx, ctx_item in enumerate(context_list):
            # 处理两种格式：字典格式 {"title": "...", "sentences": [...]} 或列表格式 ["title", ["sentence1", ...]]
            try:
                if isinstance(ctx_item, dict):
                    # 字典格式: {"title": "...", "sentences": [...]}
                    title = ctx_item.get('title', '')
                    sentences = ctx_item.get('sentences', [])
                elif isinstance(ctx_item, list) and len(ctx_item) >= 2:
                    # 列表格式: [title, [sentence1, sentence2, ...]]
                    title = ctx_item[0] if isinstance(ctx_item[0], str) else str(ctx_item[0]) if ctx_item[0] else ''
                    sentences = ctx_item[1] if isinstance(ctx_item[1], list) else []
                else:
                    # 未知格式，打印警告并跳过
                    print(f"警告: 问题 {self.current_idx} 的 context[{idx}] 格式未知: {type(ctx_item)} - {str(ctx_item)[:100]}")
                    continue
                
                if title and sentences:
                    # 确保 sentences 是列表
                    if not isinstance(sentences, list):
                        print(f"警告: 问题 {self.current_idx} 的 context[{idx}] sentences 不是列表，跳过")
                        continue
                    # 将句子列表合并为段落
                    self.current_context[title] = sentences
                elif not title:
                    print(f"警告: 问题 {self.current_idx} 的 context[{idx}] 缺少 title")
                elif not sentences:
                    print(f"警告: 问题 {self.current_idx} 的 context[{idx}] 缺少 sentences")
            except Exception as e:
                print(f"错误: 处理问题 {self.current_idx} 的 context[{idx}] 时出错: {e}")
                print(f"  ctx_item 类型: {type(ctx_item)}, 内容: {str(ctx_item)[:200]}")
                continue
        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def construct_lookup_list(self, keyword):
        """查找包含关键词的句子"""
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        parts = [p for p in sentences if keyword.lower() in p.lower()]
        return parts
    
    @staticmethod
    def get_page_obs(page):
        """从页面提取观察"""
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])
    
    def search_step(self, entity):
        """
        从数据集中搜索实体
        
        Args:
            entity: 要搜索的实体名称
        """
        import time
        old_time = time.time()
        
        # 在上下文中查找匹配的title
        entity_lower = entity.lower().strip()
        matched_title = None
        matched_sentences = None
        
        # 精确匹配
        for title, sentences in self.current_context.items():
            if title.lower() == entity_lower:
                matched_title = title
                matched_sentences = sentences
                break
        
        # 模糊匹配（如果精确匹配失败）
        if matched_title is None:
            # 检查实体是否包含在title中，或title是否包含在实体中
            for title, sentences in self.current_context.items():
                title_lower = title.lower()
                if entity_lower in title_lower or title_lower in entity_lower:
                    matched_title = title
                    matched_sentences = sentences
                    break
        
        # 如果仍然没有匹配，尝试在句子内容中搜索（适用于 SQuAD 等单 context 数据集）
        if matched_title is None and len(self.current_context) == 1:
            # **改进：对于单 context 的情况（如 SQuAD），更智能的匹配**
            for title, sentences in self.current_context.items():
                # 检查实体关键词是否在句子中出现
                entity_words = [w for w in entity_lower.split() if len(w) > 2]  # 忽略太短的词
                all_text = ' '.join(sentences).lower()
                
                # 如果实体中的关键词在文本中出现，返回该 context
                if entity_words and any(word in all_text for word in entity_words):
                    matched_title = title
                    matched_sentences = sentences
                    break
                
                # **新增：如果实体是问题中的关键词，也返回该 context（SQuAD 常见情况）**
                if self.current_question:
                    question_lower = self.current_question.lower()
                    # 检查实体是否与问题相关
                    if any(word in question_lower for word in entity_words if len(word) > 3):
                        matched_title = title
                        matched_sentences = sentences
                        break
        
        # 如果找到匹配
        if matched_title and matched_sentences:
            # 将句子列表合并为页面内容
            self.page = "\n".join(matched_sentences)
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        else:
            # 没有找到，返回相似的结果
            similar_titles = []
            for title in self.current_context.keys():
                # 简单的相似度检查：检查是否有共同单词
                entity_words = set(entity_lower.split())
                title_words = set(title.lower().split())
                if entity_words & title_words:  # 有交集
                    similar_titles.append(title)
            
            if similar_titles:
                self.result_titles = similar_titles[:5]
                self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            else:
                # **改进：对于单 context 的情况，更智能的处理**
                if len(self.current_context) == 1:
                    # 对于 SQuAD 等单 context 数据集，即使没有精确匹配，也返回该 context
                    # 因为答案很可能就在这个唯一的 context 中
                    title = list(self.current_context.keys())[0]
                    sentences = list(self.current_context.values())[0]
                    self.page = "\n".join(sentences)
                    self.obs = self.get_page_obs(self.page)
                    self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                    # **新增：添加提示信息，说明这是唯一的可用上下文**
                    # 但不改变 obs，因为环境要求返回页面内容
                else:
                    self.obs = f"Could not find {entity}. No similar results found."
                    self.page = None
        
        self.search_time += time.time() - old_time
        self.num_searches += 1
    
    def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        
        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            self.search_step(entity)
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            # 评估答案是否正确
            reward = self._evaluate_answer(answer, self.current_answer)
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = f"Invalid action: {action}"
        
        self.steps += 1
        return self.obs, reward, done, self._get_info()
    
    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }
    
    def get_current_question(self) -> str:
        """获取当前问题"""
        return self.current_question
    
    def get_current_answer(self) -> str:
        """获取当前答案"""
        return self.current_answer
    
    def get_current_context_titles(self) -> List[str]:
        """获取当前问题的所有可用上下文标题"""
        return list(self.current_context.keys())
    
    def _evaluate_answer(self, prediction: str, ground_truth: str) -> int:
        """
        评估答案是否正确（改进版，支持部分匹配和更灵活的匹配）
        
        Args:
            prediction: 预测答案
            ground_truth: 正确答案
            
        Returns:
            1 如果正确，0 如果错误
        """
        import re
        import string
        
        def normalize_answer(s):
            """归一化答案字符串"""
            if not s:
                return ""
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        # 基本归一化匹配
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(ground_truth)
        
        if pred_norm == gt_norm:
            return 1
        
        # **改进：支持部分匹配（如果预测答案包含正确答案，或正确答案包含预测答案）**
        # 这对于长答案或包含额外信息的答案很有用
        if pred_norm and gt_norm:
            # 检查是否互为子串（去除空格后）
            pred_clean = pred_norm.replace(" ", "")
            gt_clean = gt_norm.replace(" ", "")
            if pred_clean in gt_clean or gt_clean in pred_clean:
                # 但要求至少匹配一定长度（避免太短的误匹配）
                min_len = min(len(pred_clean), len(gt_clean))
                if min_len >= 3:  # 至少3个字符
                    return 1
        
        return 0

