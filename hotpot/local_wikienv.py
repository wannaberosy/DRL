"""本地 Wikipedia 环境实现，支持多种后端"""
import time
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


class LocalWikiEnv(gym.Env):
    """
    本地 Wikipedia 环境，支持多种后端：
    1. wikipedia Python 库（推荐，更稳定）
    2. 模拟环境（用于演示，不需要网络）
    """
    
    def __init__(self, backend: str = "wikipedia", cache_file: Optional[str] = None):
        """
        初始化环境
        
        Args:
            backend: 后端类型，'wikipedia' 或 'mock'
            cache_file: 缓存文件路径（用于 mock 模式）
        """
        super().__init__()
        self.backend = backend
        self.cache_file = cache_file
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0
        
        # 加载缓存（如果使用 mock 模式）
        self.cache: Dict[str, str] = {}
        if backend == "mock" and cache_file:
            self._load_cache()
        elif backend == "mock":
            # 使用默认的模拟数据
            self._init_mock_data()
        
        # 初始化 Wikipedia 库（如果使用）
        if backend == "wikipedia":
            try:
                import wikipedia
                self.wikipedia = wikipedia
                # 设置语言为英文
                wikipedia.set_lang("en")
                # 设置请求超时
                wikipedia.set_rate_limiting(True)
            except ImportError:
                print("警告: 未安装 wikipedia 库，将使用 mock 模式")
                print("安装命令: pip install wikipedia")
                self.backend = "mock"
                self._init_mock_data()
    
    def _load_cache(self):
        """从文件加载缓存"""
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"已加载 {len(self.cache)} 个缓存条目")
            except Exception as e:
                print(f"加载缓存失败: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self):
        """保存缓存到文件"""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"保存缓存失败: {e}")
    
    def _init_mock_data(self):
        """初始化模拟数据"""
        # 一些常见的实体和对应的模拟内容
        self.cache = {
            "United States": "The United States of America (USA) is a country in North America. It consists of 50 states, a federal district, and various territories.",
            "Barack Obama": "Barack Obama is an American politician who served as the 44th president of the United States from 2009 to 2017.",
            "Python": "Python is a high-level programming language known for its simplicity and readability.",
            "Wikipedia": "Wikipedia is a free, open-content online encyclopedia created through collaborative effort.",
            "New York": "New York is a state in the northeastern United States. New York City is the most populous city in the United States.",
            "China": "China is a country in East Asia. It is the world's most populous country with over 1.4 billion people.",
            "London": "London is the capital and largest city of England and the United Kingdom.",
            "Paris": "Paris is the capital and most populous city of France.",
            "Tokyo": "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world.",
            "Germany": "Germany is a country in Central Europe. It is the most populous member state of the European Union.",
        }
        print(f"已初始化 {len(self.cache)} 个模拟数据条目")
    
    def _get_obs(self):
        return self.obs
    
    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}
    
    def reset(self, seed=None, return_info=False, options=None):
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
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
    
    def search_step_wikipedia(self, entity):
        """使用 Wikipedia 库搜索"""
        try:
            old_time = time.time()
            
            # 先尝试直接搜索页面
            try:
                page = self.wikipedia.page(entity, auto_suggest=False)
                page_content = page.content
            except self.wikipedia.exceptions.DisambiguationError as e:
                # 如果有歧义，使用第一个选项
                page = self.wikipedia.page(e.options[0])
                page_content = page.content
            except self.wikipedia.exceptions.PageError:
                # 页面不存在，尝试搜索
                search_results = self.wikipedia.search(entity, results=5)
                if search_results:
                    self.result_titles = search_results
                    self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
                    self.search_time += time.time() - old_time
                    self.num_searches += 1
                    return
                else:
                    self.obs = f"Could not find {entity}. No similar results found."
                    self.search_time += time.time() - old_time
                    self.num_searches += 1
                    return
            
            # 提取前几段
            paragraphs = page_content.split('\n\n')
            self.page = ""
            for p in paragraphs[:5]:  # 只取前5段
                if len(p.split(" ")) > 2:
                    self.page += clean_str(p) + "\n"
            
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
            
            # 保存到缓存
            self.cache[entity] = self.page
            if self.cache_file:
                self._save_cache()
            
            self.search_time += time.time() - old_time
            self.num_searches += 1
            
        except Exception as e:
            error_msg = str(e)
            print(f"Wikipedia 搜索错误: {error_msg[:100]}")
            self.obs = f"Error searching for '{entity}': {error_msg[:100]}"
            self.page = None
            self.search_time += time.time() - old_time
            self.num_searches += 1
    
    def search_step_mock(self, entity):
        """使用模拟数据搜索"""
        old_time = time.time()
        
        # 在缓存中查找（不区分大小写）
        entity_lower = entity.lower()
        found = False
        
        for key, value in self.cache.items():
            if key.lower() == entity_lower or entity_lower in key.lower() or key.lower() in entity_lower:
                self.page = value
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                found = True
                break
        
        if not found:
            # 尝试模糊匹配
            similar = [key for key in self.cache.keys() 
                      if entity_lower in key.lower() or key.lower() in entity_lower]
            if similar:
                self.result_titles = similar[:5]
                self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            else:
                self.obs = f"Could not find {entity}. No similar results found. (Mock mode: limited data available)"
            self.page = None
        
        self.search_time += time.time() - old_time
        self.num_searches += 1
    
    def search_step(self, entity):
        """搜索步骤"""
        if self.backend == "wikipedia":
            self.search_step_wikipedia(entity)
        else:
            self.search_step_mock(entity)
    
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







