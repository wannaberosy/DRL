"""环境包装器，添加网络错误处理和重试机制"""
import time
import requests
import sys
from pathlib import Path
from typing import Tuple, Any, Optional

# 添加 hotpot 原始代码路径
hotpot_path = Path(__file__).parent.parent / "LanguageAgentTreeSearch-main" / "LanguageAgentTreeSearch-main" / "hotpot"
if str(hotpot_path) not in sys.path:
    sys.path.insert(0, str(hotpot_path))

from bs4 import BeautifulSoup
import wikienv


class RobustWikiEnvWrapper:
    """包装 WikiEnv，添加网络错误处理和重试机制"""
    
    def __init__(self, env, max_retries: int = 5, timeout: int = 30, retry_delay: float = 2.0):
        """
        初始化包装器
        
        Args:
            env: 原始 WikiEnv 环境
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            retry_delay: 重试延迟（秒）
        """
        self.env = env
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        
        # 保存原始的 search_step 方法
        self.original_search_step = env.search_step
        
        # 替换为增强版本
        env.search_step = self._robust_search_step
    
    def _robust_search_step(self, entity):
        """
        增强的 search_step，带重试和错误处理
        
        Args:
            entity: 要搜索的实体
        """
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                old_time = time.time()
                # 添加超时和重试
                response = requests.get(
                    search_url,
                    timeout=self.timeout,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                response.raise_for_status()  # 检查 HTTP 错误
                response_text = response.text
                
                self.env.search_time += time.time() - old_time
                self.env.num_searches += 1
                
                # 解析响应
                soup = BeautifulSoup(response_text, features="html.parser")
                result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
                
                if result_divs:  # mismatch
                    self.env.result_titles = [wikienv.clean_str(div.get_text().strip()) for div in result_divs]
                    self.env.obs = f"Could not find {entity}. Similar: {self.env.result_titles[:5]}."
                else:
                    page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
                    if any("may refer to:" in p for p in page):
                        # 递归调用（也会使用包装后的方法）
                        self._robust_search_step("[" + entity + "]")
                    else:
                        self.env.page = ""
                        for p in page:
                            if len(p.split(" ")) > 2:
                                self.env.page += wikienv.clean_str(p)
                                if not p.endswith("\n"):
                                    self.env.page += "\n"
                        self.env.obs = self.env.get_page_obs(self.env.page)
                        self.env.lookup_keyword = self.env.lookup_list = self.env.lookup_cnt = None
                
                # 成功，返回
                return
                
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException,
                    ConnectionResetError,
                    OSError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"网络错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)[:100]}... 等待 {wait_time:.1f} 秒后重试")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    print(f"网络错误，已达到最大重试次数: {str(e)[:100]}")
                    # 返回一个错误消息
                    self.env.obs = f"Network error: Unable to search for '{entity}'. Please try again later."
                    self.env.page = None
                    return
        
        # 如果所有重试都失败
        if last_error:
            self.env.obs = f"Network error after {self.max_retries} attempts: Unable to search for '{entity}'."
            self.env.page = None

