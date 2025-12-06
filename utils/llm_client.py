"""LLM客户端封装"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 从项目根目录加载 .env 文件
env_path = Path(__file__).parent.parent / '.env'
# 尝试加载 .env 文件（使用 utf-8-sig 处理 BOM）
if env_path.exists():
    # 先手动解析并设置环境变量（处理 BOM 问题）
    try:
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
    except Exception as e:
        print(f"警告: 手动加载 .env 文件失败: {e}")
    # 然后再用 load_dotenv（作为备用）
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print(f"警告: .env 文件不存在: {env_path}")


class LLMClient:
    """LLM API 客户端封装（支持 OpenAI、DeepSeek 和 Qwen）"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, 
                 api_provider: str = "deepseek"):
        """
        初始化LLM客户端
        
        Args:
            model: 使用的模型名称
            temperature: 温度参数
            api_provider: API提供商，'deepseek'、'openai' 或 'qwen'
        """
        self.api_provider = api_provider.lower()
        self.model = model
        self.temperature = temperature
        
        # 根据 API provider 获取对应的 API key 和 base_url
        if self.api_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                # Fallback to manual .env parsing if load_dotenv failed
                try:
                    if env_path.exists():
                        with open(env_path, 'r', encoding='utf-8-sig') as f:
                            for line in f:
                                # 去除 BOM 和空白字符
                                line = line.strip()
                                # 跳过注释和空行
                                if not line or line.startswith('#'):
                                    continue
                                if line.startswith("DEEPSEEK_API_KEY="):
                                    api_key = line.split('=', 1)[1].strip()
                                    # 去除可能的引号
                                    api_key = api_key.strip('"').strip("'")
                                    if api_key:
                                        os.environ["DEEPSEEK_API_KEY"] = api_key
                                        break
                except Exception as e:
                    print(f"手动加载 .env 文件失败: {e}")
                if not api_key:
                    raise ValueError(f"请设置 DEEPSEEK_API_KEY 环境变量或在 .env 文件中配置。.env 文件路径: {env_path}")
            # DeepSeek 使用 OpenAI 兼容的 API，但需要设置 base_url
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        elif self.api_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量或在 .env 文件中配置")
            self.client = OpenAI(api_key=api_key)
        elif self.api_provider == "qwen":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                # Fallback to manual .env parsing if load_dotenv failed
                try:
                    if env_path.exists():
                        with open(env_path, 'r', encoding='utf-8-sig') as f:
                            for line in f:
                                # 去除 BOM 和空白字符
                                line = line.strip()
                                # 跳过注释和空行
                                if not line or line.startswith('#'):
                                    continue
                                if line.startswith("DASHSCOPE_API_KEY="):
                                    api_key = line.split('=', 1)[1].strip()
                                    # 去除可能的引号
                                    api_key = api_key.strip('"').strip("'")
                                    if api_key:
                                        os.environ["DASHSCOPE_API_KEY"] = api_key
                                        break
                except Exception as e:
                    print(f"手动加载 .env 文件失败: {e}")
                if not api_key:
                    raise ValueError(f"请设置 DASHSCOPE_API_KEY 环境变量或在 .env 文件中配置。.env 文件路径: {env_path}")
            # Qwen 使用 OpenAI 兼容的 API，需要设置 base_url
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            raise ValueError(f"不支持的 API provider: {api_provider}，请使用 'deepseek'、'openai' 或 'qwen'")
    
    def generate(self, prompt: str, max_tokens: int = 200, retries: int = 3) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            max_tokens: 最大token数
            retries: 重试次数
            
        Returns:
            生成的文本
        """
        import time
        import json
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                
                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        return content.strip()
                    else:
                        print(f"警告: LLM 返回空内容，尝试 {attempt + 1}/{retries}")
                else:
                    print(f"警告: LLM 响应格式异常，尝试 {attempt + 1}/{retries}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误 (尝试 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return ""
            except Exception as e:
                error_msg = str(e)
                print(f"LLM调用错误 (尝试 {attempt + 1}/{retries}): {error_msg}")
                
                # 如果是速率限制错误，等待更长时间
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** (attempt + 2)
                    print(f"检测到速率限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                elif attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return ""
        
        return ""
    
    def generate_multiple(self, prompt: str, n: int = 3, max_tokens: int = 200) -> List[str]:
        """
        生成多个候选答案
        
        Args:
            prompt: 提示词
            n: 生成数量
            max_tokens: 最大token数
            
        Returns:
            生成的文本列表
        """
        results = []
        for _ in range(n):
            result = self.generate(prompt, max_tokens)
            if result:
                results.append(result)
        return results
    
    def evaluate(self, prompt: str) -> float:
        """
        评估状态（返回0-1之间的分数）
        
        Args:
            prompt: 评估提示词
            
        Returns:
            评估分数（0-1之间）
        """
        try:
            response = self.generate(prompt, max_tokens=50)
            # 尝试从响应中提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # 归一化到0-1
                if score > 1:
                    score = score / 100.0 if score <= 100 else 1.0
                return max(0.0, min(1.0, score))
            return 0.5  # 默认分数
        except:
            return 0.5

