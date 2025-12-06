"""HotpotQA ReAct 求解器"""
import re
from typing import Tuple, List, Optional
from utils.llm_client import LLMClient
import sys
from pathlib import Path

# 添加 hotpot 原始代码路径
hotpot_path = Path(__file__).parent.parent / "LanguageAgentTreeSearch-main" / "LanguageAgentTreeSearch-main" / "hotpot"
if str(hotpot_path) not in sys.path:
    sys.path.insert(0, str(hotpot_path))

import os
import wikienv
import wrappers

# 尝试导入本地环境
try:
    from hotpot.local_wikienv import LocalWikiEnv
    HAS_LOCAL_ENV = True
except ImportError:
    HAS_LOCAL_ENV = False
    print("警告: 无法导入 LocalWikiEnv，将使用原始 WikiEnv")

# 尝试导入数据集环境
try:
    from hotpot.dataset_wikienv import DatasetWikiEnv
    HAS_DATASET_ENV = True
except ImportError:
    HAS_DATASET_ENV = False
    print("警告: 无法导入 DatasetWikiEnv")

# 修改 wrappers 模块中的 DATA_DIR 为绝对路径，这样就不需要切换工作目录了
hotpot_data_dir = hotpot_path / "data"
wrappers.DATA_DIR = str(hotpot_data_dir)


class HotpotReActSolver:
    """HotpotQA ReAct 求解器：简单的思考-行动循环"""
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 10, 
                 env_backend: str = "wikipedia", cache_file: Optional[str] = None,
                 dataset_file: Optional[str] = None, dataset_url: Optional[str] = None):
        """
        初始化ReAct求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大迭代次数
            env_backend: 环境后端类型，'dataset'（使用数据集）、'wikipedia'（使用 Wikipedia 库）、'mock'（模拟环境）或 'original'（原始环境）
            cache_file: 缓存文件路径（用于 mock 模式）
            dataset_file: 数据集文件路径（用于 dataset 模式）
            dataset_url: 数据集URL（用于 dataset 模式，如果文件不存在则下载）
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        
        # 初始化环境
        if env_backend == "dataset" and HAS_DATASET_ENV:
            # 使用数据集环境
            print(f"使用数据集环境 (后端: dataset)")
            
            # 根据文件名自动检测数据集类型
            dataset_type = 'h'  # 默认 HotpotQA
            if dataset_file:
                if 'triviaqa' in dataset_file.lower() or 'verified-wikipedia' in dataset_file.lower():
                    dataset_type = 't'
                elif 'squad' in dataset_file.lower() or 'dev-v2.0' in dataset_file.lower() or 'dev-v1.1' in dataset_file.lower():
                    dataset_type = 's'
            
            # 根据数据集类型选择加载器
            if dataset_type == 't':
                from hotpot.triviaqa_data_loader import TriviaQADataLoader
                data_loader = TriviaQADataLoader()
                dataset_data = data_loader.load_full_dataset(data_file=dataset_file)
            elif dataset_type == 's':
                from hotpot.squad_data_loader import SQuADDataLoader
                data_loader = SQuADDataLoader()
                dataset_data = data_loader.load_full_dataset(data_file=dataset_file)
            else:
                from hotpot.data_loader import HotpotDataLoader
                data_loader = HotpotDataLoader()
                dataset_data = data_loader.load_full_dataset(data_file=dataset_file, url=dataset_url)
            
            base_env = DatasetWikiEnv(dataset_data=dataset_data)
        elif env_backend in ["wikipedia", "mock"] and HAS_LOCAL_ENV:
            # 使用新的本地环境
            print(f"使用本地环境 (后端: {env_backend})")
            base_env = LocalWikiEnv(backend=env_backend, cache_file=cache_file)
        else:
            # 使用原始环境
            if env_backend not in ["wikipedia", "mock", "original", "dataset"]:
                print(f"警告: 未知的后端类型 '{env_backend}'，使用 'original'")
            print("使用原始 WikiEnv（可能需要网络连接）")
            base_env = wikienv.WikiEnv()
            
            # 添加网络错误处理包装器
            try:
                import importlib.util
                env_wrapper_path = Path(__file__).parent / "env_wrapper.py"
                if env_wrapper_path.exists():
                    spec = importlib.util.spec_from_file_location("hotpot.env_wrapper", env_wrapper_path)
                    env_wrapper_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(env_wrapper_module)
                    env_wrapper_module.RobustWikiEnvWrapper(base_env, max_retries=5, timeout=30, retry_delay=2.0)
            except Exception as e:
                print(f"警告: 无法加载网络错误处理包装器: {e}")
                print("将继续使用原始环境（可能没有重试机制）")
        
        self.env = wrappers.HotPotQAWrapper(base_env, split="dev")
    
    def extract_action(self, response: str) -> Optional[str]:
        """
        从LLM响应中提取行动（必须是小写格式）
        
        Args:
            response: LLM响应文本
            
        Returns:
            提取的行动字符串（小写格式），如果未找到则返回None
        """
        # 查找 Action 模式（不区分大小写）
        action_patterns = [
            r'Action\s*\d*\s*:\s*(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
            r'(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                action = match.group(1) if match.lastindex else match.group(0)
                action = action.strip()
                
                # 转换为小写格式（环境要求小写）
                # 提取 action 类型和参数
                if '[' in action and ']' in action:
                    action_type = action.split('[')[0].lower()  # 转换为小写
                    action_param = action.split('[')[1].split(']')[0]
                    return f"{action_type}[{action_param}]"
                else:
                    return action.lower()
        
        return None
    
    def solve(self, question: str, ground_truth: str, question_idx: int = 0) -> Tuple[bool, str, List[dict]]:
        """
        使用ReAct方法解决问题
        
        Args:
            question: 问题文本
            ground_truth: 正确答案（用于评估）
            question_idx: 问题索引
            
        Returns:
            (success, answer, history): 是否成功、答案、历史记录
        """
        history = []
        
        # **修复：重置环境，确保使用正确的问题索引**
        obs = self.env.reset(idx=question_idx)
        
        # **关键修复：确保 current_state 包含当前问题信息**
        # 获取环境中的实际问题（确保一致性）
        env_question = getattr(self.env, 'get_current_question', lambda: question)()
        if env_question and env_question != question:
            # 如果环境中的问题与传入的问题不一致，使用环境中的问题
            print(f"警告: 环境问题与传入问题不一致。使用环境问题: {env_question[:50]}...")
            question = env_question
        
        # **修复：初始化 current_state 包含问题和初始观察**
        current_state = f"Question: {question}\n{obs}"
        
        # CoT prompt 模板
        cot_prompt_template = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
After each observation, provide the next Thought and next Action.

{current_state}
"""
        
        for iteration in range(self.max_iterations):
            # **修复：构建提示词（不再需要单独传递 question，因为已包含在 current_state 中）**
            prompt = cot_prompt_template.format(
                current_state=current_state
            )
            
            # 生成响应
            response = self.llm.generate(prompt, max_tokens=300)
            
            # 如果响应为空，记录并跳过这次迭代
            if not response or not response.strip():
                history.append({
                    'iteration': iteration + 1,
                    'thought': 'LLM 返回空响应，跳过此次迭代',
                    'state': current_state,
                    'error': 'empty_response'
                })
                # 使用默认行动继续
                action = "think[continue reasoning]"
            else:
                history.append({
                    'iteration': iteration + 1,
                    'thought': response,
                    'state': current_state
                })
                
                # 提取行动
                action = self.extract_action(response)
                
                if not action:
                    # 如果无法提取行动，尝试从响应中直接提取（转换为小写）
                    if 'finish[' in response.lower():
                        finish_match = re.search(r'finish\[([^\]]+)\]', response, re.IGNORECASE)
                        if finish_match:
                            action = f"finish[{finish_match.group(1)}]"
                    elif 'search[' in response.lower():
                        search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
                        if search_match:
                            action = f"search[{search_match.group(1)}]"
                    elif 'lookup[' in response.lower():
                        lookup_match = re.search(r'lookup\[([^\]]+)\]', response, re.IGNORECASE)
                        if lookup_match:
                            action = f"lookup[{lookup_match.group(1)}]"
                
                if not action:
                    # 如果仍然无法提取，使用默认的思考行动
                    action = "think[continue reasoning]"
            
            # 执行行动（带错误处理）
            try:
                obs, reward, done, info = self.env.step(action)
            except Exception as e:
                # 网络错误或其他环境错误
                error_msg = str(e)
                print(f"执行 action 时出错: {error_msg[:100]}")
                obs = f"Error executing action: {error_msg[:100]}. Please try a different approach."
                reward = 0
                done = False
                info = {}
            
            history[-1]['action'] = action
            history[-1]['observation'] = obs
            history[-1]['reward'] = reward
            
            # 检查是否完成
            if done:
                if reward == 1:
                    # 成功
                    answer = info.get('answer', '')
                    history.append({
                        'iteration': iteration + 1,
                        'status': 'success',
                        'answer': answer
                    })
                    return True, answer, history
                else:
                    # 失败
                    answer = info.get('answer', '')
                    history.append({
                        'iteration': iteration + 1,
                        'status': 'failed',
                        'answer': answer,
                        'ground_truth': ground_truth
                    })
                    return False, answer, history
            
            # 更新状态
            current_state = f"{current_state}\nThought {iteration + 1}: {response}\nAction {iteration + 1}: {action}\nObservation {iteration + 1}: {obs}"
        
        # 达到最大迭代次数，返回失败
        history.append({
            'iteration': self.max_iterations,
            'status': 'max_iterations_reached'
        })
        return False, "", history

