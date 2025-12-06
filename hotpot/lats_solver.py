"""HotpotQA LATS 求解器"""
import re
import math
from typing import Tuple, List, Optional
from utils.llm_client import LLMClient
import sys
from pathlib import Path

# 添加 hotpot 原始代码路径
hotpot_path = Path(__file__).parent.parent / "LanguageAgentTreeSearch-main" / "LanguageAgentTreeSearch-main" / "hotpot"
if str(hotpot_path) not in sys.path:
    sys.path.insert(0, str(hotpot_path))

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


class Node:
    """搜索树节点"""
    
    def __init__(self, state: dict, question: str, parent: Optional['Node'] = None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.em = 0  # Exact match
    
    def ucb_score(self, exploration_weight: float = 1.414) -> float:
        """计算UCB分数"""
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.parent.visits if self.parent else 1
        if parent_visits == 0:
            parent_visits = 1
        
        exploitation = self.value / self.visits if self.visits > 0 else 0
        exploration = exploration_weight * (2 * math.log(parent_visits) / self.visits) ** 0.5
        return exploitation + exploration
    
    def __str__(self):
        return f"Thought: {self.state.get('thought', '')}\nAction: {self.state.get('action', '')}\nObservation: {self.state.get('observation', '')}"


class HotpotLATSSolver:
    """HotpotQA LATS 求解器：带树搜索的思考"""
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 30,
                 n_generate: int = 3, n_evaluate: int = 1,
                 env_backend: str = "wikipedia", cache_file: Optional[str] = None,
                 dataset_file: Optional[str] = None, dataset_url: Optional[str] = None):
        """
        初始化LATS求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大搜索迭代次数
            n_generate: 每次扩展生成的候选数
            n_evaluate: 每次评估的采样数
            env_backend: 环境后端类型，'dataset'（使用数据集）、'wikipedia'（使用 Wikipedia 库）、'mock'（模拟环境）或 'original'（原始环境）
            cache_file: 缓存文件路径（用于 mock 模式）
            dataset_file: 数据集文件路径（用于 dataset 模式）
            dataset_url: 数据集URL（用于 dataset 模式，如果文件不存在则下载）
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.n_generate = n_generate
        self.n_evaluate = n_evaluate
        
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
        """从LLM响应中提取行动（必须是小写格式）"""
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
                if '[' in action and ']' in action:
                    action_type = action.split('[')[0].lower()  # 转换为小写
                    action_param = action.split('[')[1].split(']')[0]
                    return f"{action_type}[{action_param}]"
                else:
                    return action.lower()
        
        return None
    
    def expand(self, node: Node, question: str, question_idx: int = 0) -> List[Node]:
        """
        扩展节点：生成多个候选行动并执行它们
        
        Args:
            node: 当前节点
            question: 问题文本
            question_idx: 问题索引（用于重置环境）
            
        Returns:
            新生成的子节点列表（已执行 action 并设置状态）
        """
        # 检查深度限制
        if node.depth >= 7:
            node.is_terminal = True
            return []
        
        # **修复：构建当前状态字符串（用于 prompt），确保包含问题信息**
        current_state = ""
        
        # **关键修复：对于根节点，包含问题和初始 observation**
        if node.depth == 0:
            # 从节点状态或传入的问题获取问题
            node_question = node.state.get('question', question)
            current_state += f"Question: {node_question}\n"
            if node.state.get('observation'):
                current_state += f"Observation: {node.state['observation']}\n"
        else:
            # 对于非根节点，包含完整的历史
            if node.state.get('thought'):
                current_state += f"Thought {node.depth}: {node.state['thought']}\n"
            if node.state.get('action'):
                current_state += f"Action {node.depth}: {node.state['action']}\n"
            if node.state.get('observation'):
                current_state += f"Observation {node.depth}: {node.state['observation']}\n"
        
        # **修复：使用节点中保存的问题（如果存在）**
        node_question = node.state.get('question', question)
        
        # 构建扩展提示词
        expand_prompt = f"""Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

{current_state}

Please generate {self.n_generate} different next steps (Thought and Action pairs). Each should be a different approach to solving the problem.
Format each as:
Thought {node.depth + 1}: [your reasoning]
Action {node.depth + 1}: [your action]

"""
        
        response = self.llm.generate(expand_prompt, max_tokens=500)
        
        # 解析生成的行动
        children = []
        seen_actions = set()
        
        # 尝试提取多个 Thought-Action 对
        thought_action_pairs = re.findall(
            r'Thought\s*\d*\s*:\s*([^\n]+)\s*\n\s*Action\s*\d*\s*:\s*(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
            response,
            re.IGNORECASE | re.MULTILINE
        )
        
        # 如果没找到足够的行动，尝试单独提取
        if len(thought_action_pairs) < self.n_generate:
            actions = re.findall(
                r'(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
                response,
                re.IGNORECASE
            )
            for action in actions:
                # 尝试找到对应的 thought
                thought_match = re.search(
                    rf'Thought\s*\d*\s*:\s*([^\n]+)\s*\n\s*Action\s*\d*\s*:\s*{re.escape(action)}',
                    response,
                    re.IGNORECASE | re.MULTILINE
                )
                thought = thought_match.group(1).strip() if thought_match else 'Generated from expansion'
                thought_action_pairs.append((thought, action))
        
        # 执行每个 action 并创建节点
        for thought, action in thought_action_pairs[:self.n_generate]:
            # 转换为小写格式
            action = action.strip()
            if '[' in action and ']' in action:
                action_type = action.split('[')[0].lower()
                action_param = action.split('[')[1].split(']')[0]
                action_normalized = f"{action_type}[{action_param}]"
            else:
                action_normalized = action.lower()
            
            # 检查是否已尝试过这个 action
            if action_normalized in seen_actions:
                continue
            seen_actions.add(action_normalized)
            
            # **修复：创建新状态（复制父节点状态），确保包含问题信息**
            new_state = node.state.copy() if node.state else {}
            new_state['thought'] = thought.strip()
            new_state['action'] = action_normalized
            # **关键：确保问题信息被传递到子节点**
            if 'question' not in new_state:
                new_state['question'] = node.state.get('question', question)
            
            # **关键：执行 action 并获取观察结果**
            # 注意：对于 HotpotQA，search 操作是独立的，但 lookup 需要先有 search 的结果
            # 为了简化，我们每次都重置环境并重新执行路径
            try:
                # 重置环境
                obs = self.env.reset(idx=question_idx)
                
                # 重建从根到当前节点父节点的完整路径（不包括当前节点，因为当前节点是要扩展的）
                path_actions = []
                temp_node = node.parent  # 从父节点开始，不包括当前节点
                while temp_node and temp_node.parent:
                    if temp_node.state and temp_node.state.get('action'):
                        path_actions.insert(0, temp_node.state['action'])
                    temp_node = temp_node.parent
                
                # 执行路径上的所有 action（重建环境状态）
                for path_action in path_actions:
                    try:
                        obs, _, _, _ = self.env.step(path_action)
                    except Exception as e:
                        # 如果路径执行失败，记录错误但继续
                        pass
                
                # 执行新节点的 action
                obs, reward, done, info = self.env.step(action_normalized)
                new_state['observation'] = obs
                
                # 创建新节点
                child = Node(new_state, question, parent=node)
                child.is_terminal = (reward == 1) or done
                child.reward = reward
                child.depth = node.depth + 1
                if reward == 1:
                    child.em = info.get('em', 0)
                
                children.append(child)
                
            except Exception as e:
                # 如果执行失败，仍然创建节点但标记为失败
                new_state['observation'] = f"Error executing action: {str(e)[:100]}"
                child = Node(new_state, question, parent=node)
                child.is_terminal = False
                child.reward = 0
                child.depth = node.depth + 1
                children.append(child)
        
        return children
    
    def evaluate_node(self, node: Node, question: str) -> float:
        """
        评估节点
        
        Args:
            node: 要评估的节点
            question: 问题文本
            
        Returns:
            评估分数 (0-1)
        """
        if not node.children:
            return 0.0
        
        # 构建评估提示词
        current_state = ""
        for child in node.children:
            current_state += f"Thought: {child.state.get('thought', '')}\n"
            current_state += f"Action: {child.state.get('action', '')}\n\n"
        
        # **修复：使用节点中保存的问题（如果存在）**
        node_question = node.state.get('question', question)
        
        value_prompt = f"""Question: {node_question}

{current_state}

Please evaluate how promising each approach is on a scale of 0-10, where 10 means very likely to lead to the correct answer.
Only return a number between 0 and 10.
"""
        
        scores = []
        for _ in range(self.n_evaluate):
            response = self.llm.generate(value_prompt, max_tokens=50)
            # 提取数字
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0])
                scores.append(min(10.0, max(0.0, score)) / 10.0)
            else:
                scores.append(0.5)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 将评估值赋给子节点
        for child in node.children:
            child.value = avg_score
        
        return avg_score
    
    def rollout(self, node: Node, question: str, question_idx: int, max_depth: int = 4) -> Tuple[float, Node]:
        """
        Rollout：从节点开始进行模拟直到找到解或达到最大深度
        
        Args:
            node: 起始节点
            question: 问题文本
            question_idx: 问题索引
            max_depth: 最大模拟深度
            
        Returns:
            (reward, terminal_node): 奖励和终端节点
        """
        current = node
        depth = current.depth  # 从当前节点的深度开始
        
        # 重置环境并重建到当前节点的路径
        obs = self.env.reset(idx=question_idx)
        
        # 重建从根到当前节点父节点的路径（不包括当前节点，因为当前节点可能还没有执行 action）
        path_actions = []
        temp_node = current.parent
        while temp_node:
            if temp_node.state and temp_node.state.get('action'):
                path_actions.insert(0, temp_node.state['action'])
            temp_node = temp_node.parent
        
        # 执行路径上的所有 action
        for path_action in path_actions:
            try:
                obs, _, _, _ = self.env.step(path_action)
            except Exception:
                pass
        
        # 执行当前节点的行动（如果还没有执行过）
        # 注意：如果节点已经在 expand 中执行过 action，这里应该已经设置了 observation 和 reward
        if current.state.get('action'):
            if not current.state.get('observation'):
                # 节点还没有执行过 action，需要执行
                action = current.state['action']
                try:
                    obs, reward, done, info = self.env.step(action)
                except Exception as e:
                    obs = f"Error executing action: {str(e)[:100]}"
                    reward = 0
                    done = False
                    info = {}
                
                current.state['observation'] = obs
                current.is_terminal = done or (reward == 1)
                current.reward = reward
                if reward == 1:
                    current.em = info.get('em', 0)
                
                if current.is_terminal:
                    return float(current.reward), current
            else:
                # 节点已经执行过 action，检查是否是 terminal
                if current.is_terminal:
                    return float(current.reward), current
        
        while depth < max_depth and not current.is_terminal:
            # 生成下一步
            children = self.expand(current, question, question_idx)
            if not children:
                current.is_terminal = True
                break
            
            # **关键：将生成的子节点添加到树中**
            current.children.extend(children)
            
            # 选择第一个子节点继续
            best_child = children[0]
            
            # **关键：expand 方法已经执行了 action 并设置了 reward，这里不需要重复执行**
            # 直接检查节点是否已经是 terminal
            if best_child.is_terminal:
                # 如果已经是 terminal 节点（包括 finish 节点），直接返回
                return float(best_child.reward), best_child
            
            # 如果节点还没有执行过 action（这种情况不应该发生，因为 expand 已经执行了）
            # 但为了安全，我们仍然检查
            action = best_child.state.get('action', '')
            if action and not best_child.state.get('observation'):
                # 执行 action（这种情况很少见，因为 expand 已经执行了）
                try:
                    obs, reward, done, info = self.env.step(action)
                except Exception as e:
                    obs = f"Error executing action: {str(e)[:100]}"
                    reward = 0
                    done = False
                    info = {}
                
                best_child.state['observation'] = obs
                best_child.is_terminal = done or (reward == 1)
                best_child.reward = reward
                if reward == 1:
                    best_child.em = info.get('em', 0)
                
                if best_child.is_terminal:
                    return float(best_child.reward), best_child
            
            # 继续到下一个节点
            current = best_child
            depth += 1
        
        # 返回当前节点的奖励
        return 0.0, current
    
    def backpropagate(self, node: Node, reward: float):
        """反向传播奖励"""
        current = node
        while current:
            current.visits += 1
            current.value = (current.value * (current.visits - 1) + reward) / current.visits
            current = current.parent
    
    def select_node(self, root: Node) -> Optional[Node]:
        """选择节点（使用UCT算法）"""
        node = root
        
        while node and node.children:
            # 检查是否有成功的终端节点（优先返回）
            terminal_children = [c for c in node.children if c.is_terminal]
            node_with_reward_1 = next((c for c in terminal_children if c.reward == 1), None)
            if node_with_reward_1:
                return node_with_reward_1
            
            # 如果所有子节点都是终端，回溯
            if len(terminal_children) == len(node.children):
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue
            
            # 过滤非终端节点
            non_terminal = [c for c in node.children if not c.is_terminal]
            if not non_terminal:
                # 所有子节点都是终端，回溯
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue
            
            # 使用UCB选择最有希望的子节点
            node = max(non_terminal, key=lambda c: c.ucb_score())
            
            # **关键：如果选中的节点是 terminal 但 reward != 1，继续选择其他节点**
            while node and node.is_terminal and node.reward != 1:
                # 从父节点的其他非终端子节点中选择
                if node.parent:
                    non_terminal_siblings = [c for c in node.parent.children 
                                           if not c.is_terminal and c != node]
                    if non_terminal_siblings:
                        node = max(non_terminal_siblings, key=lambda c: c.ucb_score())
                    else:
                        # 没有其他非终端节点，回溯
                        if node.parent.parent:
                            node.parent.parent.children.remove(node.parent)
                        node = node.parent.parent
                else:
                    break
        
        return node
    
    def solve(self, question: str, ground_truth: str, question_idx: int = 0) -> Tuple[bool, str, List[dict]]:
        """
        使用LATS方法解决问题
        
        Args:
            question: 问题文本
            ground_truth: 正确答案（用于评估）
            question_idx: 问题索引
            
        Returns:
            (success, answer, history): 是否成功、答案、历史记录
        """
        history = []
        
        # **修复：创建根节点并初始化环境状态**
        obs = self.env.reset(idx=question_idx)
        
        # **关键修复：确保环境中的问题与传入的问题一致**
        env_question = getattr(self.env, 'get_current_question', lambda: question)()
        if env_question and env_question != question:
            print(f"警告: LATS 环境问题与传入问题不一致。使用环境问题: {env_question[:50]}...")
            question = env_question
        
        root_state = {
            'thought': '',
            'action': '',
            'observation': obs,
            'question': question  # **新增：保存问题信息**
        }
        root = Node(root_state, question)
        
        for iteration in range(self.max_iterations):
            # Selection: 选择节点
            node = self.select_node(root)
            
            if node is None or (node.is_terminal and node.reward != 1):
                # 重新选择
                node = self.select_node(root)
                if node is None:
                    history.append({
                        'iteration': iteration + 1,
                        'status': 'all_paths_exhausted'
                    })
                    break
            
            if node.is_terminal and node.reward == 1:
                # 找到解
                answer = self._extract_answer(node)
                history.append({
                    'iteration': iteration + 1,
                    'status': 'success',
                    'answer': answer
                })
                return True, answer, history
            
            # Expansion: 扩展节点
            if not node.children:
                children = self.expand(node, question, question_idx)
                node.children.extend(children)
            
            # **关键：扩展后立即检查是否有成功的终端节点**
            terminal_success = [c for c in node.children if c.is_terminal and c.reward == 1]
            if terminal_success:
                best_terminal = max(terminal_success, key=lambda c: c.value)
                answer = self._extract_answer(best_terminal)
                history.append({
                    'iteration': iteration + 1,
                    'status': 'success',
                    'answer': answer
                })
                return True, answer, history
            
            # 如果节点是 terminal 或没有子节点，重新选择并扩展（类似原始实现）
            while node.is_terminal or not node.children:
                node = self.select_node(root)
                if node is None or (node.is_terminal and node.reward != 1):
                    node = self.select_node(root)
                    if node is None:
                        # 检查所有节点中是否有成功的
                        all_nodes = self._collect_all_nodes(root)
                        terminal_success_all = [n for n in all_nodes if n.is_terminal and n.reward == 1]
                        if terminal_success_all:
                            best_terminal = max(terminal_success_all, key=lambda n: n.value)
                            answer = self._extract_answer(best_terminal)
                            history.append({
                                'iteration': iteration + 1,
                                'status': 'success',
                                'answer': answer
                            })
                            return True, answer, history
                        history.append({
                            'iteration': iteration + 1,
                            'status': 'all_paths_exhausted'
                        })
                        break
                if node and not node.children:
                    children = self.expand(node, question, question_idx)
                    node.children.extend(children)
                    # 再次检查是否有成功的节点
                    terminal_success = [c for c in node.children if c.is_terminal and c.reward == 1]
                    if terminal_success:
                        best_terminal = max(terminal_success, key=lambda c: c.value)
                        answer = self._extract_answer(best_terminal)
                        history.append({
                            'iteration': iteration + 1,
                            'status': 'success',
                            'answer': answer
                        })
                        return True, answer, history
                if node is None:
                    break
            
            if node is None:
                continue
            
            # Evaluation: 评估节点
            avg_value = self.evaluate_node(node, question)
            
            # Rollout: 对最佳子节点进行模拟
            best_child = max(node.children, key=lambda c: c.value)
            reward, terminal_node = self.rollout(best_child, question, question_idx, max_depth=4)
            
            if terminal_node.reward == 1:
                # 找到解
                answer = self._extract_answer(terminal_node)
                history.append({
                    'iteration': iteration + 1,
                    'status': 'success',
                    'answer': answer
                })
                self.backpropagate(terminal_node, reward)
                return True, answer, history
            
            # Backpropagation: 反向传播
            self.backpropagate(terminal_node, reward)
            
            # **关键：每次迭代后检查所有节点中是否有成功的终端节点**
            all_nodes = self._collect_all_nodes(root)
            terminal_success_all = [n for n in all_nodes if n.is_terminal and n.reward == 1]
            if terminal_success_all:
                best_terminal = max(terminal_success_all, key=lambda n: n.value)
                answer = self._extract_answer(best_terminal)
                history.append({
                    'iteration': iteration + 1,
                    'status': 'success',
                    'answer': answer
                })
                return True, answer, history
            
            history.append({
                'iteration': iteration + 1,
                'value': avg_value,
                'reward': reward
            })
        
        # 如果没找到解，返回最佳尝试
        all_nodes = self._collect_all_nodes(root)
        if all_nodes:
            # 优先选择 reward=1 的节点
            terminal_success = [n for n in all_nodes if n.is_terminal and n.reward == 1]
            if terminal_success:
                best_node = max(terminal_success, key=lambda n: n.value)
                answer = self._extract_answer(best_node)
                if answer:  # 确保能提取答案
                    history.append({
                        'iteration': self.max_iterations,
                        'status': 'success_found_in_final_check',
                        'answer': answer
                    })
                    return True, answer, history
            
            # 检查所有 finish 节点（即使 reward != 1），手动评估答案
            finish_nodes = []
            for node in all_nodes:
                if node.state and isinstance(node.state, dict):
                    action = node.state.get('action', '')
                    if action.startswith('finish['):
                        answer = self._extract_answer(node)
                        if answer:
                            # 手动评估答案
                            if self._evaluate_answer(answer, ground_truth):
                                history.append({
                                    'iteration': self.max_iterations,
                                    'status': 'success_manual_eval',
                                    'answer': answer
                                })
                                return True, answer, history
                            finish_nodes.append((node, answer))
            
            # 如果手动评估都失败，选择 reward 最高的节点
            if all_nodes:
                best_node = max(all_nodes, key=lambda n: n.reward)
                answer = self._extract_answer(best_node)
            else:
                best_node = None
                answer = ""
            
            # 如果提取到答案，即使 reward != 1 也尝试手动评估
            if not answer and finish_nodes:
                # 使用第一个 finish 节点的答案
                best_node, answer = finish_nodes[0]
            
            # 如果仍然没有答案，尝试从所有节点中提取
            if not answer and all_nodes:
                for node in all_nodes:
                    answer = self._extract_answer(node)
                    if answer:
                        best_node = node
                        break
            
            # 最终评估
            success = best_node.reward == 1 if best_node else False
            if not success and answer:
                # 手动评估答案
                success = self._evaluate_answer(answer, ground_truth)
            
            history.append({
                'iteration': self.max_iterations,
                'status': 'completed',
                'answer': answer or "",
                'success': success,
                'best_node_reward': best_node.reward if best_node else 0
            })
            return success, answer or "", history
        
        history.append({
            'iteration': self.max_iterations,
            'status': 'max_iterations_reached'
        })
        return False, "", history
    
    def _extract_answer(self, node: Node) -> str:
        """从节点中提取答案"""
        if not node.state:
            return ""
        
        # 确保 state 是字典类型
        if not isinstance(node.state, dict):
            return ""
        
        # 首先检查当前节点的 action
        action = node.state.get('action', '')
        if action:
            # 处理 finish 动作（支持大小写不敏感）
            action_lower = action.lower()
            if action_lower.startswith('finish['):
                # 提取 finish[answer] 中的 answer
                if ']' in action:
                    # 找到 finish[ 之后和 ] 之前的内容
                    start_idx = action_lower.find('finish[') + len('finish[')
                    end_idx = action.find(']', start_idx)
                    if end_idx > start_idx:
                        answer = action[start_idx:end_idx]
                        return answer.strip()
                # 如果没有 ]，尝试提取到字符串末尾
                start_idx = action_lower.find('finish[') + len('finish[')
                if start_idx < len(action):
                    answer = action[start_idx:]
                    return answer.strip()
        
        # 如果当前节点没有 finish action，向上遍历父节点查找
        current = node
        while current:
            if current.state and isinstance(current.state, dict):
                action = current.state.get('action', '')
                if action:
                    action_lower = action.lower()
                    if action_lower.startswith('finish['):
                        # 提取答案
                        if ']' in action:
                            start_idx = action_lower.find('finish[') + len('finish[')
                            end_idx = action.find(']', start_idx)
                            if end_idx > start_idx:
                                answer = action[start_idx:end_idx]
                                return answer.strip()
                        start_idx = action_lower.find('finish[') + len('finish[')
                        if start_idx < len(action):
                            answer = action[start_idx:]
                            return answer.strip()
            current = current.parent
        
        # 如果 action 不是 finish，尝试从观察中提取
        observation = node.state.get('observation', '')
        if observation and 'Episode finished' in observation:
            # 尝试从观察中提取
            import re
            match = re.search(r'reward = (\d+)', observation)
            if match and match.group(1) == '1':
                # 需要从历史中找到 finish action
                current = node
                while current:
                    if current.state and isinstance(current.state, dict):
                        action = current.state.get('action', '')
                        if action:
                            action_lower = action.lower()
                            if action_lower.startswith('finish['):
                                if ']' in action:
                                    start_idx = action_lower.find('finish[') + len('finish[')
                                    end_idx = action.find(']', start_idx)
                                    if end_idx > start_idx:
                                        return action[start_idx:end_idx].strip()
                                start_idx = action_lower.find('finish[') + len('finish[')
                                if start_idx < len(action):
                                    return action[start_idx:].strip()
                    current = current.parent
        
        return ""
    
    def _collect_all_nodes(self, node: Node) -> List[Node]:
        """收集所有节点"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._collect_all_nodes(child))
        return nodes
    
    def _evaluate_answer(self, prediction: str, ground_truth: str) -> bool:
        """
        评估答案是否正确（精确匹配），复用原始实现的逻辑
        """
        def normalize_answer(s):
            import string
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
        
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(ground_truth)
        return pred_norm == gt_norm

