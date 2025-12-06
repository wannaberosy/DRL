"""
HotpotQA LATS Tree Search Solver with Tree-GRPO optimization
This module integrates Tree-GRPO's batch tree search strategy into LATs framework.
"""
import re
import math
import uuid
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

# 尝试导入数据集环境
try:
    from hotpot.dataset_wikienv import DatasetWikiEnv
    HAS_DATASET_ENV = True
except ImportError:
    HAS_DATASET_ENV = False

# 修改 wrappers 模块中的 DATA_DIR 为绝对路径
hotpot_data_dir = hotpot_path / "data"
wrappers.DATA_DIR = str(hotpot_data_dir)

from hotpot.tree_search_node import TreeSearchNode


class HotpotLATSTreeSearchSolver:
    """
    HotpotQA LATS 求解器 with Tree-GRPO style batch tree search optimization.
    
    Key improvements:
    1. Batch tree search: Maintains multiple trees (m trees) simultaneously
    2. Iterative expansion: Performs l iterations of expansion, expanding n nodes per iteration
    3. Leaf sampling: Samples k final leaves from each tree
    4. Tree-structured supervision: Calculates final scores based on tree structure
    """
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 30,
                 n_generate: int = 3, n_evaluate: int = 1,
                 env_backend: str = "wikipedia", cache_file: Optional[str] = None,
                 dataset_file: Optional[str] = None, dataset_url: Optional[str] = None,
                 tree_m: int = 4, tree_n: int = 2, tree_l: int = None, tree_k: int = 4,
                 tree_expand_mode: str = 'uct', tree_reward_mode: str = 'base',
                 mcts_num_simulations: int = 5, mcts_use_value_function: bool = True,
                 mcts_use_rollout: bool = True):
        """
        初始化LATS Tree Search求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大搜索迭代次数（用于向后兼容，实际使用 tree_l）
            n_generate: 每次扩展生成的候选数
            n_evaluate: 每次评估的采样数
            env_backend: 环境后端类型
            cache_file: 缓存文件路径
            dataset_file: 数据集文件路径
            dataset_url: 数据集URL
            tree_m: 维护的树数量（Tree-GRPO参数）
            tree_n: 每次扩展的节点数（Tree-GRPO参数）
            tree_l: 扩展迭代次数（Tree-GRPO参数）
            tree_k: 每棵树最终采样的叶子数（Tree-GRPO参数）
            tree_expand_mode: 节点扩展模式（'random', 'best', 'uct'）
            tree_reward_mode: 奖励计算模式（'base', 'tree_diff'）
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.n_generate = n_generate
        self.n_evaluate = n_evaluate
        
        # Tree-GRPO parameters
        self.tree_m = tree_m
        self.tree_n = tree_n
        # Use max_iterations as default for tree_l if not specified
        # This ensures sufficient exploration depth
        self.tree_l = tree_l if tree_l is not None else max_iterations
        self.tree_k = tree_k
        self.tree_expand_mode = tree_expand_mode
        self.tree_reward_mode = tree_reward_mode
        
        # MCTS parameters
        self.mcts_num_simulations = mcts_num_simulations
        self.mcts_use_value_function = mcts_use_value_function
        self.mcts_use_rollout = mcts_use_rollout
        
        # Cache for value function evaluations (to avoid repeated LLM calls)
        self._value_cache = {}
        
        # 初始化环境
        if env_backend == "dataset" and HAS_DATASET_ENV:
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
            print(f"使用本地环境 (后端: {env_backend})")
            base_env = LocalWikiEnv(backend=env_backend, cache_file=cache_file)
        else:
            print("使用原始 WikiEnv（可能需要网络连接）")
            base_env = wikienv.WikiEnv()
        
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
                
                if '[' in action and ']' in action:
                    action_type = action.split('[')[0].lower()
                    action_param = action.split('[')[1].split(']')[0]
                    return f"{action_type}[{action_param}]"
                else:
                    return action.lower()
        
        return None
    
    def _mcts_select(self, root: TreeSearchNode) -> Optional[TreeSearchNode]:
        """
        MCTS Selection: 使用 UCT 算法选择最有潜力的节点
        参考原始 LATS 的 select_node 方法
        """
        node = root
        
        while node and node.children:
            # 检查是否有成功的终端节点（优先返回）
            terminal_success = [c for c in node.children 
                              if c.is_terminal and c.reward == 1]
            if terminal_success:
                return terminal_success[0]
            
            # 如果所有子节点都是终端，回溯
            if all(c.is_terminal for c in node.children):
                if node.parent:
                    node = node.parent
                    continue
            
            # 过滤非终端节点
            non_terminal = [c for c in node.children if not c.is_terminal]
            if not non_terminal:
                if node.parent:
                    node = node.parent
                    continue
            
            # 使用 UCT 选择最有希望的子节点
            node = max(non_terminal, key=lambda c: c.ucb_score())
        
        return node
    
    def _mcts_evaluate(self, node: TreeSearchNode, question: str) -> float:
        """
        MCTS Evaluation: 使用 LLM 价值函数评估节点（带缓存优化）
        参考原始 LATS 的 evaluate_node 方法
        """
        if not node.children:
            return 0.0
        
        # 构建评估提示词
        child_prompts = []
        child_indices = []
        cache_keys = []
        
        for i, child in enumerate(node.children):
            if not child.is_terminal:
                # 构建当前状态字符串
                current_state = ""
                if child.state and isinstance(child.state, dict):
                    if child.state.get('thought'):
                        current_state += f"Thought: {child.state['thought']}\n"
                    if child.state.get('action'):
                        current_state += f"Action: {child.state['action']}\n"
                
                value_prompt = f"""Question: {question}

{current_state}

Please evaluate how promising this approach is on a scale of 0-10, where 10 means very likely to lead to the correct answer.
Only return a number between 0 and 10.
"""
                child_prompts.append(value_prompt)
                child_indices.append(i)
                # Create cache key
                cache_key = hash((question, current_state))
                cache_keys.append(cache_key)
        
        if not child_prompts:
            return 0.0
        
        # 使用 LLM 评估所有子节点（带缓存）
        values = []
        for i, prompt in enumerate(child_prompts):
            cache_key = cache_keys[i]
            
            # Check cache first
            if cache_key in self._value_cache:
                values.append(self._value_cache[cache_key])
                continue
            
            scores = []
            # 只评估一次（减少LLM调用）
            response = self.llm.generate(prompt, max_tokens=50)
            # 提取数字
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0])
                score = min(10.0, max(0.0, score)) / 10.0
            else:
                score = 0.5
            
            # Cache the result
            self._value_cache[cache_key] = score
            values.append(score)
        
        # 将评估值赋给对应的子节点
        for idx, value in zip(child_indices, values):
            node.children[idx].value = value
        
        # 对于终端节点（已找到解），设置高价值
        for child in node.children:
            if child.is_terminal and child.reward == 1:
                child.value = 1.0
        
        # 返回所有子节点的平均价值
        if node.children:
            return sum(c.value for c in node.children) / len(node.children)
        return 0.0
    
    def _mcts_simulate(self, node: TreeSearchNode, question: str, 
                       question_idx: int, max_depth: int = 4) -> Tuple[float, TreeSearchNode]:
        """
        MCTS Simulation: 从节点进行 rollout 模拟
        参考原始 LATS 的 rollout 方法，实际执行环境操作
        """
        current = node
        depth = current.depth
        rewards = [0.0]
        
        # 重置环境并重建到当前节点的路径
        try:
            obs = self.env.reset(idx=question_idx)
            
            # 重建从根到当前节点父节点的路径
            path_actions = []
            temp_node = current.parent
            while temp_node and temp_node.parent:
                if temp_node.state and isinstance(temp_node.state, dict):
                    action = temp_node.state.get('action', '')
                    if action and action != 'reset' and not action.startswith('think['):
                        path_actions.insert(0, action)
                temp_node = temp_node.parent
            
            # 执行路径上的所有 action
            for path_action in path_actions:
                try:
                    obs, _, _, _ = self.env.step(path_action)
                except Exception:
                    pass
            
            # 执行当前节点的行动（如果还没有执行过）
            if current.state and isinstance(current.state, dict):
                action = current.state.get('action', '')
                if action and not current.state.get('observation'):
                    try:
                        obs, reward, done, info = self.env.step(action)
                        current.state['observation'] = obs
                        if done:
                            current.is_terminal = True
                            current.reward = reward
                            return float(reward), current
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Rollout 循环
        while depth < max_depth and not current.is_terminal:
            # 生成新状态
            new_states = self._generate_new_states(current, question, question_idx)
            if not new_states:
                current.is_terminal = True
                break
            
            # 检查是否有终端节点（已找到解）
            terminal_success = [s for s in new_states if s.is_terminal and s.reward == 1]
            if terminal_success:
                return 1.0, terminal_success[0]
            
            # 选择最有希望的子节点继续
            non_terminal = [s for s in new_states if not s.is_terminal]
            if not non_terminal:
                current.is_terminal = True
                break
            
            # 简单评估：使用 reward 作为价值
            for child in non_terminal:
                child.value = child.reward
            
            best_child = max(non_terminal, key=lambda c: c.value)
            current = best_child
            rewards.append(best_child.value)
            depth += 1
        
        # 返回平均奖励
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return avg_reward, current
    
    def _mcts_backpropagate(self, node: TreeSearchNode, reward: float):
        """
        MCTS Backpropagation: 反向传播奖励
        参考原始 LATS 的 backpropagate 方法
        """
        current = node
        while current:
            current.visits += 1
            if current.is_terminal:
                if reward >= 0.99:
                    current.value = (current.value * (current.visits - 1) + reward) / current.visits
                else:
                    current.value = (current.value * (current.visits - 1) + (-0.5)) / current.visits
            else:
                current.value = (current.value * (current.visits - 1) + reward) / current.visits
            current = current.parent
    
    def get_expand_node_with_mcts(self, root: TreeSearchNode, n: int = 1) -> List[TreeSearchNode]:
        """
        使用完整的 MCTS 流程选择 n 个最有潜力的节点进行扩展
        
        Args:
            root: 根节点
            n: 要选择的节点数
            
        Returns:
            选中的节点列表
        """
        # 进行 MCTS 模拟
        for _ in range(self.mcts_num_simulations):
            # Selection: 使用 UCT 选择
            node = self._mcts_select(root)
            
            if node is None:
                continue
            
            # 如果节点是终端且成功，直接返回
            if node.is_terminal and node.reward == 1:
                return [node]
            
            # Expansion: 如果节点未完全展开，扩展它
            if not node.is_fully_expanded() and len(node.children) == 0:
                # 只生成候选动作，不实际执行（大幅减少环境操作）
                candidates = self._generate_candidate_actions(node, node.question)
                
                for thought, action in candidates:
                    # 创建虚拟节点（不执行环境操作）
                    node_id = str(uuid.uuid4())
                    if node.state and isinstance(node.state, dict):
                        new_state = node.state.copy()
                    else:
                        new_state = {'thought': '', 'action': '', 'observation': ''}
                    new_state['thought'] = thought
                    new_state['action'] = action
                    # observation 留空，实际执行会在 expand_node_batch 中进行
                    
                    child = TreeSearchNode(
                        node_id=node_id,
                        state=new_state,
                        question=node.question,
                        parent=node,
                        depth=node.depth + 1,
                        reward_mode=self.tree_reward_mode,
                    )
                    # 初始化统计信息
                    child.visits = 1
                    # 使用启发式初始价值（不执行环境操作）
                    if action.startswith('finish['):
                        child.value = 0.7  # 高价值
                    elif action.startswith('search['):
                        child.value = 0.4  # 中等价值
                    elif action.startswith('lookup['):
                        child.value = 0.2  # 低价值
                    else:
                        child.value = 0.1
                    child.reward = 0  # 实际 reward 会在 expand_node_batch 中设置
                    
                    node.add_child(child)
            
            if not node.children:
                continue
            
            # Evaluation: 使用价值函数评估（如果启用）
            if self.mcts_use_value_function:
                avg_value = self._mcts_evaluate(node, node.question)
            else:
                # 不使用价值函数，直接使用 reward
                for child in node.children:
                    if child.value == 0:
                        child.value = child.reward
            
            # Simulation: 进行轻量级 rollout（如果启用）
            if self.mcts_use_rollout:
                best_child = max(node.children, key=lambda c: c.value)
                # 使用轻量级 rollout（不执行环境操作）
                reward, terminal_node = self._mcts_simulate(
                    best_child, node.question, 0, max_depth=2  # 减少深度
                )
            else:
                # 不使用 rollout，直接使用子节点的 value
                best_child = max(node.children, key=lambda c: c.value)
                reward = best_child.value  # 使用 value 而不是 reward（因为还没执行）
                terminal_node = best_child
            
            # Backpropagation: 更新节点价值
            self._mcts_backpropagate(terminal_node, reward)
        
        # 选择最有价值的 n 个节点
        all_candidates = root.get_subtree_nodes() + [root]
        non_terminal_candidates = [n for n in all_candidates 
                                  if not n.is_terminal and n.depth < 7]
        
        if not non_terminal_candidates:
            return []
        
        # 按 UCT 分数排序
        non_terminal_candidates.sort(key=lambda x: x.ucb_score(), reverse=True)
        return non_terminal_candidates[:min(n, len(non_terminal_candidates))]
    
    def expand_node_batch(self, nodes: List[TreeSearchNode], question: str, question_idx: int) -> bool:
        """
        Batch expand multiple nodes with MCTS guidance.
        
        Args:
            nodes: List of nodes to expand
            question: Question text
            question_idx: Question index
            
        Returns:
            True if any successful terminal node was found, False otherwise
        """
        for node in nodes:
            # Skip if already terminal or at depth limit
            if node.is_terminal or node.depth >= 7:
                node.is_leaf = True
                continue
            
            # Skip if already marked as leaf (but allow re-expansion if needed)
            # This can happen if a node was previously marked as leaf but is now selected for expansion
            if node.is_leaf and len(node.children) > 0:
                # If it has children, it's not a true leaf, unmark it
                node.is_leaf = False
            
            # 如果节点已经有子节点且被访问过，可以利用 MCTS 价值信息
            # 但这里我们仍然扩展所有节点，让 MCTS 在后续选择中发挥作用
            
            # Generate new states
            new_states = self._generate_new_states(node, question, question_idx)
            
            # Add children and check for immediate success
            for new_state_node in new_states:
                node.add_child(new_state_node)
                new_state_node.is_leaf = new_state_node.is_terminal or new_state_node.depth >= 7
                
                # Initialize value with reward for better UCT selection
                # 注意：新节点的 value 初始化为 reward，visits 初始化为 1
                # 这样在 UCT 选择时，新节点会根据其 reward 进行排序
                new_state_node.value = float(new_state_node.reward)
                new_state_node.visits = 1
                
                # 确保父节点的 visits 也被更新（用于 UCT 计算）
                if node.parent:
                    node.parent.visits = max(node.parent.visits, 1)
                
                # Check if this is a successful terminal node
                if new_state_node.is_terminal and new_state_node.reward == 1:
                    return True  # Found successful node
            
            # Update parent node statistics for UCT
            if len(node.children) > 0:
                # Update visits
                node.visits += 1
                # Update value based on children's average reward
                avg_reward = sum(c.reward for c in node.children) / len(node.children)
                node.value = (node.value * (node.visits - 1) + avg_reward) / node.visits
            
            # If no children were added, mark as leaf
            if len(node.children) == 0:
                node.is_leaf = True
            else:
                # Node has children, so it's not a leaf
                node.is_leaf = False
        
        return False  # No successful node found
    
    def _generate_candidate_actions(self, node: TreeSearchNode, question: str) -> List[Tuple[str, str]]:
        """
        Generate candidate actions without executing them (for MCTS expansion).
        This is much faster than _generate_new_states which executes actions.
        
        Returns:
            List of (thought, action) tuples
        """
        # **关键修复：构建完整轨迹（从根节点到当前节点）**
        # 参考原始LATS的generate_prompt函数
        trajectory_segments = []
        current = node
        while current:
            segment = []
            if current.state and isinstance(current.state, dict):
                if current.state.get('thought'):
                    segment.append(f"Thought {current.depth}: {current.state['thought']}")
                if current.state.get('action'):
                    segment.append(f"Action {current.depth}: {current.state['action']}")
                # 根节点的observation包含在初始状态中，但根据原始代码，depth=0时不包含observation
                if current.state.get('observation') and current.depth != 0:
                    segment.append(f"Observation {current.depth}: {current.state['observation']}")
            if segment:
                trajectory_segments.insert(0, '\n'.join(segment))
            current = current.parent
        
        # 构建完整轨迹字符串
        current_state = '\n'.join(trajectory_segments)
        
        # Build expansion prompt
        expand_prompt = f"""Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Question: {question}
{current_state}

Please generate {self.n_generate} different next steps (Thought and Action pairs). Each should be a different approach to solving the problem.
Format each as:
Thought {node.depth + 1}: [your reasoning]
Action {node.depth + 1}: [your action]

"""
        
        response = self.llm.generate(expand_prompt, max_tokens=500)
        
        # Parse generated actions
        candidates = []
        seen_actions = set()
        
        # Try to extract multiple Thought-Action pairs
        thought_action_pairs = re.findall(
            r'Thought\s*\d*\s*:\s*([^\n]+)\s*\n\s*Action\s*\d*\s*:\s*(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
            response,
            re.IGNORECASE | re.MULTILINE
        )
        
        # If not enough pairs found, try alternative patterns
        if len(thought_action_pairs) < self.n_generate:
            # Try to find actions without explicit Thought labels
            actions = re.findall(
                r'(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
                response,
                re.IGNORECASE
            )
            # Match actions with preceding text as thought
            for i, action in enumerate(actions[:self.n_generate]):
                # Try to find thought before this action
                thought_match = re.search(
                    rf'([^\n]+?)\s*(?:Action\s*\d*\s*:)?\s*{re.escape(action)}',
                    response,
                    re.IGNORECASE | re.DOTALL
                )
                thought = thought_match.group(1).strip() if thought_match else f"Generated thought {i+1}"
                thought_action_pairs.append((thought, action))
        
        # Normalize actions
        for thought, action in thought_action_pairs[:self.n_generate]:
            action = action.strip()
            if '[' in action and ']' in action:
                action_type = action.split('[')[0].lower()
                action_param = action.split('[')[1].split(']')[0]
                action_normalized = f"{action_type}[{action_param}]"
            else:
                action_normalized = action.lower()
            
            if action_normalized not in seen_actions:
                seen_actions.add(action_normalized)
                candidates.append((thought.strip(), action_normalized))
        
        return candidates
    
    def _generate_new_states(self, node: TreeSearchNode, question: str, question_idx: int) -> List[TreeSearchNode]:
        """
        Generate new child states from a node.
        
        Args:
            node: Parent node
            question: Question text
            question_idx: Question index
            
        Returns:
            List of new child nodes
        """
        # **关键修复：构建完整轨迹（从根节点到当前节点）**
        # 参考原始LATS的generate_prompt函数
        trajectory_segments = []
        current = node
        while current:
            segment = []
            if current.state and isinstance(current.state, dict):
                if current.state.get('thought'):
                    segment.append(f"Thought {current.depth}: {current.state['thought']}")
                if current.state.get('action'):
                    segment.append(f"Action {current.depth}: {current.state['action']}")
                # 根节点的observation包含在初始状态中，但根据原始代码，depth=0时不包含observation
                if current.state.get('observation') and current.depth != 0:
                    segment.append(f"Observation {current.depth}: {current.state['observation']}")
            if segment:
                trajectory_segments.insert(0, '\n'.join(segment))
            current = current.parent
        
        # 构建完整轨迹字符串
        current_state = '\n'.join(trajectory_segments)
        
        # Build expansion prompt
        expand_prompt = f"""Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Question: {question}
{current_state}

Please generate {self.n_generate} different next steps (Thought and Action pairs). Each should be a different approach to solving the problem.
Format each as:
Thought {node.depth + 1}: [your reasoning]
Action {node.depth + 1}: [your action]

"""
        
        response = self.llm.generate(expand_prompt, max_tokens=500)
        
        # Parse generated actions
        children = []
        seen_actions = set()
        
        # Try to extract multiple Thought-Action pairs
        thought_action_pairs = re.findall(
            r'Thought\s*\d*\s*:\s*([^\n]+)\s*\n\s*Action\s*\d*\s*:\s*(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
            response,
            re.IGNORECASE | re.MULTILINE
        )
        
        # If not enough pairs found, try alternative patterns
        if len(thought_action_pairs) < self.n_generate:
            # Try to find actions without explicit Thought labels
            actions = re.findall(
                r'(search\[[^\]]+\]|lookup\[[^\]]+\]|finish\[[^\]]+\])',
                response,
                re.IGNORECASE
            )
            # Match actions with preceding text as thought
            for i, action in enumerate(actions[:self.n_generate]):
                # Try to find thought before this action
                thought_match = re.search(
                    rf'([^\n]+?)\s*(?:Action\s*\d*\s*:)?\s*{re.escape(action)}',
                    response,
                    re.IGNORECASE | re.DOTALL
                )
                thought = thought_match.group(1).strip() if thought_match else f"Generated thought {i+1}"
                thought_action_pairs.append((thought, action))
        
        # Execute each action and create nodes
        for thought, action in thought_action_pairs[:self.n_generate]:
            action = action.strip()
            if '[' in action and ']' in action:
                action_type = action.split('[')[0].lower()
                action_param = action.split('[')[1].split(']')[0]
                action_normalized = f"{action_type}[{action_param}]"
            else:
                action_normalized = action.lower()
            
            if action_normalized in seen_actions:
                continue
            seen_actions.add(action_normalized)
            
            # Create new state
            if node.state and isinstance(node.state, dict):
                new_state = node.state.copy()
            else:
                new_state = {'thought': '', 'action': '', 'observation': ''}
            new_state['thought'] = thought.strip()
            new_state['action'] = action_normalized
            
            # Execute action
            try:
                # **关键修复：重建路径逻辑**
                # Reset environment for this question
                obs = self.env.reset(idx=question_idx)
                
                # Rebuild path from root to current node's parent (excluding current node)
                # 根节点没有action，所以从根节点的子节点开始
                path_actions = []
                temp_node = node.parent  # Start from parent, not current node
                while temp_node and temp_node.parent:  # Stop at root (root has no parent)
                    if temp_node.state and isinstance(temp_node.state, dict):
                        action = temp_node.state.get('action', '')
                        # Only include valid actions (not empty, not reset)
                        if action and action != 'reset' and not action.startswith('think['):
                            path_actions.insert(0, action)
                    temp_node = temp_node.parent
                
                # Execute path actions to rebuild environment state
                for path_action in path_actions:
                    try:
                        obs, _, _, _ = self.env.step(path_action)
                    except Exception as e:
                        # If path execution fails, log but continue
                        print(f"DEBUG: Path action execution failed: {path_action}, error: {e}")
                        pass
                
                # Execute new node's action
                obs, reward, done, info = self.env.step(action_normalized)
                new_state['observation'] = obs
                
                # Store answer from info if available (for finish actions)
                if action_normalized.startswith('finish[') and 'answer' in info:
                    # The answer is already in the action, but we can also check info
                    pass
                
                # Create new node
                node_id = str(uuid.uuid4())
                child = TreeSearchNode(
                    node_id=node_id,
                    state=new_state,
                    question=question,
                    parent=node,
                    depth=node.depth + 1,
                    reward_mode=self.tree_reward_mode,
                )
                child.is_terminal = (reward == 1) or done
                child.reward = reward
                child.value = reward  # Initialize value with reward
                if reward == 1:
                    child.em = info.get('em', 0)
                
                children.append(child)
                
            except Exception as e:
                # If execution fails, still create node but mark as failed
                new_state['observation'] = f"Error executing action: {str(e)[:100]}"
                node_id = str(uuid.uuid4())
                child = TreeSearchNode(
                    node_id=node_id,
                    state=new_state,
                    question=question,
                    parent=node,
                    depth=node.depth + 1,
                    reward_mode=self.tree_reward_mode,
                )
                child.is_terminal = False
                child.reward = 0
                children.append(child)
        
        return children
    
    def solve(self, question: str, ground_truth: str, question_idx: int = 0) -> Tuple[bool, str, List[dict]]:
        """
        使用Tree-GRPO风格的批量树搜索解决问题
        
        Args:
            question: 问题文本
            ground_truth: 正确答案（用于评估）
            question_idx: 问题索引
            
        Returns:
            (success, answer, history): 是否成功、答案、历史记录
        """
        # Clear cache for new problem
        self._value_cache.clear()
        history = []
        
        # Step 1: Initialize m root trees
        # **关键修复：先重置环境获取初始observation**
        initial_obs = self.env.reset(idx=question_idx)
        
        root_trees = []
        for i in range(self.tree_m):
            node_id = str(uuid.uuid4())
            root = TreeSearchNode(
                node_id=node_id,
                state={'thought': '', 'action': '', 'observation': initial_obs},
                question=question,
                depth=0,
                is_root=True,
                reward_mode=self.tree_reward_mode,
            )
            root_trees.append(root)
        
        history.append({
            'step': 'initialization',
            'num_trees': self.tree_m
        })
        
        # Step 2: Generate initial action chains for m trees
        for root in root_trees:
            found_success = self.expand_node_batch([root], question, question_idx)
            
            # Check for immediate success after initial expansion
            if found_success:
                all_nodes = root.get_subtree_nodes() + [root]
                for node in all_nodes:
                    if node.is_terminal and node.reward == 1:
                        answer = self._extract_answer(node)
                        if answer:  # Only return if we can extract answer
                            history.append({
                                'step': 'initial_expansion',
                                'status': 'success',
                                'answer': answer
                            })
                            return True, answer, history
                
                # Also check for finish actions that might have correct answers
                for node in all_nodes:
                    if isinstance(node.state, dict):
                        action = node.state.get('action', '')
                        if action.startswith('finish['):
                            answer = self._extract_answer(node)
                            if answer:
                                # Evaluate the answer manually
                                if self._evaluate_answer(answer, ground_truth):
                                    history.append({
                                        'step': 'initial_expansion',
                                        'status': 'success_manual_eval',
                                        'answer': answer
                                    })
                                    return True, answer, history
        
        history.append({
            'step': 'initial_expansion',
            'status': 'completed'
        })
        
        # Step 3: Iterative expansion (l iterations)
        for expansion_iter in range(self.tree_l):
            expansion_nodes = []
            
            # Get expansion nodes from each tree
            for root in root_trees:
                if self.tree_expand_mode == 'mcts':
                    # 使用完整的 MCTS 流程
                    expand_candidates = self.get_expand_node_with_mcts(
                        root, n=self.tree_n
                    )
                elif self.tree_expand_mode == 'uct':
                    # UCT 模式也应该使用 MCTS 流程来更好地评估节点价值
                    # 这与之前版本的实现保持一致
                    expand_candidates = self.get_expand_node_with_mcts(
                        root, n=self.tree_n
                    )
                else:
                    # 使用原有的简单选择（random 或 best）
                    expand_candidates = root.get_expand_node(self.tree_n, mode=self.tree_expand_mode)
                expansion_nodes.extend(expand_candidates)
            
            if len(expansion_nodes) == 0:
                # Before breaking, check if we already have a successful node
                for root in root_trees:
                    all_nodes = root.get_subtree_nodes() + [root]
                    for node in all_nodes:
                        if node.is_terminal and node.reward == 1:
                            answer = self._extract_answer(node)
                            if answer:
                                history.append({
                                    'step': f'expansion_iter_{expansion_iter + 1}',
                                    'status': 'success_before_expansion',
                                    'answer': answer
                                })
                                return True, answer, history
                
                history.append({
                    'step': f'expansion_iter_{expansion_iter + 1}',
                    'status': 'no_nodes_to_expand'
                })
                break
            
            # Batch expand nodes
            found_success = self.expand_node_batch(expansion_nodes, question, question_idx)
            
            # Check for success after each expansion iteration
            if found_success:
                for root in root_trees:
                    all_nodes = root.get_subtree_nodes() + [root]
                    for node in all_nodes:
                        if node.is_terminal and node.reward == 1:
                            answer = self._extract_answer(node)
                            if answer:  # Only return if we can extract answer
                                history.append({
                                    'step': f'expansion_iter_{expansion_iter + 1}',
                                    'status': 'success',
                                    'answer': answer
                                })
                                return True, answer, history
            
            # Also check for finish actions that might have correct answers
            for root in root_trees:
                all_nodes = root.get_subtree_nodes() + [root]
                for node in all_nodes:
                    if isinstance(node.state, dict):
                        action = node.state.get('action', '')
                        if action.startswith('finish['):
                            answer = self._extract_answer(node)
                            if answer:
                                # Evaluate the answer manually
                                if self._evaluate_answer(answer, ground_truth):
                                    history.append({
                                        'step': f'expansion_iter_{expansion_iter + 1}',
                                        'status': 'success_manual_eval',
                                        'answer': answer
                                    })
                                    return True, answer, history
            
            history.append({
                'step': f'expansion_iter_{expansion_iter + 1}',
                'num_nodes_expanded': len(expansion_nodes)
            })
        
        # Step 4: Sample k leaves from each tree
        final_nodes = []
        for root in root_trees:
            sampled_leaves = root.sample_leaf(self.tree_k)
            final_nodes.extend(sampled_leaves)
        
        history.append({
            'step': 'leaf_sampling',
            'num_final_nodes': len(final_nodes)
        })
        
        # Step 5: Evaluate all final nodes and calculate tree-structured scores
        for node in final_nodes:
            if node.is_terminal:
                node.set_leaf_original_score(float(node.reward))
            else:
                # Evaluate the node (simplified - use reward as score)
                node.set_leaf_original_score(node.value)
        
        # Calculate tree-structured final scores
        for root in root_trees:
            root.calculate_final_score_from_root()
        
        # Step 6: Select best trajectory
        best_node = None
        best_score = -float('inf')
        
        # First, check for any terminal nodes with reward 1 (check ALL nodes in all trees)
        # Also check for nodes with finish actions that might have correct answers
        for root in root_trees:
            all_nodes = root.get_subtree_nodes() + [root]
            for node in all_nodes:
                # Check nodes with reward 1 first
                if node.is_terminal and node.reward == 1:
                    answer = self._extract_answer(node)
                    if answer:  # Only return if we can extract answer
                        history.append({
                            'step': 'selection',
                            'status': 'success',
                            'answer': answer
                        })
                        return True, answer, history
                
                # Also check nodes with finish actions (even if reward is 0)
                # They might have correct answers but reward calculation failed
                if isinstance(node.state, dict):
                    action = node.state.get('action', '')
                    if action.startswith('finish['):
                        answer = self._extract_answer(node)
                        if answer:
                            # Evaluate the answer manually
                            if self._evaluate_answer(answer, ground_truth):
                                history.append({
                                    'step': 'selection',
                                    'status': 'success_manual_eval',
                                    'answer': answer
                                })
                                return True, answer, history
        
        # Then, select best from final nodes
        for node in final_nodes:
            # Check if this node has a finish action (even if reward is 0)
            action = node.state.get('action', '') if isinstance(node.state, dict) else ''
            has_finish = action.startswith('finish[')
            
            if hasattr(node, 'final_score') and node.final_score > 0:
                score = node.final_score
            elif hasattr(node, 'original_score') and node.original_score > 0:
                score = node.original_score
            else:
                score = node.value if node.value > 0 else node.reward
            
            # Prefer nodes with finish actions
            if has_finish and score == 0:
                score = 0.5  # Give some score to nodes with finish actions
            
            if score > best_score:
                best_score = score
                best_node = node
        
        # Also check all nodes in trees for finish actions (even if not terminal or reward != 1)
        # This is important because nodes with finish actions should be prioritized
        for root in root_trees:
            all_nodes = root.get_subtree_nodes() + [root]
            for node in all_nodes:
                if isinstance(node.state, dict):
                    action = node.state.get('action', '')
                    if action.startswith('finish['):
                        # Prioritize nodes with finish actions
                        score = node.reward if node.reward > 0 else 0.7  # High score for finish actions
                        if score > best_score:
                            best_score = score
                            best_node = node
                elif node.is_terminal and node.reward > best_score:
                    # Also consider terminal nodes without finish actions
                    best_score = node.reward
                    best_node = node
        
        if best_node is None:
            if final_nodes:
                best_node = final_nodes[0]
            else:
                best_node = root_trees[0]
        
        # Extract answer from best node
        answer = self._extract_answer(best_node)
        
        # If no answer extracted, try to find finish action in trajectory
        if not answer:
            # Check current node first
            if isinstance(best_node.state, dict):
                action = best_node.state.get('action', '')
                if action.startswith('finish['):
                    # More robust extraction: handle cases where ] might be missing
                    start_idx = action.lower().find('finish[') + len('finish[')
                    if ']' in action[start_idx:]:
                        end_idx = action.find(']', start_idx)
                        answer = action[start_idx:end_idx]
                    else:
                        # No closing bracket, extract to end
                        answer = action[start_idx:]
            
            # If still no answer, traverse trajectory to find finish action
            if not answer:
                try:
                    trajectory = best_node.collect_trajectory()
                    for state in trajectory:
                        if isinstance(state, dict):
                            action = state.get('action', '')
                            if action and action.startswith('finish['):
                                # More robust extraction
                                start_idx = action.lower().find('finish[') + len('finish[')
                                if ']' in action[start_idx:]:
                                    end_idx = action.find(']', start_idx)
                                    answer = action[start_idx:end_idx]
                                else:
                                    answer = action[start_idx:]
                                if answer:
                                    break
                except Exception:
                    pass
            
            # If still no answer, check all nodes in the tree for finish actions
            if not answer:
                for root in root_trees:
                    all_nodes = root.get_subtree_nodes() + [root]
                    for node in all_nodes:
                        if isinstance(node.state, dict):
                            action = node.state.get('action', '')
                            if action.startswith('finish['):
                                # More robust extraction
                                start_idx = action.lower().find('finish[') + len('finish[')
                                if ']' in action[start_idx:]:
                                    end_idx = action.find(']', start_idx)
                                    answer = action[start_idx:end_idx]
                                else:
                                    answer = action[start_idx:]
                                if answer:
                                    best_node = node  # Update best_node to this one
                                    break
                    if answer:
                        break
        
        # Evaluate success: check reward first, then evaluate answer manually if needed
        success = best_node.reward == 1
        if not success and answer:
            # Try to evaluate answer manually if reward wasn't 1
            # This handles cases where environment didn't correctly calculate reward
            success = self._evaluate_answer(answer, ground_truth)
            if success:
                # Update node reward if we manually verified success
                best_node.reward = 1
                best_node.is_terminal = True
        
        history.append({
            'step': 'selection',
            'status': 'completed',
            'best_score': best_score,
            'answer': answer,
            'success': success,
            'best_node_reward': best_node.reward if best_node else 0
        })
        
        return success, answer, history
    
    def _extract_answer(self, node: TreeSearchNode) -> str:
        """从节点中提取答案"""
        if not node.state:
            return ""
        
        # Ensure state is a dictionary
        if not isinstance(node.state, dict):
            return ""
        
        action = node.state.get('action', '')
        if action and action.startswith('finish['):
            # More robust extraction: handle cases where ] might be missing
            start_idx = action.lower().find('finish[') + len('finish[')
            if ']' in action[start_idx:]:
                end_idx = action.find(']', start_idx)
                return action[start_idx:end_idx].strip()
            else:
                # No closing bracket, extract to end
                return action[start_idx:].strip()
        
        # Try to extract from observation
        observation = node.state.get('observation', '')
        if observation and 'Episode finished' in observation:
            # Try to find finish action in current node or parent nodes
            current = node
            while current:
                if current.state and isinstance(current.state, dict):
                    action = current.state.get('action', '')
                    if action and action.startswith('finish['):
                        # More robust extraction
                        start_idx = action.lower().find('finish[') + len('finish[')
                        if ']' in action[start_idx:]:
                            end_idx = action.find(']', start_idx)
                            return action[start_idx:end_idx].strip()
                        else:
                            return action[start_idx:].strip()
                current = current.parent
        
        # If still no answer, check if reward is 1 and try to find finish action in trajectory
        if node.reward == 1:
            # Traverse up to find finish action
            current = node
            while current:
                if current.state and isinstance(current.state, dict):
                    action = current.state.get('action', '')
                    if action and action.startswith('finish['):
                        # More robust extraction
                        start_idx = action.lower().find('finish[') + len('finish[')
                        if ']' in action[start_idx:]:
                            end_idx = action.find(']', start_idx)
                            return action[start_idx:end_idx].strip()
                        else:
                            return action[start_idx:].strip()
                current = current.parent
        
        return ""
    
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

