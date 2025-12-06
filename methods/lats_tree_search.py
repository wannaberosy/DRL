"""
LATS Tree Search for Game24 with Tree-GRPO optimization
This module integrates Tree-GRPO's batch tree search strategy into Game24 LATs framework.
"""
from typing import List, Tuple, Optional
import random
import re
import math
import uuid
from utils.llm_client import LLMClient
from game24.validator import validate_solution, extract_solution_from_text
from hotpot.tree_search_node import TreeSearchNode


class Game24LATSTreeSearchSolver:
    """
    Game24 LATS 求解器 with Tree-GRPO style batch tree search optimization.
    
    Key improvements:
    1. Batch tree search: Maintains multiple trees (m trees) simultaneously
    2. Iterative expansion: Performs l iterations of expansion, expanding n nodes per iteration
    3. Leaf sampling: Samples k final leaves from each tree
    4. Tree-structured supervision: Calculates final scores based on tree structure
    """
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 10, 
                 n_generate: int = 3, n_evaluate: int = 2,
                 tree_m: int = 4, tree_n: int = 2, tree_l: int = 1, tree_k: int = 4,
                 tree_expand_mode: str = 'random', tree_reward_mode: str = 'base',
                 mcts_num_simulations: int = 5, mcts_use_value_function: bool = True,
                 mcts_use_rollout: bool = True):
        """
        初始化LATS Tree Search求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大搜索迭代次数（用于向后兼容，实际使用 tree_l）
            n_generate: 每次扩展生成的候选数
            n_evaluate: 每次评估的采样数
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
        self.tree_l = tree_l
        self.tree_k = tree_k
        self.tree_expand_mode = tree_expand_mode
        self.tree_reward_mode = tree_reward_mode
        
        # MCTS parameters
        self.mcts_num_simulations = mcts_num_simulations
        self.mcts_use_value_function = mcts_use_value_function
        self.mcts_use_rollout = mcts_use_rollout
    
    def _mcts_select(self, root: TreeSearchNode) -> Optional[TreeSearchNode]:
        """
        MCTS Selection: 使用 UCT 算法选择最有潜力的节点
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
    
    def _mcts_evaluate(self, node: TreeSearchNode, numbers: List[int], 
                       problem_description: str) -> float:
        """
        MCTS Evaluation: 使用 LLM 价值函数评估节点（Game24 版本）
        """
        if not node.children:
            return 0.0
        
        # 构建评估提示词
        child_prompts = []
        child_indices = []
        
        for i, child in enumerate(node.children):
            if not child.is_terminal and child.solution:
                # 构建评估提示词
                prompt = f"""{problem_description}
当前尝试: {child.solution}
状态: {child.state if isinstance(child.state, str) else child.state.get('state', '') if isinstance(child.state, dict) else ''}

请评估这个尝试的进展，给出0-100的分数（越接近正确答案分数越高）。
只返回数字。"""
                child_prompts.append(prompt)
                child_indices.append(i)
        
        if not child_prompts:
            return 0.0
        
        # 使用 LLM 评估所有子节点
        values = []
        for prompt in child_prompts:
            scores = []
            for _ in range(self.n_evaluate):
                try:
                    response = self.llm.generate(prompt, max_tokens=50)
                    # 提取数字
                    numbers_found = re.findall(r'\d+', response)
                    if numbers_found:
                        score = float(numbers_found[0])
                        scores.append(min(100.0, max(0.0, score)) / 100.0)
                    else:
                        scores.append(0.5)
                except:
                    scores.append(0.5)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            values.append(avg_score)
        
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
    
    def _mcts_simulate(self, node: TreeSearchNode, numbers: List[int], 
                       problem_description: str, max_depth: int = 3) -> Tuple[float, TreeSearchNode]:
        """
        MCTS Simulation: 从节点进行 rollout 模拟（Game24 版本）
        """
        current = node
        depth = 0
        rewards = [0.0]
        
        # Rollout 循环
        while depth < max_depth and not current.is_terminal:
            # 如果节点有解，先验证
            if current.solution:
                is_valid, _ = validate_solution(numbers, current.solution)
                if is_valid:
                    current.is_terminal = True
                    current.value = 1.0
                    current.reward = 1
                    return 1.0, current
            
            # 生成新的候选状态
            new_states = self._generate_new_states(current, numbers, problem_description)
            if not new_states:
                current.is_terminal = True
                break
            
            # 检查是否有终端节点（已找到解）
            terminal_success = [s for s in new_states 
                              if s.solution and validate_solution(numbers, s.solution)[0]]
            if terminal_success:
                terminal_node = terminal_success[0]
                terminal_node.is_terminal = True
                terminal_node.value = 1.0
                terminal_node.reward = 1
                return 1.0, terminal_node
            
            # 评估非终端节点
            non_terminal = [s for s in new_states if not s.is_terminal]
            if not non_terminal:
                current.is_terminal = True
                break
            
            # 严格验证：只有通过验证的解决方案才有价值
            for child in non_terminal:
                if child.solution:
                    is_valid, _ = validate_solution(numbers, child.solution)
                    if is_valid:
                        child.is_terminal = True
                        child.value = 1.0
                        child.reward = 1
                    else:
                        # 验证失败，不设置价值
                        child.value = 0.0
                else:
                    child.value = 0.0
            
            # 选择价值最高的子节点继续
            best_child = max(non_terminal, key=lambda c: c.value)
            current = best_child
            rewards.append(best_child.value)
            depth += 1
        
        # 最后检查是否找到解
        if current.solution:
            is_valid, _ = validate_solution(numbers, current.solution)
            if is_valid:
                current.is_terminal = True
                current.value = 1.0
                current.reward = 1
                return 1.0, current
        
        # 返回平均奖励
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return avg_reward, current
    
    def _mcts_backpropagate(self, node: TreeSearchNode, reward: float):
        """
        MCTS Backpropagation: 反向传播奖励
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
    
    def get_expand_node_with_mcts(self, root: TreeSearchNode, numbers: List[int],
                                   problem_description: str, n: int = 1) -> List[TreeSearchNode]:
        """
        使用完整的 MCTS 流程选择 n 个最有潜力的节点进行扩展（Game24 版本）
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
                # 生成新状态
                new_states = self._generate_new_states(node, numbers, problem_description)
                for new_state in new_states:
                    node.add_child(new_state)
                    # 初始化统计信息
                    new_state.visits = 1
                    # 初始化价值 - 严格验证：只有通过验证的解决方案才被接受
                    if new_state.solution:
                        is_valid, _ = validate_solution(numbers, new_state.solution)
                        if is_valid:
                            new_state.is_terminal = True
                            new_state.reward = 1
                            new_state.value = 1.0
                        else:
                            # 验证失败：不设置启发式价值，标记为无效
                            # 只有通过严格验证（数字使用正确且结果=24）的解决方案才有效
                            new_state.is_terminal = False
                            new_state.reward = 0
                            new_state.value = 0.0
                    else:
                        new_state.value = 0.0
            
            if not node.children:
                continue
            
            # Evaluation: 使用价值函数评估（如果启用）
            if self.mcts_use_value_function:
                avg_value = self._mcts_evaluate(node, numbers, problem_description)
            else:
                # 不使用价值函数，但必须验证解决方案
                for child in node.children:
                    if child.value == 0 and child.solution:
                        # 严格验证：只有通过验证的解决方案才有价值
                        is_valid, _ = validate_solution(numbers, child.solution)
                        if is_valid:
                            child.is_terminal = True
                            child.reward = 1
                            child.value = 1.0
                        else:
                            # 验证失败，不设置价值
                            child.value = 0.0
            
            # Simulation: 进行 rollout（如果启用）
            if self.mcts_use_rollout:
                best_child = max(node.children, key=lambda c: c.value)
                reward, terminal_node = self._mcts_simulate(
                    best_child, numbers, problem_description, max_depth=3
                )
            else:
                # 不使用 rollout，直接使用子节点的 reward
                best_child = max(node.children, key=lambda c: c.value)
                reward = best_child.reward
                terminal_node = best_child
            
            # Backpropagation: 更新节点价值
            self._mcts_backpropagate(terminal_node, reward)
        
        # 选择最有价值的 n 个节点
        all_candidates = root.get_subtree_nodes() + [root]
        non_terminal_candidates = [n for n in all_candidates 
                                  if not n.is_terminal]
        
        if not non_terminal_candidates:
            return []
        
        # 按 UCT 分数排序
        non_terminal_candidates.sort(key=lambda x: x.ucb_score(), reverse=True)
        return non_terminal_candidates[:min(n, len(non_terminal_candidates))]
    
    def expand_node_batch(self, nodes: List[TreeSearchNode], numbers: List[int], 
                         problem_description: str) -> None:
        """
        Batch expand multiple nodes.
        
        Args:
            nodes: List of nodes to expand
            numbers: Four numbers for the game
            problem_description: Problem description
        """
        for node in nodes:
            if node.is_terminal:
                node.is_leaf = True
                continue
            
            # Generate new states
            new_states = self._generate_new_states(node, numbers, problem_description)
            
            # Add children
            for new_state_node in new_states:
                node.add_child(new_state_node)
                # Check if solution is valid
                if new_state_node.solution:
                    is_valid, _ = validate_solution(numbers, new_state_node.solution)
                    if is_valid:
                        new_state_node.is_terminal = True
                        new_state_node.reward = 1
                        new_state_node.value = 1.0
                    else:
                        # Calculate partial score based on how close to 24
                        try:
                            expr = new_state_node.solution.split('=')[0].strip()
                            result = eval(expr)
                            distance = abs(result - 24)
                            if distance < 1e-6:
                                new_state_node.value = 1.0
                                new_state_node.is_terminal = True
                                new_state_node.reward = 1
                            elif distance < 1:
                                new_state_node.value = 0.8
                            elif distance < 5:
                                new_state_node.value = 0.5
                            elif distance < 10:
                                new_state_node.value = 0.3
                            else:
                                new_state_node.value = 0.1
                        except:
                            new_state_node.value = 0.0
                
                new_state_node.is_leaf = new_state_node.is_terminal
            
            # If no children were added, mark as leaf
            if len(node.children) == 0:
                node.is_leaf = True
    
    def _generate_new_states(self, node: TreeSearchNode, numbers: List[int], 
                            problem_description: str) -> List[TreeSearchNode]:
        """
        Generate new child states from a node.
        
        Args:
            node: Parent node
            numbers: Four numbers
            problem_description: Problem description
            
        Returns:
            List of new child nodes
        """
        # Build expansion prompt
        expand_prompt = f"""你正在解决一个24点游戏问题。
{problem_description}

当前状态: {node.state if hasattr(node, 'state') and node.state else '初始状态'}

请生成 {self.n_generate} 个可能的数学表达式，使用数字 {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} 和运算符 +, -, *, / 得到24。

要求：
1. 每个表达式必须使用所有四个数字：{numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}
2. 表达式必须语法正确，括号要匹配
3. 每行只写一个表达式，不要写等号和结果
4. 格式示例：(8 + 4) * 2 或 8 * (9 - 6)

请直接输出 {self.n_generate} 个表达式，每行一个：
"""
        
        response = self.llm.generate(expand_prompt, max_tokens=300)
        
        # Parse generated actions
        actions = []
        seen_actions = set()
        
        # Extract expressions from response
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            # 移除编号前缀
            if line[0].isdigit() and ('.' in line or ')' in line[:3]):
                line = re.sub(r'^\d+[.)]\s*', '', line)
            
            # 优先使用extract_solution_from_text提取
            action = extract_solution_from_text(line)
            if action and action not in seen_actions:
                # 清理表达式：移除等号和结果部分
                if '=' in action:
                    action = action.split('=')[0].strip()
                # 移除末尾可能的"24"
                if action.endswith('24'):
                    action = action[:-2].strip()
            if action and action not in seen_actions:
                actions.append(action)
                seen_actions.add(action)
            elif any(op in line for op in ['+', '-', '*', '/', '(', ')']):
                # 备用提取方法
                expr_match = re.search(r'[\(\)\d+\-*/\.\s]+', line)
                if expr_match:
                    potential_expr = expr_match.group().strip()
                    # 移除等号和结果部分
                    if '=' in potential_expr:
                        potential_expr = potential_expr.split('=')[0].strip()
                    if any(op in potential_expr for op in ['+', '-', '*', '/']):
                        if potential_expr not in seen_actions:
                            actions.append(potential_expr)
                            seen_actions.add(potential_expr)
        
        # If not enough actions, generate more
        attempts = 0
        while len(actions) < self.n_generate and attempts < self.n_generate * 2:
            single_action = self.llm.generate(
                f"{problem_description}\n当前状态: {node.state if hasattr(node, 'state') and node.state else '初始状态'}\n请给出一个使用 {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} 得到24的表达式（只返回表达式，不要其他文字）：",
                max_tokens=50
            )
            action = extract_solution_from_text(single_action)
            if action and action not in seen_actions:
                actions.append(action)
                seen_actions.add(action)
            attempts += 1
        
        # Create child nodes
        children = []
        seen_solutions = set()
        
        for action in actions[:self.n_generate * 2]:
            if not action or action in seen_solutions:
                continue
            
            # Validate expression syntax
            try:
                # Check parentheses
                if action.count('(') != action.count(')'):
                    open_count = action.count('(')
                    close_count = action.count(')')
                    if open_count > close_count:
                        action = action + ')' * (open_count - close_count)
                    elif close_count > open_count:
                        temp = action
                        for _ in range(close_count - open_count):
                            temp = temp.rsplit(')', 1)[0] if ')' in temp else temp
                        action = temp
                
                if not re.search(r'\d+', action):
                    continue
                
                if not any(op in action for op in ['+', '-', '*', '/']):
                    continue
                
                # Test syntax
                test_expr = re.sub(r'\d+', '1', action)
                try:
                    eval(test_expr)
                except:
                    continue
                
                seen_solutions.add(action)
                
                # Create new node
                node_id = str(uuid.uuid4())
                parent_state = node.state if isinstance(node.state, str) else (node.state.get('state', '初始状态') if isinstance(node.state, dict) else '初始状态')
                new_state = f"{parent_state}\n尝试: {action}"
                child = TreeSearchNode(
                    node_id=node_id,
                    state={'state': new_state, 'solution': action},
                    question=None,
                    parent=node,
                    depth=node.depth + 1,
                    reward_mode=self.tree_reward_mode,
                )
                child.solution = action
                # Store state as string for compatibility
                if not isinstance(child.state, str):
                    child.state = new_state
                children.append(child)
                
                if len(children) >= self.n_generate:
                    break
                    
            except Exception:
                continue
        
        return children
    
    def solve(self, numbers: List[int], problem_description: str) -> Tuple[bool, str, List[str]]:
        """
        使用Tree-GRPO风格的批量树搜索解决问题
        
        Args:
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            (success, solution, history): 是否成功、解决方案、历史记录
        """
        history = []
        
        # Step 1: Initialize m root trees
        root_trees = []
        initial_state = f"初始数字: {numbers}"
        for i in range(self.tree_m):
            node_id = str(uuid.uuid4())
            root = TreeSearchNode(
                node_id=node_id,
                state={'state': initial_state, 'solution': None},
                question=None,
                depth=0,
                is_root=True,
                reward_mode=self.tree_reward_mode,
            )
            # Store state as string for compatibility with original Node class
            root.state = initial_state
            root_trees.append(root)
        
        history.append(f"初始化: 创建 {self.tree_m} 棵树")
        
        # Step 2: Generate initial action chains for m trees
        for root in root_trees:
            self.expand_node_batch([root], numbers, problem_description)
        
        history.append("初始扩展: 为每棵树生成初始候选解")
        
        # Step 3: Iterative expansion (l iterations)
        for expansion_iter in range(self.tree_l):
            expansion_nodes = []
            
            # Get expansion nodes from each tree
            for root in root_trees:
                if self.tree_expand_mode == 'mcts':
                    # 使用完整的 MCTS 流程
                    expand_candidates = self.get_expand_node_with_mcts(
                        root, numbers, problem_description, n=self.tree_n
                    )
                else:
                    # 使用原有的简单选择
                    expand_candidates = root.get_expand_node(self.tree_n, mode=self.tree_expand_mode)
                    expansion_nodes.extend(expand_candidates)
            
            if len(expansion_nodes) == 0:
                history.append(f"扩展迭代 {expansion_iter + 1}: 没有节点可扩展")
                break
            
            # Batch expand nodes
            self.expand_node_batch(expansion_nodes, numbers, problem_description)
            
            history.append(f"扩展迭代 {expansion_iter + 1}: 扩展了 {len(expansion_nodes)} 个节点")
            
            # Check for solutions during expansion
            for root in root_trees:
                all_nodes = root.get_subtree_nodes() + [root]
                for node in all_nodes:
                    if node.is_terminal and node.reward == 1 and node.solution:
                        is_valid, _ = validate_solution(numbers, node.solution)
                        if is_valid:
                            history.append(f"扩展迭代 {expansion_iter + 1}: 找到解决方案: {node.solution}")
                            return True, node.solution, history
        
        # Step 4: Sample k leaves from each tree
        final_nodes = []
        for root in root_trees:
            sampled_leaves = root.sample_leaf(self.tree_k)
            final_nodes.extend(sampled_leaves)
        
        history.append(f"叶子采样: 从 {self.tree_m} 棵树中采样了 {len(final_nodes)} 个叶子节点")
        
        # Step 5: Evaluate all final nodes and calculate tree-structured scores
        # 严格验证：只有通过验证的解决方案才得分
        for node in final_nodes:
            if node.is_terminal and node.reward == 1:
                # 再次验证以确保正确性
                if node.solution:
                    is_valid, _ = validate_solution(numbers, node.solution)
                    if is_valid:
                        node.set_leaf_original_score(1.0)
                    else:
                        # 验证失败，重置状态
                        node.is_terminal = False
                        node.reward = 0
                        node.set_leaf_original_score(0.0)
                else:
                    node.set_leaf_original_score(0.0)
            elif node.solution:
                # 严格验证：只有通过验证的解决方案才得分
                is_valid, _ = validate_solution(numbers, node.solution)
                if is_valid:
                    node.is_terminal = True
                    node.reward = 1
                    node.set_leaf_original_score(1.0)
                else:
                    # 验证失败：不计算部分分数，严格遵循规则
                        node.set_leaf_original_score(0.0)
            else:
                node.set_leaf_original_score(0.0)
        
        # Calculate tree-structured final scores
        for root in root_trees:
            root.calculate_final_score_from_root()
        
        # Step 6: Select best trajectory
        # 严格验证：只选择通过验证的解决方案
        best_node = None
        best_score = -float('inf')
        
        # First, check for any terminal nodes with reward 1 (必须通过验证)
        for root in root_trees:
            all_nodes = root.get_subtree_nodes() + [root]
            for node in all_nodes:
                if node.solution:
                    is_valid, _ = validate_solution(numbers, node.solution)
                    if is_valid:
                        # 只有通过严格验证的解决方案才被接受
                        history.append(f"选择: 找到有效解决方案: {node.solution}")
                        return True, node.solution, history
        
        # Then, select best from final nodes (但必须通过验证)
        for node in final_nodes:
            if node.solution:
                is_valid, _ = validate_solution(numbers, node.solution)
                if is_valid:
                    score = 1.0
                    if score > best_score:
                        best_score = score
                        best_node = node
        
        # Also check all nodes in trees (必须通过验证)
        for root in root_trees:
            all_nodes = root.get_subtree_nodes() + [root]
            for node in all_nodes:
                if node.solution:
                    is_valid, _ = validate_solution(numbers, node.solution)
                    if is_valid:
                        score = 1.0
                        if score > best_score:
                            best_score = score
                            best_node = node
        
        # 如果找到通过验证的节点，返回它
        if best_node and best_node.solution:
            is_valid, _ = validate_solution(numbers, best_node.solution)
            if is_valid:
                history.append(f"选择: 最佳节点，解决方案: {best_node.solution}, 有效: True")
                return True, best_node.solution, history
        
        # 如果没有找到通过验证的解决方案，返回失败
        history.append("选择: 未找到有效解决方案（所有候选方案都未通过验证）")
        return False, "", history
    
    def solve_batch(self, problems: List[Tuple[List[int], str]]) -> List[Tuple[bool, str]]:
        """
        批量解决问题（推理模式）
        
        Args:
            problems: 问题列表
            
        Returns:
            结果列表 (success, solution)
        """
        results = []
        for numbers, description in problems:
            success, solution, _ = self.solve(numbers, description)
            results.append((success, solution))
        return results
    
    def search_for_training(self, numbers: List[int], problem_description: str) -> List[TreeSearchNode]:
        """
        为训练生成树结构（不返回最佳解，而是返回所有根节点）
        
        Args:
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            所有根节点列表（用于后续扁平化）
        """
        # Step 1: Initialize m root trees
        root_trees = []
        initial_state = f"初始数字: {numbers}"
        for i in range(self.tree_m):
            node_id = str(uuid.uuid4())
            root = TreeSearchNode(
                node_id=node_id,
                state={'state': initial_state, 'solution': None},
                question=None,
                depth=0,
                is_root=True,
                reward_mode=self.tree_reward_mode,
            )
            root.state = initial_state
            root_trees.append(root)
        
        # Step 2: Generate initial action chains for m trees
        for root in root_trees:
            self.expand_node_batch([root], numbers, problem_description)
        
        # Step 3: Iterative expansion (l iterations)
        for expansion_iter in range(self.tree_l):
            expansion_nodes = []
            
            # Get expansion nodes from each tree
            for root in root_trees:
                if self.tree_expand_mode == 'mcts':
                    expand_candidates = self.get_expand_node_with_mcts(
                        root, numbers, problem_description, n=self.tree_n
                    )
                else:
                    expand_candidates = root.get_expand_node(self.tree_n, mode=self.tree_expand_mode)
                expansion_nodes.extend(expand_candidates)
            
            if len(expansion_nodes) == 0:
                break
            
            # Batch expand nodes
            self.expand_node_batch(expansion_nodes, numbers, problem_description)
            
            # Check for early termination (如果找到解，可以继续探索其他路径用于训练)
            # 注意：训练模式下，我们保留所有路径，包括失败的
        
        # Step 4: Sample k leaves from each tree (保留所有叶子用于训练)
        for root in root_trees:
            sampled_leaves = root.sample_leaf(self.tree_k)
            # 标记为叶子节点
            for leaf in sampled_leaves:
                leaf.is_leaf = True
        
        # Step 5: Evaluate all leaves and calculate tree-structured scores
        for root in root_trees:
            all_leaves = root.get_subtree_leaves()
            for leaf in all_leaves:
                if leaf.solution:
                    is_valid, _ = validate_solution(numbers, leaf.solution)
                    if is_valid:
                        leaf.is_terminal = True
                        leaf.reward = 1
                        leaf.set_leaf_original_score(1.0)
                    else:
                        leaf.set_leaf_original_score(0.0)
                else:
                    leaf.set_leaf_original_score(0.0)
        
        # Calculate tree-structured final scores
        for root in root_trees:
            root.calculate_final_score_from_root()
        
        return root_trees
    
    def flatten_trees_to_batch(self, roots: List[TreeSearchNode], 
                                prompts: List[str]) -> List[dict]:
        """
        将LATS树扁平化为Tree-GRPO训练格式
        
        Args:
            roots: 根节点列表（每个prompt对应一个或多个根节点）
            prompts: 对应的prompt列表
            
        Returns:
            扁平化的训练数据列表，格式：
            [
                {
                    "prompt": str,
                    "response": str,  # 从根到叶子的完整路径
                    "reward": float,  # 最终奖励（0或1）
                    "group_id": int,  # 同一prompt下的样本共享group_id
                    "solution": str,  # 解决方案（如果有）
                },
                ...
            ]
        """
        flattened_data = []
        
        # 如果roots和prompts数量不匹配，需要重新组织
        # 假设每个prompt对应tree_m个根节点
        if len(roots) % len(prompts) == 0:
            trees_per_prompt = len(roots) // len(prompts)
        else:
            # 如果数量不匹配，假设每个prompt对应一个根节点
            trees_per_prompt = 1
        
        for group_idx, prompt in enumerate(prompts):
            # 获取该prompt对应的所有根节点
            start_idx = group_idx * trees_per_prompt
            end_idx = start_idx + trees_per_prompt
            prompt_roots = roots[start_idx:end_idx] if trees_per_prompt > 1 else [roots[group_idx] if group_idx < len(roots) else roots[0]]
            
            for root in prompt_roots:
                # 收集该树下所有的叶子节点
                leaves = root.get_subtree_leaves()
                
                # 如果没有叶子节点，尝试使用根节点本身
                if not leaves:
                    leaves = [root]
                
                for leaf in leaves:
                    # 重构完整路径
                    full_text = self._reconstruct_path(leaf)
                    
                    # 获取最终奖励（outcome reward）
                    final_reward = leaf.reward if leaf.is_terminal else 0.0
                    # 如果节点有original_score，也可以使用
                    if hasattr(leaf, 'original_score') and leaf.original_score > 0:
                        final_reward = leaf.original_score
                    
                    # 获取解决方案
                    solution = leaf.solution if hasattr(leaf, 'solution') else None
                    
                    flattened_data.append({
                        "prompt": prompt,
                        "response": full_text,
                        "reward": float(final_reward),
                        "group_id": group_idx,  # 关键：同一prompt下的样本共享group_id
                        "solution": solution,
                        "node_id": leaf.node_id,
                        "depth": leaf.depth,
                    })
        
        return flattened_data
    
    def _reconstruct_path(self, node: TreeSearchNode) -> str:
        """
        重构从根节点到当前节点的完整路径文本
        
        Args:
            node: 目标节点
            
        Returns:
            完整的路径文本
        """
        path_parts = []
        current = node
        
        # 从叶子节点回溯到根节点
        while current:
            if current.solution:
                path_parts.insert(0, current.solution)
            elif isinstance(current.state, str):
                path_parts.insert(0, current.state)
            elif isinstance(current.state, dict):
                state_str = current.state.get('state', '')
                if state_str:
                    path_parts.insert(0, state_str)
            
            current = current.parent
        
        # 合并路径
        return "\n".join(path_parts) if path_parts else ""
    
    def generate_training_data(self, problems: List[Tuple[List[int], str]], 
                              prompts: List[str] = None) -> List[dict]:
        """
        生成训练数据（参考文档要求的接口）
        
        Args:
            problems: 问题列表，格式 [(numbers, description), ...]
            prompts: 对应的prompt列表（如果为None，使用description作为prompt）
            
        Returns:
            扁平化的训练数据列表
        """
        if prompts is None:
            prompts = [desc for _, desc in problems]
        
        all_roots = []
        for numbers, description in problems:
            roots = self.search_for_training(numbers, description)
            all_roots.extend(roots)
        
        # 扁平化树结构
        training_data = self.flatten_trees_to_batch(all_roots, prompts)
        
        return training_data

