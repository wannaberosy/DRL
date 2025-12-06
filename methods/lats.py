"""LATS (Language Agent Tree Search) 方法实现"""
from typing import List, Tuple, Optional, Dict
import random
import re
import math
from utils.llm_client import LLMClient
from game24.validator import validate_solution, extract_solution_from_text


class Node:
    """搜索树节点"""
    
    def __init__(self, state: str, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0  # 累计价值
        self.solution: Optional[str] = None
        self.is_terminal = False
    
    def ucb_score(self, exploration_weight: float = 1.414) -> float:
        """计算UCB分数（Upper Confidence Bound）"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先选择
        
        parent_visits = self.parent.visits if self.parent else 1
        if parent_visits == 0:
            parent_visits = 1
        
        exploitation = self.value / self.visits  # 平均价值
        exploration = exploration_weight * (2 * math.log(parent_visits) / self.visits) ** 0.5
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """检查是否完全展开"""
        return len(self.children) > 0 and all(child.visits > 0 for child in self.children)


class LATSSolver:
    """LATS 求解器：带树搜索的思考"""
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 10, 
                 n_generate: int = 3, n_evaluate: int = 2):
        """
        初始化LATS求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大搜索迭代次数
            n_generate: 每次扩展生成的候选数
            n_evaluate: 每次评估的采样数
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.n_generate = n_generate
        self.n_evaluate = n_evaluate
    
    def expand(self, node: Node, numbers: List[int], problem_description: str) -> List[Node]:
        """
        扩展节点：生成多个候选行动
        
        Args:
            node: 当前节点
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            新生成的子节点列表
        """
        # 构建扩展提示词
        expand_prompt = f"""你正在解决一个24点游戏问题。
{problem_description}

当前状态: {node.state}

请生成 {self.n_generate} 个可能的数学表达式，使用数字 {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} 和运算符 +, -, *, / 得到24。

要求：
1. 每个表达式必须使用所有四个数字：{numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}
2. 表达式必须语法正确，括号要匹配
3. 每行只写一个表达式，不要写等号和结果
4. 格式示例：(8 + 4) * 2 或 8 * (9 - 6)

请直接输出 {self.n_generate} 个表达式，每行一个：
"""
        
        response = self.llm.generate(expand_prompt, max_tokens=300)
        
        # 解析生成的行动
        actions = []
        seen_actions = set()  # 避免重复
        
        # 首先尝试从响应中提取所有可能的表达式
        lines = response.split('\n')
        for line in lines:
            # 跳过空行和编号
            line = line.strip()
            if not line or line[0].isdigit() and '.' in line:
                # 移除编号前缀（如 "1. " 或 "1)"）
                line = re.sub(r'^\d+[.)]\s*', '', line)
            
            # 提取表达式
            action = extract_solution_from_text(line)
            if action and action not in seen_actions:
                actions.append(action)
                seen_actions.add(action)
            
            # 如果行中包含运算符，尝试直接提取
            elif any(op in line for op in ['+', '-', '*', '/', '(', ')']):
                # 提取数学表达式
                expr_match = re.search(r'[\(\)\d+\-*/\.\s]+', line)
                if expr_match:
                    potential_expr = expr_match.group().strip()
                    # 确保包含至少一个运算符
                    if any(op in potential_expr for op in ['+', '-', '*', '/']):
                        if potential_expr not in seen_actions:
                            actions.append(potential_expr)
                            seen_actions.add(potential_expr)
        
        # 如果没找到足够的行动，使用LLM多次生成
        attempts = 0
        while len(actions) < self.n_generate and attempts < self.n_generate * 2:
            single_action = self.llm.generate(
                f"{problem_description}\n当前状态: {node.state}\n请给出一个使用 {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} 得到24的表达式（只返回表达式，不要其他文字）：",
                max_tokens=50
            )
            action = extract_solution_from_text(single_action)
            if action and action not in seen_actions:
                actions.append(action)
                seen_actions.add(action)
            attempts += 1
        
        # 如果还是没有足够的行动，生成一些简单的组合
        if len(actions) < self.n_generate:
            # 生成一些基本的表达式组合
            basic_actions = [
                f"({numbers[0]} + {numbers[1]}) * ({numbers[2]} + {numbers[3]})",
                f"({numbers[0]} * {numbers[1]}) + ({numbers[2]} * {numbers[3]})",
                f"({numbers[0]} - {numbers[1]}) * ({numbers[2]} - {numbers[3]})",
            ]
            for action in basic_actions:
                if action not in seen_actions and len(actions) < self.n_generate:
                    actions.append(action)
                    seen_actions.add(action)
        
        # 创建子节点，过滤无效表达式
        children = []
        seen_solutions = set()
        
        for action in actions[:self.n_generate * 2]:  # 生成更多候选，然后过滤
            if not action or action in seen_solutions:
                continue
            
            # 验证表达式基本语法（括号匹配、只包含允许的字符）
            try:
                # 检查括号是否匹配
                if action.count('(') != action.count(')'):
                    # 尝试修复括号
                    open_count = action.count('(')
                    close_count = action.count(')')
                    if open_count > close_count:
                        action = action + ')' * (open_count - close_count)
                    elif close_count > open_count:
                        # 移除多余的右括号
                        temp = action
                        for _ in range(close_count - open_count):
                            temp = temp.rsplit(')', 1)[0] if ')' in temp else temp
                        action = temp
                
                # 只保留包含数字和运算符的有效表达式
                if not re.search(r'\d+', action):
                    continue
                
                if not any(op in action for op in ['+', '-', '*', '/']):
                    continue
                
                # 尝试验证表达式是否可以计算（基本语法检查）
                test_expr = re.sub(r'\d+', '1', action)  # 替换数字进行语法检查
                try:
                    eval(test_expr)
                except:
                    continue  # 语法错误，跳过
                
                seen_solutions.add(action)
                new_state = f"{node.state}\n尝试: {action}"
                child = Node(new_state, parent=node)
                child.solution = action
                children.append(child)
                
                if len(children) >= self.n_generate:
                    break
                    
            except Exception as e:
                continue  # 跳过无效表达式
        
        return children
    
    def evaluate_node(self, node: Node, numbers: List[int], problem_description: str) -> float:
        """
        评估节点的所有子节点（参考LATS实现）
        
        Args:
            node: 要评估的节点（父节点）
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            平均评估分数
        """
        if not node.children:
            return 0.0
        
        # 评估所有非终端子节点
        child_prompts = []
        child_indices = []
        for i, child in enumerate(node.children):
            if not child.is_terminal and child.solution:
                # 构建评估提示词
                prompt = f"""{problem_description}
当前尝试: {child.solution}
状态: {child.state}

请评估这个尝试的进展，给出0-100的分数（越接近正确答案分数越高）。
只返回数字。"""
                child_prompts.append(prompt)
                child_indices.append(i)
        
        # 如果没有可评估的子节点，返回0
        if not child_prompts:
            return 0.0
        
        # 使用LLM评估所有子节点
        values = []
        for prompt in child_prompts:
            scores = []
            for _ in range(self.n_evaluate):
                score = self.llm.evaluate(prompt)
                scores.append(score)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            values.append(max(0.0, min(1.0, avg_score / 100.0)))  # 转换为0-1范围
        
        # 将评估值赋给对应的子节点
        for idx, value in zip(child_indices, values):
            node.children[idx].value = value
        
        # 对于终端节点（已找到解），设置高价值
        for child in node.children:
            if child.is_terminal and child.solution:
                is_valid, _ = validate_solution(numbers, child.solution)
                if is_valid:
                    child.value = 1.0
        
        # 返回所有子节点的平均价值
        if node.children:
            return sum(c.value for c in node.children) / len(node.children)
        return 0.0
    
    def select(self, node: Node) -> Node:
        """
        选择节点（UCB算法）
        
        Args:
            node: 当前节点
            
        Returns:
            选择的子节点
        """
        if not node.children:
            return node
        
        # 选择UCB分数最高的子节点
        return max(node.children, key=lambda n: n.ucb_score())
    
    def rollout(self, node: Node, numbers: List[int], problem_description: str, max_depth: int = 3) -> Tuple[float, Node]:
        """
        Rollout（模拟）：从节点开始进行深度优先搜索直到找到解或达到最大深度
        参考LATS实现：生成多个候选，评估它们，选择最好的继续
        
        Args:
            node: 起始节点
            numbers: 四个数字
            problem_description: 问题描述
            max_depth: 最大模拟深度
            
        Returns:
            (reward, terminal_node): 奖励和终端节点
        """
        current = node
        depth = 0
        rewards = [0.0]  # 初始奖励
        
        while depth < max_depth and not current.is_terminal:
            # 如果节点有解，先验证
            if current.solution:
                is_valid, _ = validate_solution(numbers, current.solution)
                if is_valid:
                    current.is_terminal = True
                    current.value = 1.0
                    return 1.0, current
            
            # 生成新的候选状态
            new_states = []
            while len(new_states) == 0 and depth < max_depth:
                children = self.expand(current, numbers, problem_description)
                new_states.extend(children)
                if not new_states:
                    # 无法生成新状态，标记为终端
                    current.is_terminal = True
                    break
            
            if not new_states:
                break
            
            # 检查是否有终端节点（已找到解）
            terminal_success = [s for s in new_states if s.solution and validate_solution(numbers, s.solution)[0]]
            if terminal_success:
                terminal_node = terminal_success[0]
                terminal_node.is_terminal = True
                terminal_node.value = 1.0
                return 1.0, terminal_node
            
            # 评估所有新状态（非终端节点）
            non_terminal = [s for s in new_states if not s.is_terminal]
            if not non_terminal:
                # 所有都是终端但都失败
                current.is_terminal = True
                break
            
            # 评估非终端节点
            for child in non_terminal:
                if child.solution:
                    # 计算部分分数
                    try:
                        expr = child.solution.split('=')[0].strip()
                        result = eval(expr)
                        distance = abs(result - 24)
                        if distance < 1:
                            child.value = 0.8
                        elif distance < 5:
                            child.value = 0.5
                        elif distance < 10:
                            child.value = 0.3
                        else:
                            child.value = 0.1
                    except:
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
                return 1.0, current
        
        # 返回平均奖励
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return avg_reward, current
    
    def backpropagate(self, node: Node, reward: float):
        """
        反向传播奖励（参考LATS实现）
        
        Args:
            node: 节点
            reward: 奖励值（0-1之间）
        """
        current = node
        while current:
            current.visits += 1
            if current.is_terminal:
                # 终端节点：如果reward=1则成功，否则失败
                if reward >= 0.99:  # 成功
                    current.value = (current.value * (current.visits - 1) + reward) / current.visits
                else:  # 失败
                    current.value = (current.value * (current.visits - 1) + (-0.5)) / current.visits
            else:
                # 非终端节点：使用奖励更新价值
                current.value = (current.value * (current.visits - 1) + reward) / current.visits
            current = current.parent
    
    def select_node(self, root: Node) -> Optional[Node]:
        """
        选择节点（使用UCT算法，参考LATS实现）
        
        Args:
            root: 根节点
            
        Returns:
            选择的节点，如果所有路径都耗尽则返回None
        """
        node = root
        
        while node and node.children:
            # 检查是否有终端节点且reward=1（成功）
            terminal_success = [c for c in node.children if c.is_terminal and c.value >= 0.99]
            if terminal_success:
                return terminal_success[0]
            
            # 过滤掉所有都是终端的子节点
            non_terminal = [c for c in node.children if not c.is_terminal]
            if not non_terminal:
                # 所有子节点都是终端，回溯
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue
            
            # 使用UCT选择最有希望的子节点
            node = max(non_terminal, key=lambda c: c.ucb_score())
        
        return node
    
    def solve(self, numbers: List[int], problem_description: str) -> Tuple[bool, str, List[str]]:
        """
        使用LATS方法解决问题（参考LanguageAgentTreeSearch实现）
        
        Args:
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            (success, solution, history): 是否成功、解决方案、历史记录
        """
        history = []
        root = Node(f"初始数字: {numbers}")
        
        # 首先扩展根节点，生成初始候选解
        if not root.children:
            children = self.expand(root, numbers, problem_description)
            root.children.extend(children)
            history.append(f"初始化: 生成 {len(children)} 个候选解")
        
        for iteration in range(self.max_iterations):
            # Selection: 选择节点
            node = self.select_node(root)
            
            # 如果选择失败或节点是终端且失败，重新选择
            while node is None or (node.is_terminal and node.value < 0.99):
                if node is None:
                    break
                node = self.select_node(root)
                if node is None:
                    break
            
            if node is None:
                history.append(f"迭代 {iteration + 1}: 所有路径已耗尽")
                break
            
            # 检查是否已经找到解
            if node.is_terminal and node.value >= 0.99:
                history.append(f"迭代 {iteration + 1}: 找到解决方案: {node.solution}")
                return True, node.solution, history
            
            # Expansion: 扩展节点
            if not node.children:
                children = self.expand(node, numbers, problem_description)
                node.children.extend(children)
            
            # 如果扩展后仍然是终端或没有子节点，重新选择
            while node.is_terminal or not node.children:
                node = self.select_node(root)
                if node is None:
                    break
                if not node.children:
                    children = self.expand(node, numbers, problem_description)
                    node.children.extend(children)
                if node is None:
                    break
            
            if node is None:
                break
            
            # Evaluation: 评估节点的所有子节点
            avg_value = self.evaluate_node(node, numbers, problem_description)
            
            # Rollout: 对价值最高的子节点进行模拟
            best_child = max(node.children, key=lambda c: c.value)
            reward, terminal_node = self.rollout(best_child, numbers, problem_description, max_depth=3)
            
            # 检查rollout是否找到解
            if terminal_node.solution:
                is_valid, _ = validate_solution(numbers, terminal_node.solution)
                if is_valid:
                    terminal_node.is_terminal = True
                    terminal_node.value = 1.0
                    history.append(f"迭代 {iteration + 1}: Rollout找到解决方案: {terminal_node.solution}")
                    self.backpropagate(terminal_node, 1.0)
                    return True, terminal_node.solution, history
            
            # Backpropagation: 反向传播rollout的奖励
            self.backpropagate(terminal_node, reward)
            
            # 检查是否有终端节点找到解
            all_nodes = self._collect_all_nodes(root)
            terminal_success = [n for n in all_nodes if n.is_terminal and n.value >= 0.99 and n.solution]
            if terminal_success:
                best_node = max(terminal_success, key=lambda n: n.value)
                history.append(f"迭代 {iteration + 1}: 找到解决方案: {best_node.solution}")
                return True, best_node.solution, history
            
            # 记录历史
            history.append(f"迭代 {iteration + 1}: 评估={avg_value:.2f}, rollout奖励={reward:.2f}, 尝试={best_child.solution or 'N/A'}")
        
        # 如果没找到解，返回最佳尝试
        all_nodes = self._collect_all_nodes(root)
        if all_nodes:
            best_node = max(all_nodes, key=lambda n: n.value / n.visits if n.visits > 0 else n.value)
            if best_node.solution:
                is_valid, _ = validate_solution(numbers, best_node.solution)
                return is_valid, best_node.solution, history
        
        return False, "", history
    
    def _collect_all_nodes(self, node: Node) -> List[Node]:
        """收集所有节点"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._collect_all_nodes(child))
        return nodes
    
    def solve_batch(self, problems: List[Tuple[List[int], str]]) -> List[Tuple[bool, str]]:
        """
        批量解决问题
        
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

