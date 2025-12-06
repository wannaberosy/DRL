"""ReAct (Reasoning and Acting) 方法实现"""
from typing import List, Tuple, Optional
from utils.llm_client import LLMClient
from game24.validator import validate_solution, extract_solution_from_text


class ReActSolver:
    """ReAct 求解器：简单的思考-行动循环"""
    
    def __init__(self, llm_client: LLMClient, max_iterations: int = 10):
        """
        初始化ReAct求解器
        
        Args:
            llm_client: LLM客户端
            max_iterations: 最大迭代次数
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
    
    def solve(self, numbers: List[int], problem_description: str) -> Tuple[bool, str, List[str]]:
        """
        使用ReAct方法解决问题
        
        Args:
            numbers: 四个数字
            problem_description: 问题描述
            
        Returns:
            (success, solution, history): 是否成功、解决方案、历史记录
        """
        history = []
        current_state = f"当前数字: {numbers}"
        
        for iteration in range(self.max_iterations):
            # 思考阶段
            think_prompt = f"""你正在解决一个24点游戏问题。
{problem_description}

{current_state}

历史尝试：
{chr(10).join(history[-3:]) if history else "无"}

请思考下一步应该做什么。格式：
Thought: [你的思考]
Action: [你的行动，比如尝试一个表达式]
"""
            
            think_response = self.llm.generate(think_prompt, max_tokens=150)
            history.append(f"迭代 {iteration + 1} - 思考: {think_response}")
            
            # 提取行动
            if "Action:" in think_response:
                action = think_response.split("Action:")[-1].strip()
            else:
                # 尝试直接提取表达式
                action = extract_solution_from_text(think_response)
                if not action:
                    action = think_response
            
            # 验证行动
            is_valid, message = validate_solution(numbers, action)
            
            if is_valid:
                history.append(f"成功！解决方案: {action}")
                return True, action, history
            
            # 观察结果
            observation = f"尝试失败: {message}"
            history.append(f"迭代 {iteration + 1} - 行动: {action}")
            history.append(f"迭代 {iteration + 1} - 观察: {observation}")
            current_state = f"{current_state}\n上次尝试: {action} -> {observation}"
        
        return False, "", history
    
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

