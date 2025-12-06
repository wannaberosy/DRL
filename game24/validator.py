"""24点游戏答案验证器"""
import re
from collections import Counter
from typing import Optional, Tuple, List


def validate_solution(numbers: List[int], solution: str) -> Tuple[bool, str]:
    """
    验证24点游戏的解决方案
    
    Args:
        numbers: 原始四个数字
        solution: 解决方案字符串（如 "8 * (9 - 6) = 24"）
        
    Returns:
        (is_valid, message): 是否有效和消息
    """
    try:
        # 提取表达式部分（去掉 "= 24" 等）
        # 支持多种格式：如 "8 * (9 - 6)" 或 "8 * (9 - 6) = 24"
        expr = solution.split('=')[0].strip()
        
        # 移除可能的 "24" 在末尾
        if expr.endswith('24'):
            expr = expr[:-2].strip()
        # 提取表达式中出现的所有整数（包含负数）
        expr_numbers = re.findall(r'-?\d+', expr)
        expr_numbers = [abs(int(n)) for n in expr_numbers]
        
        required_counts = Counter(numbers)
        expr_counts = Counter(expr_numbers)
        
        if expr_counts != required_counts:
            actual_numbers = sorted(list(expr_counts.elements()))
            return False, (
                f"数字使用不符合规则。需要: {sorted(numbers)}, 实际: {actual_numbers}"
            )
        
        # 计算表达式结果
        # 使用 eval 需要小心，但在这个受控环境中可以接受
        result = eval(expr)
        
        if abs(result - 24) < 1e-6:
            return True, "解决方案正确！"
        else:
            return False, f"结果不正确。计算得到: {result}, 需要: 24"
            
    except Exception as e:
        return False, f"验证失败: {str(e)}"


def extract_solution_from_text(text: str) -> Optional[str]:
    """
    从LLM返回的文本中提取解决方案
    
    Args:
        text: LLM返回的文本
        
    Returns:
        提取的解决方案字符串，如果未找到则返回None
    """
    # 首先尝试提取完整的数学表达式（包含括号）
    # 匹配类似 "(11 - 9) * 6 * 1" 或 "12 * (4 - 3 + 1)" 的表达式
    expr_pattern = r'[\(\)\d+\-*/\.\s]+'
    
    # 按行处理
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # 跳过空行和编号行
        if not line or (line[0].isdigit() and ('.' in line or ')' in line[:3])):
            # 移除编号前缀
            line = re.sub(r'^\d+[.)]\s*', '', line)
        
        # 提取表达式（查找包含运算符和数字的连续字符串）
        if any(op in line for op in ['+', '-', '*', '/']):
            # 尝试提取完整的表达式
            expr_match = re.search(r'[\(\)\d+\-*/\.\s]+', line)
            if expr_match:
                expr = expr_match.group().strip()
                # 检查括号是否匹配
                if expr.count('(') == expr.count(')'):
                    # 移除等号和结果部分
                    if '=' in expr:
                        expr = expr.split('=')[0].strip()
                    # 清理多余的空白
                    expr = ' '.join(expr.split())
                    if len(expr) > 3:  # 确保不是太短的片段
                        return expr
    
    # 如果按行提取失败，尝试在整个文本中查找
    full_match = re.search(r'(\([^)]+\)|[\(\)\d+\-*/\.\s]{5,})', text)
    if full_match:
        expr = full_match.group().strip()
        if '=' in expr:
            expr = expr.split('=')[0].strip()
        # 检查括号匹配
        if expr.count('(') == expr.count(')'):
            expr = ' '.join(expr.split())
            if len(expr) > 3:
                return expr
    
    return None

