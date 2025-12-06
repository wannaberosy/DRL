"""24点游戏问题生成器，基于 HuggingFace 数据集 nlile/24-game。"""
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - 依赖缺失时在运行期提示
    load_dataset = None

DATASET_NAME = "nlile/24-game"
DATASET_SPLIT = "train"

# 本地数据集路径（优先使用）
LOCAL_PARQUET = Path(__file__).parent.parent / "data" / "train-00000-of-00001.parquet"

_DATASET_CACHE = None


def _load_dataset():
    """延迟加载 24game 数据集，优先使用本地 parquet 文件，以便所有生成逻辑复用同一份缓存。"""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if load_dataset is None:
            raise ImportError(
                "需要安装 datasets 库才能使用 24-game 数据集，"
                "请运行 `pip install datasets`。"
            )
        
        # 优先使用本地 parquet 文件
        if LOCAL_PARQUET.exists():
            try:
                print(f"使用本地数据集: {LOCAL_PARQUET}")
                _DATASET_CACHE = load_dataset(
                    "parquet",
                    data_files={"train": str(LOCAL_PARQUET)},
                )["train"]
                print(f"成功加载本地数据集，共 {len(_DATASET_CACHE)} 个样本")
                return _DATASET_CACHE
            except Exception as exc:
                print(f"警告: 无法加载本地数据集 {LOCAL_PARQUET}: {exc}")
                print("尝试从 HuggingFace Hub 加载...")
        
        # 如果本地文件不存在或加载失败，尝试从 HuggingFace Hub 加载
        try:
            print(f"从 HuggingFace Hub 加载数据集: {DATASET_NAME} (split={DATASET_SPLIT})")
            _DATASET_CACHE = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
            print(f"成功加载 HuggingFace 数据集，共 {len(_DATASET_CACHE)} 个样本")
        except Exception as exc:
            raise RuntimeError(
                f"无法加载数据集。"
                f"本地文件不存在: {LOCAL_PARQUET}，"
                f"且无法从 HuggingFace Hub 加载 {DATASET_NAME}（split={DATASET_SPLIT}）。"
                f"请确认本地文件存在或已正确登录 (`huggingface-cli login`) 并具有访问权限。"
                f"错误详情: {exc}"
            ) from exc
    return _DATASET_CACHE


def _parse_numbers_from_string(value: str) -> List[int]:
    tokens = re.findall(r'-?\d+', value or "")
    numbers = [int(token) for token in tokens]
    if len(numbers) >= 4:
        return numbers[:4]
    return numbers


def _normalize_number_list(values: List[Any]) -> List[int]:
    if len(values) != 4:
        return []
    normalized: List[int] = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            return []
    return normalized


def _extract_numbers(sample: Dict[str, Any]) -> List[int]:
    # 首先尝试常见字段
    for key in ("numbers", "cards", "digits", "values"):
        if key in sample and isinstance(sample[key], list):
            candidate = _normalize_number_list(sample[key])
            if candidate:
                return candidate
    
    # 其次尝试解析字符串描述
    for key in ("problem", "question", "description", "text"):
        if key in sample and isinstance(sample[key], str):
            numbers = _parse_numbers_from_string(sample[key])
            if len(numbers) == 4:
                return numbers
    
    # 最后遍历所有字段寻找满足条件的列表
    for value in sample.values():
        if isinstance(value, list):
            candidate = _normalize_number_list(value)
            if candidate:
                return candidate
    
    raise ValueError("数据集中未找到包含四个整数的字段，无法构造 24 点问题。")


def _build_description(numbers: List[int], sample: Dict[str, Any]) -> str:
    base = None
    for key in ("description", "problem", "question", "text"):
        if isinstance(sample.get(key), str):
            base = sample[key].strip()
            break
    if not base:
        nums = ", ".join(str(n) for n in numbers)
        base = f"使用数字 {nums} 和运算符 +, -, *, / 得到24"
    
    # 根据数据源添加描述
    if LOCAL_PARQUET.exists():
        return f"{base}\n(数据集: 本地文件 {LOCAL_PARQUET.name})"
    else:
        return f"{base}\n(数据集: {DATASET_NAME}, split={DATASET_SPLIT})"


def generate_problem() -> Tuple[List[int], str]:
    """
    从 HuggingFace 数据集中随机抽取一个 24 点问题。
    
    Returns:
        (numbers, description): 四个数字和问题描述
    """
    dataset = _load_dataset()
    sample = dataset[random.randint(0, len(dataset) - 1)]
    numbers = _extract_numbers(sample)
    description = _build_description(numbers, sample)
    return numbers, description


def generate_problems(n: int) -> List[Tuple[List[int], str]]:
    """
    生成多个24点游戏问题（允许重复采样）。
    
    Args:
        n: 问题数量
        
    Returns:
        问题列表
    """
    dataset = _load_dataset()
    total = len(dataset)
    indices = [random.randint(0, total - 1) for _ in range(n)]
    problems: List[Tuple[List[int], str]] = []
    for idx in indices:
        sample = dataset[idx]
        numbers = _extract_numbers(sample)
        description = _build_description(numbers, sample)
        problems.append((numbers, description))
    return problems


def is_solvable(numbers: List[int]) -> bool:
    """
    检查问题是否可解（保留占位函数，数据集中已包含可行问题）。
    """
    return True

