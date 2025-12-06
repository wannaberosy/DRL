"""运行 HotpotQA 对比实验"""
import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hotpot.data_loader import HotpotDataLoader
from hotpot.triviaqa_data_loader import TriviaQADataLoader
from hotpot.squad_data_loader import SQuADDataLoader
from hotpot.react_solver import HotpotReActSolver
from hotpot.lats_solver import HotpotLATSSolver
from hotpot.lats_tree_search_solver import HotpotLATSTreeSearchSolver
from utils.llm_client import LLMClient
import random


def normalize_answer(s):
    """标准化答案用于比较"""
    import re
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


def evaluate_answer(prediction: str, ground_truth: str) -> bool:
    """
    评估答案是否正确（精确匹配）
    
    Args:
        prediction: 预测答案
        ground_truth: 正确答案
        
    Returns:
        是否匹配
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm


def generate_demo_results(problems, num_problems):
    """
    生成演示模式的结果（模拟数据）
    预期结果：ReAct 40%, LATS 60%
    """
    react_results = []
    lats_results = []
    
    react_target_success = int(num_problems * 0.40)  # 40% 成功率
    lats_target_success = int(num_problems * 0.60)   # 60% 成功率
    
    # 为 ReAct 生成结果（40% 成功）
    react_success_indices = set(random.sample(range(num_problems), react_target_success))
    for i, (question, answer) in enumerate(problems):
        success = i in react_success_indices
        react_results.append({
            'question': question,
            'ground_truth': answer,
            'success': success,
            'answer': answer if success else 'incorrect answer',
            'history': [{'thought': '思考中...', 'action': 'search', 'observation': '找到信息' if success else '未找到'}]
        })
    
    # 为 LATS 生成结果（60% 成功）
    lats_success_indices = set(random.sample(range(num_problems), lats_target_success))
    for i, (question, answer) in enumerate(problems):
        success = i in lats_success_indices
        lats_results.append({
            'question': question,
            'ground_truth': answer,
            'success': success,
            'answer': answer if success else 'incorrect answer',
            'history': [{'thought': '使用树搜索...', 'action': 'expand', 'observation': '找到解' if success else '继续搜索'}]
        })
    
    return react_results, lats_results


def run_experiment(num_problems: int = 20, max_iterations: int = 10,
                   model: str = "deepseek-chat", n_generate: int = 3,
                   n_evaluate: int = 1, api_provider: str = "deepseek",
                   demo_mode: bool = False, split: str = "dev",
                   start_idx: int = 0, env_backend: str = "wikipedia",
                   dataset_file: str = None, dataset_url: str = None,
                   use_tree_search: bool = False, tree_m: int = 4,
                   tree_n: int = 2, tree_l: int = 1, tree_k: int = 4,
                   tree_expand_mode: str = 'random', tree_reward_mode: str = 'base',
                   mcts_num_simulations: int = 5, mcts_use_value_function: bool = True,
                   mcts_use_rollout: bool = True, random_sample: bool = True,
                   random_seed: Optional[int] = None, dataset_type: str = 'h'):
    """
    运行对比实验
    
    Args:
        num_problems: 测试问题数量
        max_iterations: 最大迭代次数
        model: 使用的模型
        n_generate: LATS每次扩展生成的候选数
        n_evaluate: LATS每次评估的采样数
        api_provider: API提供商，'deepseek'、'openai' 或 'qwen'
        demo_mode: 是否使用演示模式
        split: 数据集分割
        start_idx: 起始索引
        env_backend: 环境后端类型，'dataset'（使用数据集）、'wikipedia'（使用 Wikipedia 库）、'mock'（模拟环境）或 'original'（原始环境）
        dataset_file: 数据集文件路径（用于 dataset 模式）
        dataset_url: 数据集URL（用于 dataset 模式，如果文件不存在则下载）
        dataset_type: 数据集类型，'h' (HotpotQA), 't' (TriviaQA), 's' (SQuAD)
    """
    dataset_names = {'h': 'HotpotQA', 't': 'TriviaQA', 's': 'SQuAD'}
    dataset_name = dataset_names.get(dataset_type, 'HotpotQA')
    
    print(f"开始 {dataset_name} 实验: {num_problems} 个问题, 最大迭代次数: {max_iterations}")
    print(f"使用 API: {api_provider.upper()}, 模型: {model}")
    print(f"环境后端: {env_backend}")
    print(f"数据集类型: {dataset_name} ({dataset_type})")
    
    # 根据数据集类型选择加载器
    if dataset_type == 'h':
        data_loader = HotpotDataLoader()
        if env_backend == "dataset":
            if dataset_url is None:
                dataset_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
            if dataset_file is None:
                dataset_file = "hotpot_train_v1.1.json"
    elif dataset_type == 't':
        data_loader = TriviaQADataLoader()
        if env_backend == "dataset":
            if dataset_file is None:
                dataset_file = "verified-wikipedia-dev.json"
    elif dataset_type == 's':
        data_loader = SQuADDataLoader()
        if env_backend == "dataset":
            if dataset_file is None:
                dataset_file = "dev-v2.0.json"
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}，必须是 'h' (HotpotQA), 't' (TriviaQA), 或 's' (SQuAD)")
    
    # 加载数据
    print("加载测试数据...")
    
    # 如果使用数据集环境，需要加载完整格式的数据
    if env_backend == "dataset":
        # 加载完整数据集
        if dataset_type == 'h':
            full_data = data_loader.load_full_dataset(data_file=dataset_file, url=dataset_url)
        else:
            full_data = data_loader.load_full_dataset(data_file=dataset_file)
        
        # 随机采样或顺序加载
        if random_sample:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            if num_problems >= len(full_data):
                sampled_data = full_data.copy()
            else:
                sampled_data = random.sample(full_data, num_problems)
            problems = [(item.get('question', ''), item.get('answer', '')) 
                        for item in sampled_data]
        else:
            # 顺序加载
        problems = [(item.get('question', ''), item.get('answer', '')) 
                    for item in full_data[start_idx:start_idx+num_problems]]
    else:
        # 使用简化格式
        problems = data_loader.get_problems(num_problems, split=split, start_idx=start_idx,
                                           random_sample=random_sample, random_seed=random_seed)
    
    if not problems:
        print("错误: 无法加载数据")
        return 0.0, 0.0
    
    print(f"成功加载 {len(problems)} 个问题")
    
    # 初始化LLM客户端
    print(f"初始化LLM客户端 (模型: {model}, API: {api_provider})...")
    llm_client = LLMClient(model=model, api_provider=api_provider)
    
    # 初始化求解器
    print("初始化求解器...")
    
    # 根据数据集类型准备数据集数据（用于 dataset 环境）
    dataset_data_for_env = None
    if env_backend == "dataset":
        if dataset_type == 'h':
            if dataset_file is None:
                dataset_file = "hotpot_train_v1.1.json"
            if dataset_url is None:
                dataset_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
            dataset_data_for_env = data_loader.load_full_dataset(data_file=dataset_file, url=dataset_url)
        elif dataset_type == 't':
            if dataset_file is None:
                dataset_file = "verified-wikipedia-dev.json"
            dataset_data_for_env = data_loader.load_full_dataset(data_file=dataset_file)
        elif dataset_type == 's':
            if dataset_file is None:
                dataset_file = "dev-v2.0.json"
            dataset_data_for_env = data_loader.load_full_dataset(data_file=dataset_file)
    
    react_solver = HotpotReActSolver(llm_client, max_iterations=max_iterations, 
                                     env_backend=env_backend, 
                                     dataset_file=dataset_file, dataset_url=dataset_url)
    
    if use_tree_search:
        print(f"使用 Tree-GRPO 优化版本 (m={tree_m}, n={tree_n}, l={tree_l}, k={tree_k})")
        lats_solver = HotpotLATSTreeSearchSolver(
            llm_client, max_iterations=max_iterations,
            n_generate=n_generate, n_evaluate=n_evaluate, 
            env_backend=env_backend,
            dataset_file=dataset_file, dataset_url=dataset_url,
            tree_m=tree_m, tree_n=tree_n, tree_l=tree_l, tree_k=tree_k,
            tree_expand_mode=tree_expand_mode, tree_reward_mode=tree_reward_mode,
            mcts_num_simulations=mcts_num_simulations,
            mcts_use_value_function=mcts_use_value_function,
            mcts_use_rollout=mcts_use_rollout
        )
    else:
        lats_solver = HotpotLATSSolver(llm_client, max_iterations=max_iterations,
                                       n_generate=n_generate, n_evaluate=n_evaluate, 
                                       env_backend=env_backend,
                                       dataset_file=dataset_file, dataset_url=dataset_url)
    
    # 演示模式：使用模拟数据生成预期结果
    if demo_mode:
        print("\n[演示模式] 使用模拟数据生成预期结果...")
        print("注意: 这是用于演示的模拟数据，实际结果需要 API 调用。")
        react_results, lats_results = generate_demo_results(problems, num_problems)
    else:
        # 运行ReAct实验
        print("\n运行 ReAct 实验...")
        react_results = []
        api_error_count = 0
        for i, (question, ground_truth) in enumerate(tqdm(problems, desc="ReAct")):
            try:
                success, answer, history = react_solver.solve(question, ground_truth, question_idx=i)
                
                # 使用精确匹配评估
                if not success and answer:
                    success = evaluate_answer(answer, ground_truth)
                
                react_results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'success': success,
                    'answer': answer,
                    'history': history
                })
                
                # 检查是否有太多连续错误（可能是 API 问题）
                if i >= 2:
                    recent_errors = sum(1 for r in react_results[-3:] if not r.get('answer') or r.get('history', [{}])[0].get('error'))
                    if recent_errors >= 3:
                        print(f"\n警告: 检测到连续 {recent_errors} 个问题出现错误，可能是 API 调用问题。")
                        print("建议检查：")
                        print("1. API Key 是否正确配置")
                        print("2. API 余额是否充足")
                        print("3. 网络连接是否正常")
                        print("继续运行剩余问题...\n")
                
            except KeyboardInterrupt:
                print("\n\n用户中断实验")
                break
            except Exception as e:
                import traceback
                error_msg = str(e)
                print(f"\n处理问题 {i+1} 时出错: {error_msg}")
                if "JSON" in error_msg or "json" in error_msg.lower():
                    print("提示: 这可能是 API 响应格式问题，将重试或跳过")
                
                react_results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'success': False,
                    'answer': '',
                    'history': [{'error': error_msg, 'traceback': traceback.format_exc()}]
                })
        
        if not demo_mode and len(react_results) == len(problems):
            # 正常运行完成，继续运行 LATS
            print("\n运行 LATS 实验...")
            lats_results = []
            for i, (question, ground_truth) in enumerate(tqdm(problems, desc="LATS")):
                try:
                    success, answer, history = lats_solver.solve(question, ground_truth, question_idx=i)
                    
                    # 使用精确匹配评估
                    if not success and answer:
                        success = evaluate_answer(answer, ground_truth)
                    
                    lats_results.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'success': success,
                        'answer': answer,
                        'history': history
                    })
                except KeyboardInterrupt:
                    print("\n\n用户中断实验")
                    break
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    print(f"\n处理问题 {i+1} 时出错: {error_msg}")
                    if "JSON" in error_msg or "json" in error_msg.lower():
                        print("提示: 这可能是 API 响应格式问题，将重试或跳过")
                    
                    lats_results.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'success': False,
                        'answer': '',
                        'history': [{'error': error_msg, 'traceback': traceback.format_exc()}]
                    })
    
    # 计算成功率
    react_success_rate = sum(1 for r in react_results if r['success']) / len(react_results) if react_results else 0.0
    lats_success_rate = sum(1 for r in lats_results if r['success']) / len(lats_results) if lats_results else 0.0
    
    print(f"\n实验结果:")
    print(f"ReAct 成功率: {react_success_rate:.2%} ({sum(1 for r in react_results if r['success'])}/{len(react_results)})")
    print(f"LATS 成功率: {lats_success_rate:.2%} ({sum(1 for r in lats_results if r['success'])}/{len(lats_results)})")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    dataset_prefix = {'h': 'hotpot', 't': 'triviaqa', 's': 'squad'}.get(dataset_type, 'hotpot')
    results_file = os.path.join(results_dir, f"{dataset_prefix}_experiment_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'num_problems': num_problems,
                'max_iterations': max_iterations,
                'model': model,
                'n_generate': n_generate,
                'n_evaluate': n_evaluate,
                'api_provider': api_provider,
                'split': split,
                'start_idx': start_idx,
                'env_backend': env_backend,
                'dataset_type': dataset_type,
                'dataset_name': dataset_name,
                'use_tree_search': use_tree_search,
                'tree_m': tree_m if use_tree_search else None,
                'tree_n': tree_n if use_tree_search else None,
                'tree_l': tree_l if use_tree_search else None,
                'tree_k': tree_k if use_tree_search else None,
                'tree_expand_mode': tree_expand_mode if use_tree_search else None,
                'tree_reward_mode': tree_reward_mode if use_tree_search else None,
                'mcts_num_simulations': mcts_num_simulations if use_tree_search else None,
                'mcts_use_value_function': mcts_use_value_function if use_tree_search else None,
                'mcts_use_rollout': mcts_use_rollout if use_tree_search else None
            },
            'react_results': react_results,
            'lats_results': lats_results,
            'react_success_rate': react_success_rate,
            'lats_success_rate': lats_success_rate,
            'demo_mode': demo_mode
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 生成可视化
    visualize_results(react_success_rate, lats_success_rate, results_dir, timestamp, dataset_type)
    
    return react_success_rate, lats_success_rate


def visualize_results(react_rate: float, lats_rate: float,
                     output_dir: str, timestamp: str, dataset_type: str = 'h'):
    """
    可视化结果
    
    Args:
        react_rate: ReAct成功率
        lats_rate: LATS成功率
        output_dir: 输出目录
        timestamp: 时间戳
        dataset_type: 数据集类型
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['ReAct', 'LATS']
    success_rates = [react_rate * 100, lats_rate * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(methods, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 设置标题和标签
    ax.set_ylabel('成功率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('HotpotQA: ReAct vs LATS 成功率对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加结论文本
    if lats_rate > react_rate:
        improvement = ((lats_rate - react_rate) / react_rate) * 100 if react_rate > 0 else 0
        conclusion = f'LATS 比 ReAct 提升了 {improvement:.1f}%'
        ax.text(0.5, 0.95, conclusion, transform=ax.transAxes,
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    dataset_prefix = {'h': 'hotpot', 't': 'triviaqa', 's': 'squad'}.get(dataset_type, 'hotpot')
    output_file = os.path.join(output_dir, f"{dataset_prefix}_comparison_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 HotpotQA ReAct vs LATS 对比实验")
    parser.add_argument("--num_problems", type=int, default=10,
                       help="测试问题数量 (默认: 10)")
    parser.add_argument("--max_iterations", type=int, default=10,
                       help="最大迭代次数 (默认: 10)")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="使用的模型 (默认: deepseek-chat)")
    parser.add_argument("--api_provider", type=str, default="deepseek",
                       choices=["deepseek", "openai", "qwen"],
                       help="API提供商 (默认: deepseek)")
    parser.add_argument("--n_generate", type=int, default=3,
                       help="LATS每次扩展生成的候选数 (默认: 3)")
    parser.add_argument("--n_evaluate", type=int, default=1,
                       help="LATS每次评估的采样数 (默认: 1)")
    parser.add_argument("--demo", action="store_true",
                       help="使用演示模式（生成模拟数据，无需API调用）")
    parser.add_argument("--split", type=str, default="dev",
                       choices=["train", "dev", "test"],
                       help="数据集分割 (默认: dev)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="起始索引 (默认: 0)")
    parser.add_argument("--env_backend", type=str, default="dataset",
                       choices=["dataset", "wikipedia", "mock", "original"],
                       help="环境后端类型: 'dataset' (使用HotpotQA数据集，推荐，无需网络), 'wikipedia' (使用 Wikipedia 库), 'mock' (模拟环境，无需网络), 'original' (原始环境，需要网络) (默认: dataset)")
    parser.add_argument("--dataset_file", type=str, default=None,
                       help="数据集文件路径（用于 dataset 模式，如果未指定则使用默认路径）")
    parser.add_argument("--dataset_url", type=str, default=None,
                       help="数据集URL（用于 dataset 模式，如果文件不存在则从URL下载，默认: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json）")
    parser.add_argument("--use_tree_search", action="store_true",
                       help="使用 Tree-GRPO 优化的树搜索版本（批量树搜索）")
    parser.add_argument("--tree_m", type=int, default=4,
                       help="Tree-GRPO: 维护的树数量 (默认: 4)")
    parser.add_argument("--tree_n", type=int, default=2,
                       help="Tree-GRPO: 每次扩展的节点数 (默认: 2)")
    parser.add_argument("--tree_l", type=int, default=None,
                       help="Tree-GRPO: 扩展迭代次数 (默认: None, 使用max_iterations)")
    parser.add_argument("--tree_k", type=int, default=4,
                       help="Tree-GRPO: 每棵树最终采样的叶子数 (默认: 4)")
    parser.add_argument("--tree_expand_mode", type=str, default='uct',
                       choices=['random', 'best', 'uct', 'mcts'],
                       help="Tree-GRPO: 节点扩展模式 (默认: uct, 'mcts' 使用完整 MCTS 流程)")
    parser.add_argument("--tree_reward_mode", type=str, default='base',
                       choices=['base', 'tree_diff'],
                       help="Tree-GRPO: 奖励计算模式 (默认: base)")
    parser.add_argument("--mcts_num_simulations", type=int, default=3,
                       help="MCTS: 模拟次数 (默认: 3, 优化后更快)")
    parser.add_argument("--mcts_use_value_function", action="store_true", default=False,
                       help="MCTS: 使用价值函数评估 (默认: False, 优化后更快)")
    parser.add_argument("--mcts_use_rollout", action="store_true", default=False,
                       help="MCTS: 使用 rollout 模拟 (默认: False, 优化后更快)")
    parser.add_argument("--random_sample", action="store_true", default=True,
                       help="随机采样数据集（默认: True，每次运行加载不同的问题）")
    parser.add_argument("--no_random_sample", dest="random_sample", action="store_false",
                       help="禁用随机采样，使用顺序加载（从start_idx开始）")
    parser.add_argument("--random_seed", type=int, default=None,
                       help="随机种子（用于可复现性，如果指定则每次运行相同的问题）")
    parser.add_argument("--dataset_type", type=str, default="h",
                       choices=["h", "t", "s"],
                       help="数据集类型: 'h' (HotpotQA), 't' (TriviaQA), 's' (SQuAD) (默认: h)")
    
    args = parser.parse_args()
    
    # 根据 API provider 设置默认模型
    if args.api_provider == "deepseek" and args.model == "gpt-3.5-turbo":
        args.model = "deepseek-chat"
    elif args.api_provider == "openai" and args.model == "deepseek-chat":
        args.model = "gpt-3.5-turbo"
    elif args.api_provider == "qwen" and args.model in ["gpt-3.5-turbo", "deepseek-chat"]:
        args.model = "qwen-plus"  # Qwen默认模型
    
    run_experiment(
        num_problems=args.num_problems,
        max_iterations=args.max_iterations,
        model=args.model,
        n_generate=args.n_generate,
        n_evaluate=args.n_evaluate,
        api_provider=args.api_provider,
        demo_mode=args.demo,
        split=args.split,
        start_idx=args.start_idx,
        env_backend=args.env_backend,
        dataset_file=args.dataset_file,
        dataset_url=args.dataset_url,
        use_tree_search=args.use_tree_search,
        tree_m=args.tree_m,
        tree_n=args.tree_n,
        tree_l=args.tree_l,
        tree_k=args.tree_k,
        tree_expand_mode=args.tree_expand_mode,
        tree_reward_mode=args.tree_reward_mode,
        mcts_num_simulations=args.mcts_num_simulations,
        mcts_use_value_function=args.mcts_use_value_function,
        mcts_use_rollout=args.mcts_use_rollout,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
        dataset_type=args.dataset_type
    )

