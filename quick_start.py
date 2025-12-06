"""快速开始脚本 - 运行一个小规模测试"""
import os
from pathlib import Path
from dotenv import load_dotenv
from experiments.run_experiment import run_experiment

# 先加载 .env 文件（从项目根目录）
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# 如果 load_dotenv 没有加载成功，手动读取 .env 文件作为备用方案
if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
        except Exception:
            pass

if __name__ == "__main__":
    # 检查 API Key (优先使用 DeepSeek，如果没有则尝试 OpenAI)
    api_provider = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        if os.getenv("OPENAI_API_KEY"):
            api_provider = "openai"
            api_key_env = "OPENAI_API_KEY"
            model = "gpt-3.5-turbo"
        else:
            print("错误: 请先设置 API Key")
            print("方法1 (推荐): 设置 DEEPSEEK_API_KEY 环境变量")
            print("   export DEEPSEEK_API_KEY=your_key")
            print("方法2: 设置 OPENAI_API_KEY 环境变量")
            print("   export OPENAI_API_KEY=your_key")
            print("方法3: 创建 .env 文件，添加:")
            print("   DEEPSEEK_API_KEY=your_key")
            print("   或")
            print("   OPENAI_API_KEY=your_key")
            exit(1)
    else:
        model = "deepseek-chat"
    
    print("=" * 60)
    print("LATS vs ReAct 对比实验 - 快速测试")
    print("=" * 60)
    print(f"使用 API: {api_provider.upper()}")
    print("\n这将运行一个小规模测试（5个问题）来验证设置是否正确。")
    print("如果测试成功，可以使用完整实验：")
    print("  python experiments/run_experiment.py --num_problems 50\n")
    
    # 运行小规模测试
    react_rate, lats_rate = run_experiment(
        num_problems=5,
        max_iterations=5,
        model=model,
        api_provider=api_provider,
        n_generate=2,
        n_evaluate=1
    )
    
    print("\n" + "=" * 60)
    print("快速测试完成！")
    print("=" * 60)
    print(f"ReAct 成功率: {react_rate:.2%}")
    print(f"LATS 成功率: {lats_rate:.2%}")
    print("\n如果结果正常，可以运行完整实验了！")

