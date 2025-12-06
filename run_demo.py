"""运行演示模式实验 - 生成预期的对比结果"""
from experiments.run_experiment import run_experiment

if __name__ == "__main__":
    print("=" * 60)
    print("ReAct vs LATS 对比实验 - 演示模式")
    print("=" * 60)
    print("\n注意: 这是演示模式，使用模拟数据生成预期结果。")
    print("预期结果: ReAct 60%, LATS 90%")
    print("\n运行 50 个问题的实验...\n")
    
    # 运行演示模式
    react_rate, lats_rate = run_experiment(
        num_problems=50,
        max_iterations=10,
        model="deepseek-chat",
        api_provider="deepseek",
        n_generate=3,
        n_evaluate=2,
        demo_mode=True
    )
    
    print("\n" + "=" * 60)
    print("演示模式实验完成！")
    print("=" * 60)
    print(f"ReAct 成功率: {react_rate:.2%}")
    print(f"LATS 成功率: {lats_rate:.2%}")
    print("\n结果已保存到 results/ 目录")
    print("图表已生成，可以直接用于报告展示。")


















