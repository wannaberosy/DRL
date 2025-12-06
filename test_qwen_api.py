"""
测试 Qwen API 配置
用于验证 Qwen API 是否正确配置
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_qwen_api():
    """测试Qwen API配置"""
    print("=" * 60)
    print("测试 Qwen API 配置")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未找到 DASHSCOPE_API_KEY 环境变量")
        print("\n请设置环境变量：")
        print("  方法1: 在 .env 文件中添加：")
        print("    DASHSCOPE_API_KEY=")
        print("  方法2: 在命令行中设置：")
        print("    Windows: $env:DASHSCOPE_API_KEY=''")
        print("    Linux/Mac: export DASHSCOPE_API_KEY=''")
        return False
    
    print(f"✅ 找到 DASHSCOPE_API_KEY: {api_key[:10]}...")
    
    # 测试LLM客户端
    try:
        from utils.llm_client import LLMClient
        
        print("\n正在初始化 Qwen LLM 客户端...")
        client = LLMClient(
            model="qwen-plus",
            api_provider="qwen"
        )
        print("✅ LLM 客户端初始化成功")
        
        # 测试生成
        print("\n正在测试 API 调用...")
        response = client.generate("你好，请简单介绍一下你自己。", max_tokens=100)
        
        if response:
            print("✅ API 调用成功")
            print(f"响应: {response[:100]}...")
            return True
        else:
            print("❌ API 调用失败：返回空响应")
            return False
            
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        return False
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print("\n可能的原因：")
        print("  1. API Key 不正确")
        print("  2. 网络连接问题")
        print("  3. API 服务暂时不可用")
        return False


if __name__ == "__main__":
    success = test_qwen_api()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Qwen API 配置正确，可以开始使用！")
        print("=" * 60)
        print("\n使用示例：")
        print("  python experiments/run_experiment.py \\")
        print("      --num_problems 10 \\")
        print("      --api_provider qwen \\")
        print("      --model qwen-plus")
    else:
        print("\n" + "=" * 60)
        print("❌ Qwen API 配置失败，请检查配置")
        print("=" * 60)
        print("\n参考文档: QWEN_API_SETUP.md")
        sys.exit(1)










