"""测试 DeepSeek API 连接"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# 尝试从环境变量获取 API key
api_key = os.environ.get('DEEPSEEK_API_KEY') or os.getenv("DEEPSEEK_API_KEY")

print("=" * 60)
print("DeepSeek API 连接测试")
print("=" * 60)

if api_key:
    print(f"✓ API Key 已找到 (前10个字符: {api_key[:10]}...)")
    
    # 测试导入 LLMClient
    try:
        from utils.llm_client import LLMClient
        client = LLMClient(api_provider='deepseek', model='deepseek-chat')
        print("✓ LLMClient 初始化成功")
        
        # 测试 API 调用
        print("\n正在测试 API 调用...")
        response = client.generate("Hello, 请回复'测试成功'", max_tokens=50)
        print(f"✓ API 调用成功!")
        print(f"响应: {response}")
        print("\n" + "=" * 60)
        print("所有测试通过！可以开始运行实验了。")
        print("=" * 60)
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ 错误: 未找到 DEEPSEEK_API_KEY")
    print("\n请使用以下方法之一设置 API Key:")
    print("1. 在 PowerShell 中设置环境变量:")
    print('   $env:DEEPSEEK_API_KEY="sk-f5985088a1074af794d88636163df7d2"')
    print("\n2. 在项目根目录创建 .env 文件，内容为:")
    print("   DEEPSEEK_API_KEY=sk-f5985088a1074af794d88636163df7d2")
    exit(1)


















