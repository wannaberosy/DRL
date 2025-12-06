"""设置环境变量 - 创建/修复 .env 文件"""
import os
from pathlib import Path

# API Key
api_key = "sk-f5985088a1074af794d88636163df7d2"

# 项目根目录
root_dir = Path(__file__).parent
env_file = root_dir / '.env'

print("=" * 60)
print("设置 .env 文件")
print("=" * 60)

# 创建 .env 文件
try:
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(f"DEEPSEEK_API_KEY={api_key}\n")
    print(f"✓ .env 文件已创建/更新: {env_file}")
    print(f"✓ 内容: DEEPSEEK_API_KEY={api_key[:20]}...")
except Exception as e:
    print(f"✗ 创建 .env 文件失败: {e}")
    exit(1)

# 验证文件
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    print(f"✓ 文件内容验证: {content}")
else:
    print("✗ 文件不存在")
    exit(1)

# 测试加载
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_file)

api_key_loaded = os.environ.get('DEEPSEEK_API_KEY') or os.getenv("DEEPSEEK_API_KEY")
if api_key_loaded:
    print(f"✓ 环境变量加载成功: {api_key_loaded[:20]}...")
else:
    print("✗ 环境变量加载失败")

print("=" * 60)


















