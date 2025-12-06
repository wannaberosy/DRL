"""测试新的 WikiEnv 实现"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hotpot.local_wikienv import LocalWikiEnv

def test_mock_backend():
    """测试 mock 后端"""
    print("=" * 50)
    print("测试 Mock 后端")
    print("=" * 50)
    
    env = LocalWikiEnv(backend="mock")
    obs = env.reset()
    print(f"初始观察: {obs[:100]}...")
    
    # 测试搜索
    obs, reward, done, info = env.step("search[United States]")
    print(f"\n搜索 'United States':")
    print(f"观察: {obs[:200]}...")
    print(f"完成: {done}, 奖励: {reward}")
    
    # 测试查找
    obs, reward, done, info = env.step("lookup[country]")
    print(f"\n查找 'country':")
    print(f"观察: {obs[:200]}...")
    
    print("\n✓ Mock 后端测试通过！\n")

def test_wikipedia_backend():
    """测试 Wikipedia 后端"""
    print("=" * 50)
    print("测试 Wikipedia 后端")
    print("=" * 50)
    
    try:
        env = LocalWikiEnv(backend="wikipedia")
        obs = env.reset()
        print(f"初始观察: {obs[:100]}...")
        
        # 测试搜索
        print("\n搜索 'Python (programming language)'...")
        obs, reward, done, info = env.step("search[Python]")
        print(f"观察: {obs[:200]}...")
        print(f"完成: {done}, 奖励: {reward}")
        
        print("\n✓ Wikipedia 后端测试通过！\n")
    except ImportError:
        print("⚠ Wikipedia 库未安装，跳过测试")
        print("安装命令: pip install wikipedia\n")
    except Exception as e:
        print(f"⚠ Wikipedia 后端测试失败: {e}\n")

if __name__ == "__main__":
    print("\n开始测试 WikiEnv 实现...\n")
    
    # 测试 mock 后端
    test_mock_backend()
    
    # 测试 Wikipedia 后端
    test_wikipedia_backend()
    
    print("=" * 50)
    print("测试完成！")
    print("=" * 50)
    print("\n使用建议：")
    print("1. 如果网络不稳定，使用 --env_backend mock 进行演示")
    print("2. 如果安装了 wikipedia 库，使用 --env_backend wikipedia 进行实际实验")
    print("3. 如果必须使用原始方式，使用 --env_backend original（需要稳定的网络连接）")
















