#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试遗传算法挖掘功能
"""
import sys
import os
import asyncio
import requests

# API base URL
BASE_URL = "http://localhost:8000"

def test_mining():
    """测试完整的挖掘流程"""
    print("=== 测试遗传算法挖掘 ===\n")

    # 1. 启动挖掘任务
    print("1. 启动挖掘任务...")
    start_payload = {
        "stock_code": "000001.SZ",
        "base_factors": ["RSI", "SMA"],  # 使用简单的因子名称
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "population_size": 20,
        "n_generations": 5,  # 减少代数以便快速测试
        "cx_prob": 0.7,
        "mut_prob": 0.2,
        "elite_size": 3,
        "fitness_objective": "ic_mean",
        "ic_threshold": 0.03
    }

    try:
        response = requests.post(f"{BASE_URL}/api/mining/genetic", json=start_payload)
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            print(f"[FAIL] 启动挖掘失败: {result.get('message')}")
            return

        task_id = result["data"]["task_id"]
        print(f"[OK] 挖掘任务已启动")
        print(f"   任务ID: {task_id}\n")

        # 2. 轮询获取进度
        print("2. 监控挖掘进度...")
        import time

        max_wait = 30  # 最多等待30秒
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/api/mining/status/{task_id}")
            status_response.raise_for_status()
            status_data = status_response.json()

            if status_data["success"]:
                data = status_data["data"]
                status = data["status"]
                current_gen = data.get("current_generation", 0)
                total_gen = data.get("total_generations", 0)
                progress = data.get("progress", 0)

                print(f"   状态: {status}, 进度: {progress}%, 代数: {current_gen}/{total_gen}")

                if status == "completed":
                    print("\n[OK] 挖掘完成!\n")

                    # 3. 获取结果
                    print("3. 获取挖掘结果...")
                    results_response = requests.get(f"{BASE_URL}/api/mining/results/{task_id}")
                    results_response.raise_for_status()
                    results_data = results_response.json()

                    if results_data["success"]:
                        result_data = results_data["data"]
                        factors = result_data.get("factors", [])

                        print(f"[OK] 发现 {len(factors)} 个因子:")
                        for i, factor in enumerate(factors):
                            print(f"   {i+1}. {factor['name']}")
                            print(f"      表达式: {factor['expression']}")
                            print(f"      IC: {factor['ic']:.4f}, IR: {factor['ir']:.4f}")
                            print(f"      适应度: {factor['fitness']:.4f}\n")

                        print("[SUCCESS] 所有测试通过!")
                        return

                elif status == "failed":
                    print(f"\n[FAIL] 挖掘失败: {data.get('error', '未知错误')}")
                    return

            time.sleep(1)  # 等待1秒后再次查询

        print("\n[TIMEOUT] 挖掘任务在30秒内未完成")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 网络错误: {e}")
        print("\n请确保:")
        print("  1. 后端服务正在运行 (python -m uvicorn backend.api.main:app --reload --port 8000)")
        print("  2. 可以访问 http://localhost:8000/docs")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")

if __name__ == "__main__":
    test_mining()
