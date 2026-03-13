#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试：验证因子代码字符检查
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

test_code = """def calculate_factor(df: pd.DataFrame) -> pd.Series:
    \"\"\"
    12日指数移动平均（EMA12）
    df: 包含OHLCV数据的DataFrame
    返回: 与df长度相同的因子值Series
    \"\"\"
    close = df['close']
    factor_values = EMA(close, timeperiod=12)
    return factor_values
"""

print("=== 测试字符检查 ===")
print("\n测试代码:")
print(test_code)

import re
# 新的正则表达式 - 只检查控制字符
print("\n\n使用新的检查方法...")
if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', test_code):
    print("❌ 代码包含非法控制字符")
else:
    print("✅ 代码通过字符检查")

print("\n✅ 修改完成！现在带中文的函数形式因子代码应该可以通过验证了。")
