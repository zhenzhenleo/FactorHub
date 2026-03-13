"""
因子生成器服务 - 基于预置因子生成新因子
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import random
from itertools import combinations, product


class FactorGeneratorService:
    """因子生成器服务"""

    def __init__(self):
        # 可用的运算符（扩展）
        self.operators = {
            "+": "加法",
            "-": "减法",
            "*": "乘法",
            "/": "除法",
            "**": "幂运算",
            "%": "取模",
        }

        # 可用的统计函数（扩展）
        self.statistics = {
            "rank": "排名",
            "zscore": "Z-score标准化",
            "mean": "均值",
            "std": "标准差",
            "max": "最大值",
            "min": "最小值",
            "median": "中位数",
            "skew": "偏度",
            "kurtosis": "峰度",
            "quantile": "分位数",
            "diff": "差分",
            "pct_change": "百分比变化",
            "log": "对数变换",
            "abs": "绝对值",
            "sqrt": "平方根",
            "exp": "指数",
        }

        # 可用的技术指标（扩展）
        self.indicators = {
            "SMA": "简单移动平均",
            "EMA": "指数移动平均",
            "RSI": "相对强弱指标",
            "MACD": "MACD",
            "BBANDS": "布林带",
            "STOCH": "随机指标",
            "ADX": "平均趋向指数",
            "CCI": "顺势指标",
            "ATR": "真实波幅",
            "VOLATILITY": "波动率",
        }

    def generate_binary_combinations(
        self,
        base_factors: List[str],
        max_depth: int = 3,
        max_combinations: int = 100
    ) -> List[str]:
        """
        生成二元运算组合因子

        Args:
            base_factors: 基础因子列表
            max_depth: 最大深度（嵌套层数）
            max_combinations: 最大组合数

        Returns:
            因子表达式列表
        """
        expressions = []

        if len(base_factors) < 2:
            return expressions

        # 生成深度1的组合
        for factor1, factor2 in combinations(base_factors, 2):
            for op in self.operators.keys():
                expr = f"({factor1} {op} {factor2})"
                expressions.append(expr)

        # 生成深度2的组合（如果需要）
        if max_depth >= 2 and len(base_factors) >= 3:
            for _ in range(min(max_combinations // 2, 50)):  # 限制数量
                # 随机选择3个因子
                selected = random.sample(base_factors, min(3, len(base_factors)))
                ops = random.sample(list(self.operators.keys()), 2)

                # 生成嵌套表达式
                expr = f"(({selected[0]} {ops[0]} {selected[1]}) {ops[1]} {selected[2]})"
                expressions.append(expr)

        # 生成深度3的组合（如果需要）
        if max_depth >= 3 and len(base_factors) >= 4:
            for _ in range(min(max_combinations // 4, 25)):  # 限制数量
                selected = random.sample(base_factors, min(4, len(base_factors)))
                ops = random.sample(list(self.operators.keys()), 3)

                expr = f"(({selected[0]} {ops[0]} {selected[1]}) {ops[1]} ({selected[2]} {ops[2]} {selected[3]}))"
                expressions.append(expr)

        return expressions[:max_combinations]

    def generate_statistical_combinations(
        self,
        base_factors: List[str],
        window_sizes: List[int] = [5, 10, 20, 60],
        max_combinations: int = 50
    ) -> List[str]:
        """
        生成统计函数组合因子

        Args:
            base_factors: 基础因子列表
            window_sizes: 窗口大小列表
            max_combinations: 最大组合数

        Returns:
            因子表达式列表（使用pandas链式调用语法）
        """
        expressions = []

        # 需要窗口参数的函数（使用rolling）
        window_functions = {
            "mean": "mean()",
            "std": "std()",
            "max": "max()",
            "min": "min()",
            "median": "median()",
            "skew": "skew()",
            "kurtosis": "kurtosis()",
        }

        # 不需要窗口参数的函数
        no_window_functions = {
            "diff": "diff()",
            "pct_change": "pct_change()",
            "abs": "abs()",
        }

        # 特殊处理的函数
        special_functions = {
            "rank": "rank(pct=True)",
            "log": "np.log",
            "sqrt": "np.sqrt",
            "exp": "np.exp",
            "zscore": None,  # 需要特殊处理
        }

        for factor in base_factors:
            for stat_func in self.statistics.keys():
                if stat_func in window_functions:
                    # 这些函数使用rolling
                    for window in window_sizes:
                        # 生成pandas链式调用语法: factor.rolling(window).mean()
                        expr = f"({factor}.rolling({window}, min_periods=1).{window_functions[stat_func]})"
                        expressions.append(expr)
                elif stat_func in special_functions:
                    if special_functions[stat_func] is not None:
                        # 生成pandas/numpy函数调用语法
                        expr = f"{special_functions[stat_func]}({factor})"
                        expressions.append(expr)
                    elif stat_func == "zscore":
                        # zscore需要特殊处理: (x - mean) / std
                        expr = f"(({factor} - {factor}.rolling(252, min_periods=1).mean()) / ({factor}.rolling(252, min_periods=1).std() + 1e-8))"
                        expressions.append(expr)
                elif stat_func in no_window_functions:
                    # 直接调用方法
                    expr = f"({factor}.{no_window_functions[stat_func]})"
                    expressions.append(expr)
                elif stat_func == "quantile":
                    # 分位数函数
                    for q in [0.25, 0.5, 0.75]:
                        expr = f"({factor}.rolling(252, min_periods=1).quantile({q}))"
                        expressions.append(expr)

        return expressions[:max_combinations]

    def generate_indicator_combinations(
        self,
        base_factors: List[str],
        price_column: str = "close",
        max_combinations: int = 30
    ) -> List[str]:
        """
        生成技术指标组合因子

        Args:
            base_factors: 基础因子列表
            price_column: 价格列名
            max_combinations: 最大组合数

        Returns:
            因子表达式列表
        """
        expressions = []

        # 为每个基础因子生成与技术指标的组合
        for factor in base_factors:
            for indicator in self.indicators.keys():
                if indicator == "SMA":
                    for window in [5, 10, 20, 60]:
                        expr = f"({factor} / SMA({price_column}, {window}))"
                        expressions.append(expr)
                        expr = f"({factor} - SMA({price_column}, {window}))"
                        expressions.append(expr)
                elif indicator == "EMA":
                    for window in [5, 10, 20, 60]:
                        expr = f"({factor} / EMA({price_column}, {window}))"
                        expressions.append(expr)
                elif indicator == "RSI":
                    expr = f"({factor} * RSI({price_column}, 14))"
                    expressions.append(expr)
                elif indicator == "MACD":
                    expr = f"({factor} * MACD({price_column}))"
                    expressions.append(expr)

        return expressions[:max_combinations]

    def generate_hybrid_factors(
        self,
        base_factors: List[str],
        n_factors: int = 100
    ) -> List[Dict]:
        """
        生成混合因子（结合多种方法）

        Args:
            base_factors: 基础因子列表
            n_factors: 生成因子数量

        Returns:
            因子字典列表，包含表达式和元数据
        """
        factors = []

        # 1. 二元运算组合（40%）
        n_binary = int(n_factors * 0.4)
        binary_exprs = self.generate_binary_combinations(
            base_factors,
            max_combinations=n_binary
        )

        for expr in binary_exprs:
            factors.append({
                "expression": expr,
                "type": "binary_operation",
                "complexity": "medium",
            })

        # 2. 统计函数组合（30%）
        n_statistical = int(n_factors * 0.3)
        stat_exprs = self.generate_statistical_combinations(
            base_factors,
            max_combinations=n_statistical
        )

        for expr in stat_exprs:
            factors.append({
                "expression": expr,
                "type": "statistical",
                "complexity": "low",
            })

        # 3. 技术指标组合（20%）
        n_indicator = int(n_factors * 0.2)
        indicator_exprs = self.generate_indicator_combinations(
            base_factors,
            max_combinations=n_indicator
        )

        for expr in indicator_exprs:
            factors.append({
                "expression": expr,
                "type": "indicator_based",
                "complexity": "high",
            })

        # 4. 随机组合（10%）
        n_random = n_factors - len(factors)

        for _ in range(n_random):
            if len(base_factors) >= 2:
                factor1, factor2 = random.sample(base_factors, 2)
                op = random.choice(list(self.operators.keys()))

                # 随机添加统计函数
                if random.random() < 0.3:
                    stat_func = random.choice(list(self.statistics.keys()))
                    if stat_func in ["mean", "std", "max", "min"]:
                        window = random.choice([5, 10, 20])
                        expr = f"{stat_func}({factor1} {op} {factor2}, {window})"
                    else:
                        expr = f"{stat_func}({factor1} {op} {factor2})"
                else:
                    expr = f"({factor1} {op} {factor2})"

                factors.append({
                    "expression": expr,
                    "type": "random_hybrid",
                    "complexity": random.choice(["low", "medium", "high"]),
                })

        # 打乱顺序
        random.shuffle(factors)

        return factors[:n_factors]

    def compile_expression_to_code(
        self,
        expression: str,
        data_column: str = "close"
    ) -> str:
        """
        将因子表达式编译为可执行代码

        Args:
            expression: 因子表达式
            data_column: 数据列名

        Returns:
            可执行的Python代码
        """
        # 替换函数为实际实现
        code = expression

        # 替换统计函数
        # rank
        code = code.replace(
            "rank(",
            f".rank(pct=True).rolling(252, min_periods=1)."
        )

        # zscore (需要特殊处理)
        code = code.replace("zscore(", "((")

        # 均值、标准差等滚动函数
        for func in ["mean", "std", "max", "min", "median", "skew", "kurtosis"]:
            code = code.replace(
                f"{func}(",
                f".rolling(window=252, min_periods=1).{func}("
            )

        # 其他统计函数
        code = code.replace("diff(", ".diff(")
        code = code.replace("pct_change(", ".pct_change(")
        code = code.replace("log(", "np.log(")
        code = code.replace("abs(", "np.abs(")
        code = code.replace("sqrt(", "np.sqrt(")
        code = code.replace("exp(", "np.exp(")

        # quantile需要特殊处理
        code = code.replace("quantile(", ".quantile(")

        # 替换技术指标
        code = code.replace("SMA(", "talib.SMA(")
        code = code.replace("EMA(", "talib.EMA(")
        code = code.replace("RSI(", "talib.RSI(")
        code = code.replace("MACD(", "talib.MACD(")
        code = code.replace("BBANDS(", "talib.BBANDS(")
        code = code.replace("STOCH(", "talib.STOCH(")
        code = code.replace("ADX(", "talib.ADX(")
        code = code.replace("CCI(", "talib.CCI(")
        code = code.replace("ATR(", "talib.ATR(")

        # 包装成完整的代码
        full_code = f"""
import talib
import pandas as pd
import numpy as np

def calculate_factor(df):
    '''计算因子: {expression}'''

    # 确保有必要的列
    if '{data_column}' not in df.columns:
        raise ValueError("数据中缺少 '{data_column}' 列")

    # 计算因子
    try:
        factor = {code}

        # 处理zscore特殊情况
        if isinstance(factor, pd.Series):
            # 检查是否是zscore表达式（以"("开头但没有匹配的右括号）
            if '{expression}'.startswith('zscore('):
                # 移除最外层的多余括号
                factor = (factor - factor.mean()) / (factor.std() + 1e-8)

        return factor
    except Exception as e:
        print(f"计算因子时出错: {{e}}")
        return pd.Series(index=df.index, dtype=float)
"""

        return full_code

    def validate_expression(self, expression: str) -> tuple[bool, str]:
        """
        验证因子表达式是否有效

        Args:
            expression: 因子表达式

        Returns:
            (是否有效, 错误信息)
        """
        # 基本语法检查
        if not expression or expression.strip() == "":
            return False, "表达式为空"

        # 检查括号匹配
        if expression.count("(") != expression.count(")"):
            return False, "括号不匹配"

        # 检查是否有非法字符
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()., _"
        )
        for char in expression:
            if char not in allowed_chars:
                return False, f"包含非法字符: {char}"

        # 检查是否有运算符
        has_operator = any(op in expression for op in ["+", "-", "*", "/"])
        has_function = any(func in expression for func in self.statistics.keys())

        if not (has_operator or has_function):
            return False, "表达式缺少运算符或函数"

        return True, ""

    def parse_expression(self, expression: str) -> Dict:
        """
        解析因子表达式，提取结构

        Args:
            expression: 因子表达式

        Returns:
            解析后的结构信息
        """
        structure = {
            "expression": expression,
            "components": [],
            "operators": [],
            "functions": [],
            "depth": 0,
        }

        # 提取运算符
        for op in self.operators.keys():
            if op in expression:
                structure["operators"].append(op)

        # 提取函数
        for func in self.statistics.keys():
            if f"{func}(" in expression:
                structure["functions"].append(func)

        for func in self.indicators.keys():
            if f"{func}(" in expression:
                structure["functions"].append(func)

        # 计算深度（括号嵌套层数）
        max_depth = 0
        current_depth = 0
        for char in expression:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1

        structure["depth"] = max_depth

        return structure

    def preselect_factors(
        self,
        factors: List[Dict],
        factor_data_map: Dict[str, pd.Series],
        return_data: pd.Series,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.5,
        min_valid_ratio: float = 0.7
    ) -> List[Dict]:
        """
        预筛选因子 - 根据IC、IR等指标筛选有潜力的因子

        Args:
            factors: 因子字典列表
            factor_data_map: 因子数据字典 {factor_name: factor_series}
            return_data: 收益率数据
            ic_threshold: IC阈值（绝对值）
            ir_threshold: IR阈值
            min_valid_ratio: 最小有效数据比例

        Returns:
            筛选后的因子列表
        """
        selected_factors = []

        for factor_info in factors:
            expression = factor_info["expression"]
            factor_name = f"factor_{len(selected_factors)}"

            if expression in factor_data_map:
                factor_values = factor_data_map[expression]

                # 对齐数据
                aligned_data = pd.DataFrame({
                    "factor": factor_values,
                    "return": return_data
                }).dropna()

                # 检查数据比例
                valid_ratio = len(aligned_data) / len(factor_values)
                if valid_ratio < min_valid_ratio:
                    continue

                # 计算IC
                ic = aligned_data["factor"].corr(aligned_data["return"])

                if pd.isna(ic) or abs(ic) < ic_threshold:
                    continue

                # 计算IR（IC均值/IC标准差）- 使用正确的滚动相关系数计算方法
                window = 20
                min_periods = 10
                rolling_ic_values = []

                for i in range(len(aligned_data)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1

                    window_factor = aligned_data["factor"].iloc[start_idx:end_idx]
                    window_return = aligned_data["return"].iloc[start_idx:end_idx]

                    valid_data = pd.DataFrame({
                        "factor": window_factor,
                        "return": window_return
                    }).dropna()

                    if len(valid_data) >= min_periods:
                        ic_val = valid_data["factor"].corr(valid_data["return"])
                        if not pd.isna(ic_val):
                            rolling_ic_values.append(ic_val)
                        else:
                            rolling_ic_values.append(np.nan)
                    else:
                        rolling_ic_values.append(np.nan)

                if rolling_ic_values:
                    ic_mean = np.nanmean(rolling_ic_values)
                    ic_std = np.nanstd(rolling_ic_values)

                    if pd.isna(ic_std) or ic_std == 0:
                        ir = 0
                    else:
                        ir = ic_mean / ic_std
                else:
                    ir = 0

                if ir < ir_threshold:
                    continue

                # 通过筛选
                factor_info["ic"] = float(ic)
                factor_info["ir"] = float(ir)
                factor_info["valid_ratio"] = float(valid_ratio)
                selected_factors.append(factor_info)

        return selected_factors

    def calculate_factor_metrics(
        self,
        factor_values: pd.Series,
        return_values: pd.Series
    ) -> Dict:
        """
        计算因子的质量指标

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列

        Returns:
            质量指标字典
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(aligned_data) < 10:
            return {
                "valid": False,
                "message": "数据不足"
            }

        # 计算IC
        ic = aligned_data["factor"].corr(aligned_data["return"])

        # 计算滚动IR - 使用正确的两变量滚动相关系数计算方法
        window = 20
        min_periods = 10
        rolling_ic_values = []

        for i in range(len(aligned_data)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1

            window_factor = aligned_data["factor"].iloc[start_idx:end_idx]
            window_return = aligned_data["return"].iloc[start_idx:end_idx]

            valid_data = pd.DataFrame({
                "factor": window_factor,
                "return": window_return
            }).dropna()

            if len(valid_data) >= min_periods:
                ic_val = valid_data["factor"].corr(valid_data["return"])
                if not pd.isna(ic_val):
                    rolling_ic_values.append(ic_val)
            else:
                rolling_ic_values.append(np.nan)

        rolling_ic = pd.Series(rolling_ic_values, index=aligned_data.index)

        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0

        # 计算胜率
        ic_win_rate = (rolling_ic > 0).sum() / rolling_ic.count() if rolling_ic.count() > 0 else 0

        # 计算因子分布特征
        factor_stats = {
            "mean": float(aligned_data["factor"].mean()),
            "std": float(aligned_data["factor"].std()),
            "skew": float(aligned_data["factor"].skew()),
            "kurtosis": float(aligned_data["factor"].kurtosis()),
        }

        return {
            "valid": True,
            "ic": float(ic),
            "ir": float(ir),
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_win_rate": float(ic_win_rate),
            "n_obs": len(aligned_data),
            "factor_stats": factor_stats,
        }


# 全局因子生成器服务实例
factor_generator_service = FactorGeneratorService()
