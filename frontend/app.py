"""
FactorFlow - 因子分析系统
基于 Streamlit 的股票因子分析平台
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.settings import settings
from backend.core.database import init_db
from backend.services.factor_service import factor_service
from backend.services.analysis_service import analysis_service
from backend.services.vectorbt_backtest_service import VectorBTBacktestService, check_vectorbt_available
from backend.repositories.backtest_repository import BacktestRepository


# ============ 初始化 ============
st.set_page_config(
    page_title="FactorFlow - 因子分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化数据库
@st.cache_resource
def init_database():
    """初始化数据库和预置因子"""
    init_db()
    factor_service.load_preset_factors()


init_database()


# ============ 辅助函数 ============
def normalize_price_data(df_dict: dict, stock_codes: list) -> pd.DataFrame:
    """归一化价格数据（初始值为100）"""
    normalized = {}
    for code in stock_codes:
        if code in df_dict:
            df = df_dict[code]
            if "close" in df.columns and len(df) > 0:
                normalized[code] = (df["close"] / df["close"].iloc[0]) * 100
    return pd.DataFrame(normalized)


def plot_price_chart(df_dict: dict, stock_codes: list, selected_factors: list = None):
    """绘制价格走势图"""
    normalized_df = normalize_price_data(df_dict, stock_codes)

    fig = go.Figure()
    for code in stock_codes:
        if code in normalized_df.columns:
            fig.add_trace(go.Scatter(
                x=normalized_df.index,
                y=normalized_df[code],
                mode="lines",
                name=f"{code} 价格",
                line=dict(width=2),
            ))

    fig.update_layout(
        title="股票价格走势（归一化，初始100）",
        xaxis_title="日期",
        yaxis_title="相对价格",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_factor_with_price(df: pd.DataFrame, factor_name: str, stock_code: str):
    """绘制因子与价格叠加图"""
    fig = go.Figure()

    # 添加价格（归一化）
    if "close" in df.columns:
        normalized_price = (df["close"] / df["close"].iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=df.index,
            y=normalized_price,
            mode="lines",
            name="价格 (归一化)",
            line=dict(color="blue", width=2),
            yaxis="y",
        ))

    # 添加因子
    if factor_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[factor_name],
            mode="lines",
            name=factor_name,
            line=dict(color="orange", width=1.5),
            yaxis="y2",
        ))

    fig.update_layout(
        title=f"{stock_code} - {factor_name} 与价格走势",
        xaxis_title="日期",
        yaxis=dict(title="相对价格", side="left"),
        yaxis2=dict(title=factor_name, side="right", overlaying="y"),
        hovermode="x unified",
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_ic_time_series(ic_data: dict):
    """绘制IC时间序列图"""
    if not ic_data:
        return

    fig = go.Figure()
    for factor_name, ic_series in ic_data.items():
        if isinstance(ic_series, dict):
            dates = list(ic_series.keys())
            values = list(ic_series.values())
        elif hasattr(ic_series, 'index'):
            dates = ic_series.index.astype(str).tolist()
            values = ic_series.values.tolist()
        else:
            continue

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name=factor_name,
        ))

    fig.update_layout(
        title="IC时间序列",
        xaxis_title="日期",
        yaxis_title="IC值",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_ic_heatmap(monthly_ic_df: pd.DataFrame, factor_name: str):
    """绘制月度IC热力图"""
    if monthly_ic_df.empty:
        st.warning(f"因子 {factor_name} 没有月度IC数据")
        return

    fig = go.Figure(data=go.Heatmap(
        z=monthly_ic_df.values,
        x=["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
        y=monthly_ic_df.index.astype(str),
        colorscale="RdBu",
        zmid=0,
        text=monthly_ic_df.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title=f"{factor_name} - 月度IC热力图",
        xaxis_title="月份",
        yaxis_title="年份",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_ir(rolling_ir: dict):
    """绘制滚动窗口IR图"""
    if not rolling_ir:
        return

    fig = go.Figure()
    for factor_name, ir_series in rolling_ir.items():
        if isinstance(ir_series, dict):
            dates = list(ir_series.keys())
            values = list(ir_series.values())
        elif hasattr(ir_series, 'index'):
            dates = ir_series.index.astype(str).tolist()
            values = ir_series.values.tolist()
        else:
            continue

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name=factor_name,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="滚动窗口IR（60日）",
        xaxis_title="日期",
        yaxis_title="IR值",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_shap_importance(feature_importance: list):
    """绘制SHAP特征重要性柱状图"""
    df = pd.DataFrame(feature_importance)
    df = df.sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
    ))

    fig.update_layout(
        title="SHAP特征重要性",
        xaxis_title="重要性（mean(|SHAP|)）",
        yaxis_title="特征名称",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(factor_data: dict, factor_names: list, stock_code: str = None):
    """绘制因子相关性矩阵热力图"""
    # 合并所有股票的因子数据
    all_dfs = []
    for code, df in factor_data.items():
        df_copy = df.copy()
        df_copy["stock_code"] = code
        all_dfs.append(df_copy)

    merged_df = pd.concat(all_dfs, ignore_index=False)

    # 选择因子列
    factor_cols = [col for col in factor_names if col in merged_df.columns]

    # 添加收益率列（如果存在）
    if "future_return_1" in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df["未来收益率"] = merged_df["future_return_1"]
        factor_cols.append("未来收益率")
    elif "close" in merged_df.columns:
        # 使用价格收益率作为替代
        merged_df = merged_df.copy()
        merged_df["未来收益率"] = merged_df["close"].pct_change(1)
        factor_cols.append("未来收益率")

    if not factor_cols:
        st.warning("没有可用的因子数据")
        return

    # 计算相关性矩阵
    corr_df = merged_df[factor_cols].corr()

    # 绘制热力图
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_df.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="相关系数"),
    ))

    title = f"因子相关性矩阵 ({stock_code})" if stock_code else "因子相关性矩阵 (多股票)"
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        height=500,
        width=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示相关性统计说明
    st.caption("""
    💡 **说明**：相关系数范围为[-1, 1]，绝对值越接近1表示相关性越强。
    - 正值（红色）：正相关，因子值上升时另一个也上升
    - 负值（蓝色）：负相关，因子值上升时另一个下降
    - 接近0（白色）：无明显相关性
    """)


# ============ 回测可视化函数 ============

def plot_equity_curve(equity_series, benchmark_series: pd.Series = None, title: str = "净值曲线"):
    """绘制净值曲线图"""
    fig = go.Figure()

    # 策略净值
    if hasattr(equity_series, 'index'):
        x_values = equity_series.index.astype(str) if hasattr(equity_series.index, 'astype') else equity_series.index
        y_values = equity_series.values

        # 如果是2D数组（多资产情况），求和得到组合净值
        if len(y_values.shape) > 1 and y_values.shape[1] > 1:
            y_values = y_values.sum(axis=1)
    else:
        x_values = list(equity_series.keys())
        y_values = list(equity_series.values())

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='策略净值',
        line=dict(color='blue', width=2),
    ))

    # 基准净值（如果有）
    if benchmark_series is not None and len(benchmark_series) > 0:
        if hasattr(benchmark_series, 'index'):
            x_bench = benchmark_series.index.astype(str) if hasattr(benchmark_series.index, 'astype') else benchmark_series.index
            y_bench = benchmark_series.values
        else:
            x_bench = list(benchmark_series.keys())
            y_bench = list(benchmark_series.values())

        fig.add_trace(go.Scatter(
            x=x_bench,
            y=y_bench,
            mode='lines',
            name='基准净值',
            line=dict(color='gray', width=1, dash='dash'),
        ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='净值',
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown_curve(equity_series, title: str = "回撤曲线"):
    """绘制回撤曲线"""
    # 计算回撤
    equity_array = equity_series.values if hasattr(equity_series, 'values') else list(equity_series.values())

    # 如果是2D数组（多资产情况），求和得到组合净值
    if len(equity_array.shape) > 1 and equity_array.shape[1] > 1:
        equity_array = equity_array.sum(axis=1)

    equity_cummax = pd.Series(equity_array).cummax()
    drawdown = (equity_cummax - pd.Series(equity_array)) / equity_cummax

    x_values = equity_series.index.astype(str) if hasattr(equity_series, 'index') and hasattr(equity_series.index, 'astype') else list(range(len(drawdown)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=drawdown.values * 100,  # 转换为百分比
        mode='lines',
        name='回撤',
        fill='tozeroy',
        line=dict(color='red', width=1.5),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        hovermode='x unified',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_returns_heatmap(returns, title: str = "月度收益热力图"):
    """绘制月度收益热力图"""
    if len(returns) == 0:
        st.warning("没有收益率数据")
        return

    # 确保索引是datetime
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # 如果是DataFrame（多资产），计算组合收益
    if isinstance(returns, pd.DataFrame):
        # 对多资产求和或平均
        returns = returns.sum(axis=1) if len(returns.columns) > 1 else returns.iloc[:, 0]

    # 计算月度收益
    monthly_returns = (1 + returns).resample('M').prod() - 1

    # 创建透视表
    if isinstance(monthly_returns, pd.Series):
        monthly_df = monthly_returns.to_frame(name='return')
    else:
        # 如果monthly_returns还是DataFrame，取第一列
        monthly_df = monthly_returns.iloc[:, [0]].copy()
        monthly_df.columns = ['return']

    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month

    pivot_table = monthly_df.pivot(index='year', columns='month', values='return') * 100

    # 月份名称
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=month_names,
        y=pivot_table.index.astype(str),
        colorscale='RdBu',
        zmid=0,
        text=np.round(pivot_table.values, 2),
        texttemplate='%{text}%',
        textfont={'size': 10},
    ))

    fig.update_layout(
        title=title,
        xaxis_title='月份',
        yaxis_title='年份',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_quantile_returns_comparison(quantile_returns: dict, title: str = "各分层收益对比"):
    """绘制各分层收益对比图"""
    fig = go.Figure()

    for quantile_name, returns in quantile_returns.items():
        if len(returns) == 0:
            continue

        # 计算累计收益
        cumulative_returns = (1 + returns.fillna(0)).cumprod()

        x_values = cumulative_returns.index.astype(str) if hasattr(cumulative_returns.index, 'astype') else cumulative_returns.index

        fig.add_trace(go.Scatter(
            x=x_values,
            y=cumulative_returns.values,
            mode='lines',
            name=quantile_name,
        ))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='累计收益',
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_backtest_metrics(metrics: dict):
    """显示回测指标"""
    # 创建指标字典（包含说明）
    metric_info = {
        'total_return': {
            'label': '累计收益率',
            'format': '{:.2%}',
            'help': '整个回测期间的总收益率，使用复利计算'
        },
        'annual_return': {
            'label': '年化收益率',
            'format': '{:.2%}',
            'help': '将累计收益率按时间折算为年化收益率'
        },
        'volatility': {
            'label': '年化波动率',
            'format': '{:.2%}',
            'help': '收益率序列的年化标准差，衡量策略风险水平'
        },
        'sharpe_ratio': {
            'label': '夏普比率',
            'format': '{:.4f}',
            'help': '>1为良好，>2为优秀，衡量单位风险的超额收益'
        },
        'max_drawdown': {
            'label': '最大回撤',
            'format': '{:.2%}',
            'help': '从历史最高点到最低点的最大跌幅'
        },
        'calmar_ratio': {
            'label': '卡玛比率',
            'format': '{:.4f}',
            'help': '年化收益与最大回撤的比值，>1为优秀'
        },
        'win_rate': {
            'label': '胜率',
            'format': '{:.2%}',
            'help': '盈利交易日占总交易日的比例'
        },
        'sortino_ratio': {
            'label': '索提诺比率',
            'format': '{:.4f}',
            'help': '类似夏普比率，但只考虑下行风险（负收益的波动）'
        },
        'var_95': {
            'label': 'VaR (95%)',
            'format': '{:.2%}',
            'help': '在95%置信水平下，单日最大可能损失'
        },
        'cvar_95': {
            'label': 'CVaR (95%)',
            'format': '{:.2%}',
            'help': '超过VaR时的平均损失（条件VaR或期望损失）'
        },
    }

    # 按两列显示指标
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for idx, (key, info) in enumerate(metric_info.items()):
        if key in metrics and metrics[key] is not None:
            with cols[idx % 4]:
                st.metric(info['label'], info['format'].format(metrics[key]), help=info['help'])


def plot_factor_decay_curve(decay_data: dict, title: str = "因子衰减曲线"):
    """绘制因子衰减曲线"""
    periods = []
    ic_values = []

    for key, value in decay_data.items():
        if key.startswith('period_'):
            period = int(key.split('_')[1])
            if not np.isnan(value):
                periods.append(period)
                ic_values.append(value)

    if not periods:
        st.warning("没有衰减数据")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods,
        y=ic_values,
        mode='lines+markers',
        name='IC值',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
    ))

    fig.add_hline(y=0, line_dash='dash', line_color='gray')

    fig.update_layout(
        title=title,
        xaxis_title='向前周期',
        yaxis_title='IC值',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============ 策略回测页面 ============
def backtest_main():
    """策略回测主页面"""
    st.title("策略回测")

    # 检查vectorbt是否可用
    if not check_vectorbt_available():
        st.error("❌ VectorBT未安装，请运行以下命令安装：")
        st.code("pip install vectorbt")
        st.stop()

    # 初始化服务
    if "backtest_repo" not in st.session_state:
        st.session_state.backtest_repo = BacktestRepository()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("回测配置")

        # 数据模式
        data_mode = st.radio("数据模式", ["单股票", "股票池"])

        # 股票代码输入
        if data_mode == "单股票":
            stock_codes_input = st.text_input(
                "股票代码",
                value="000001",
                help="支持格式：000001 或 000001.SZ"
            )
            stock_codes = [stock_codes_input.strip()]
        else:
            stock_codes_input = st.text_area(
                "股票代码（每行一个）",
                value="000001\n600000",
                height=100,
            )
            stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

        # 选择因子
        all_factors = factor_service.get_all_factors()
        factor_options = {f["name"]: f["description"] for f in all_factors}
        selected_factors = st.multiselect(
            "选择因子",
            options=list(factor_options.keys()),
            format_func=lambda x: f"{x} - {factor_options.get(x, '')}",
            default=[],
        )

        # 回测参数
        st.markdown("### 回测参数")
        initial_capital = st.number_input("初始资金", value=1000000, min_value=10000, step=10000)
        commission_rate = st.number_input("费率(%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01) / 100
        slippage = st.number_input("滑点(%)", value=0.00, min_value=0.0, max_value=1.0, step=0.01, help="买卖时的价格滑点百分比，例如0.01表示万分之一") / 100

        # 策略类型
        strategy_type = st.selectbox(
            "策略类型",
            options=["单因子分层", "多因子组合"],
            index=0,
        )

        if strategy_type == "单因子分层":
            percentile = st.slider("分层阈值（百分位）", 0, 100, 50)
            direction = st.selectbox("交易方向", ["long", "short"])
            n_quantiles = st.slider("分层数量", 3, 10, 5)
        else:
            weight_method = st.selectbox(
                "权重分配",
                ["等权重", "风险平价"],
                index=0,
            )
            percentile = st.slider("组合分层阈值（百分位）", 0, 100, 50)
            direction = st.selectbox("交易方向", ["long", "short"])

        # 时间范围
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365*3)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
        with col2:
            end_date = st.date_input("结束日期", value=default_end)

        # 运行回测按钮
        run_backtest = st.button("运行回测", type="primary", use_container_width=True)

        # 历史回测记录
        st.divider()
        if st.button("查看历史记录"):
            st.session_state.show_backtest_history = True

    # 显示历史记录
    if st.session_state.get("show_backtest_history", False):
        st.subheader("历史回测记录")

        history = st.session_state.backtest_repo.get_history(limit=10)

        if not history:
            st.info("暂无历史记录")
        else:
            history_data = []
            for record in history:
                history_data.append({
                    "策略名称": record.strategy_name,
                    "因子组合": record.factor_combination[:50] + "..." if len(record.factor_combination) > 50 else record.factor_combination,
                    "时间范围": f"{record.start_date} ~ {record.end_date}",
                    "累计收益": f"{record.total_return:.2%}" if record.total_return else "N/A",
                    "夏普比率": f"{record.sharpe_ratio:.4f}" if record.sharpe_ratio else "N/A",
                    "最大回撤": f"{record.max_drawdown:.2%}" if record.max_drawdown else "N/A",
                    "创建时间": record.created_at.strftime("%Y-%m-%d %H:%M"),
                })

            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True,
            )

            # 删除记录
            with st.expander("删除历史记录"):
                record_to_delete = st.selectbox(
                    "选择要删除的记录",
                    options=range(len(history)),
                    format_func=lambda i: f"{history[i].strategy_name} - {history[i].created_at.strftime('%Y-%m-%d')}",
                )
                if st.button("确认删除选中记录", type="secondary"):
                    if st.session_state.backtest_repo.delete_by_id(history[record_to_delete].id):
                        st.success("删除成功")
                        st.rerun()
                    else:
                        st.error("删除失败")

        if st.button("返回回测配置"):
            st.session_state.show_backtest_history = False
            st.rerun()

        return

    # 执行回测
    if run_backtest:
        if not stock_codes or not stock_codes[0]:
            st.error("请输入股票代码")
        elif not selected_factors:
            st.error("请至少选择一个因子")
        elif strategy_type == "单因子分层" and len(selected_factors) > 1:
            st.warning("单因子分层策略只选择一个因子，将使用第一个因子")
            selected_factors = [selected_factors[0]]
        else:
            with st.spinner("正在运行回测，请稍候..."):
                try:
                    # 获取数据
                    from backend.services.data_service import data_service

                    all_factor_data = {}
                    all_price_data = {}

                    for stock_code in stock_codes:
                        # 获取股票数据
                        stock_data = data_service.get_stock_data(
                            stock_code,
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )

                        if stock_data is not None and len(stock_data) > 0:
                            # 计算因子
                            factor_calculator = factor_service.calculator
                            for factor_name in selected_factors:
                                factor_code = None
                                for f in all_factors:
                                    if f["name"] == factor_name:
                                        factor_code = f["code"]
                                        break

                                if factor_code:
                                    factor_values = factor_calculator.calculate(
                                        stock_data, factor_code
                                    )
                                    stock_data[factor_name] = factor_values

                            all_factor_data[stock_code] = stock_data

                    if not all_factor_data:
                        st.error("未获取到有效数据，请检查股票代码和时间范围")
                        return

                    # 使用VectorBT回测引擎执行回测
                    backtest_service = VectorBTBacktestService(
                        initial_capital=initial_capital,
                        commission_rate=commission_rate,
                        slippage=slippage,
                    )

                    # 判断是单股票还是股票池
                    is_single_stock = len(all_factor_data) == 1

                    if strategy_type == "单因子分层":
                        factor_name = selected_factors[0]

                        if is_single_stock:
                            # 单股票：使用时序回测
                            df = list(all_factor_data.values())[0].copy()
                            result = backtest_service.single_factor_backtest(
                                df=df,
                                factor_name=factor_name,
                                percentile=percentile,
                                direction=direction,
                                n_quantiles=n_quantiles,
                            )
                            # 单因子时序回测的指标已在回测方法中计算
                            metrics = {k: v for k, v in result.items() if k in [
                                "total_return", "annual_return", "volatility", "sharpe_ratio",
                                "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                                "var_95", "cvar_95"
                            ]}
                        else:
                            # 股票池：使用横截面回测
                            # 合并数据
                            merged_list = []
                            for code, data in all_factor_data.items():
                                data_copy = data.copy()
                                data_copy["stock_code"] = code
                                # 确保有日期列
                                if "date" not in data_copy.columns:
                                    data_copy = data_copy.reset_index()
                                merged_list.append(data_copy)

                            df = pd.concat(merged_list, ignore_index=True)

                            # 使用横截面回测
                            top_percentile = (100 - percentile) / 100.0  # 转换为选择比例
                            result = backtest_service.cross_sectional_backtest(
                                df=df,
                                factor_name=factor_name,
                                top_percentile=top_percentile,
                                direction=direction,
                            )
                            # 横截面回测的指标已在回测方法中计算
                            metrics = {k: v for k, v in result.items() if k in [
                                "total_return", "annual_return", "volatility", "sharpe_ratio",
                                "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                                "var_95", "cvar_95"
                            ]}
                    else:
                        # 多因子组合回测（目前仅支持单股票时序回测）
                        if not is_single_stock:
                            st.warning("多因子组合策略目前仅支持单股票模式，将使用第一只股票")
                            df = list(all_factor_data.values())[0].copy()
                        else:
                            df = list(all_factor_data.values())[0].copy()

                        weight_map = {"等权重": "equal_weight", "风险平价": "risk_parity"}
                        result = backtest_service.multi_factor_backtest(
                            df=df,
                            factor_names=selected_factors,
                            method=weight_map[weight_method],
                            percentile=percentile,
                            direction=direction,
                        )
                        # 多因子回测的指标已在回测方法中计算
                        metrics = {k: v for k, v in result.items() if k in [
                            "total_return", "annual_return", "volatility", "sharpe_ratio",
                            "max_drawdown", "calmar_ratio", "win_rate", "sortino_ratio",
                            "var_95", "cvar_95"
                        ]}

                    # 保存到session state
                    st.session_state.backtest_result = {
                        "result": result,
                        "metrics": metrics,
                        "config": {
                            "strategy_type": strategy_type,
                            "factors": selected_factors,
                            "stock_codes": stock_codes,
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                            "initial_capital": initial_capital,
                        },
                        "raw_data": all_factor_data  # 保存原始的价格和因子数据
                    }

                    st.success("回测完成！")

                except Exception as e:
                    st.error(f"回测失败: {e}")
                    import traceback
                    st.error(traceback.format_exc())

    # 显示回测结果
    if "backtest_result" in st.session_state:
        result = st.session_state.backtest_result["result"]
        metrics = st.session_state.backtest_result["metrics"]
        config = st.session_state.backtest_result["config"]

        st.subheader("回测结果")

        # Tab1: 概览, Tab2: 净值分析, Tab3: 分层分析, Tab4: 交易日志
        tab1, tab2, tab3, tab4 = st.tabs(["概览", "净值分析", "分层分析", "交易日志"])

        with tab1:
            # 策略说明
            with st.expander("📖 策略说明与指标解释", expanded=False):
                st.markdown("""
                ### 策略概述（基于VectorBT回测引擎）

                **回测引擎说明**：
                本系统使用 VectorBT 专业回测引擎，提供高效、准确的向量化回测计算。
                VectorBT 基于 pandas 和 NumPy，采用事件驱动的回测框架，能够精确模拟真实交易环境。

                ---

                ### 📊 单因子分层策略交易逻辑

                **适用场景**：单股票回测、单一因子的择时策略

                **核心步骤**：

                1️⃣ **因子排名计算**
                - 使用 **252天滚动窗口** 计算因子值的百分位排名
                - 排名范围 0-1 之间（0% 最低，100% 最高）
                - 公式：`factor_rank = factor.rolling(252).rank(pct=True)`

                2️⃣ **交易信号生成**（以做多为例）
                | 条件 | 信号 | 持仓状态 |
                |------|------|----------|
                | 因子排名 ≥ 阈值（默认50%） | 买入（Entry） | 满仓（100%） |
                | 因子排名 < 阈值 | 卖出（Exit） | 空仓（0%） |

                3️⃣ **交易执行**
                - 二元交易策略：要么满仓，要么空仓
                - 当信号触发时，VectorBT 自动执行交易
                - 自动扣除交易费用和滑点

                **示例**：
                ```
                日期       因子值    因子排名   阈值50%   信号   持仓
                2024-01-01   0.8      0.85      ✓       买入   100%
                2024-01-02   0.7      0.75      ✓       持有   100%
                2024-01-03   0.3      0.45      ✗       卖出   0%
                ```

                ---

                ### 🎯 多因子组合策略交易逻辑

                **适用场景**：多个因子共同决策、提升策略稳定性

                **核心步骤**：

                1️⃣ **因子标准化（Z-score Normalization）**
                - 消除不同因子间的量纲差异
                - 公式：`normalized_factor = (factor - mean) / std`
                - 所有因子转换为均值为0、标准差为1的分布

                2️⃣ **综合得分计算**
                | 权重方法 | 计算逻辑 | 适用场景 |
                |----------|----------|----------|
                | 等权重 | 所有标准化因子简单平均 | 因子重要性相近 |
                | 风险平价 | 波动率小的因子权重更大 | 追求风险均衡 |
                | IC加权 | 与收益率相关性高的因子权重更大 | 基于历史效果 |

                3️⃣ **交易执行**
                - 将综合得分作为"超级因子"
                - 后续逻辑与单因子策略完全相同
                - 按综合得分的百分位排名生成买卖信号

                **示例**：
                ```
                日期       PE因子   ROE因子   MOM因子   综合得分   排名   信号
                2024-01-01  -0.5     1.2      0.8       0.5       0.70  买入
                2024-01-02  -0.3     1.1      0.9       0.57      0.75  持有
                2024-01-03   0.2     0.3      0.1       0.2       0.45  卖出
                ```

                ---

                ### 🏢 股票池横截面回测逻辑

                **适用场景**：多股票选股策略、因子有效性验证

                **核心步骤**：
                1. 每个交易日计算所有股票的因子值排名
                2. 选择排名前 N% 的股票（做多）或后 N% 的股票（做空）
                3. 对选中的股票进行等权重配置，每日调仓
                4. 使用 VectorBT 的多资产回测功能，精确计算每只股票的收益贡献

                ---

                ### 📈 关键指标说明（VectorBT标准计算）

                - **累计收益率**：投资组合在整个回测期间的总收益率，使用复利计算
                - **年化收益率**：将累计收益率按时间折算为年化收益率
                - **年化波动率**：收益率序列的年化标准差，衡量策略风险水平
                - **夏普比率**：(年化收益 - 无风险利率) / 年化波动率
                  - 衡量单位风险的超额收益，> 1 为良好，> 2 为优秀
                - **最大回撤**：从历史最高净值到最低点的最大跌幅
                - **卡玛比率**：年化收益率 / 最大回撤，衡量单位回撤风险获得的收益
                - **胜率**：盈利交易日占总交易日的比例
                - **索提诺比率**：类似夏普比率，但只考虑下行风险（负收益的波动）
                - **VaR (95%)**：在95%置信水平下，单日最大可能损失
                - **CVaR (95%)**：超过VaR时的平均损失（条件VaR或期望损失）
                """)

            st.markdown("### 性能指标")
            display_backtest_metrics(metrics)

            st.markdown("### 回测配置")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**策略类型**:", config["strategy_type"])
                st.write("**因子**:", ", ".join(config["factors"]))
            with col2:
                st.write("**股票代码**:", ", ".join(config["stock_codes"]))
                st.write("**时间范围**:", f"{config['start_date']} ~ {config['end_date']}")

            # 保存回测结果
            if st.button("保存回测结果", type="primary"):
                try:
                    save_data = {
                        "strategy_name": f"{config['strategy_type']}_{','.join(config['factors'])}",
                        "factor_combination": str(config["factors"]),
                        "start_date": config["start_date"],
                        "end_date": config["end_date"],
                        "initial_capital": config["initial_capital"],
                        "final_capital": result["equity_curve"].iloc[-1] if hasattr(result["equity_curve"], "iloc") else list(result["equity_curve"].values())[-1],
                        **metrics,
                        "trades_count": result.get("trades_count", 0),
                    }

                    saved_record = st.session_state.backtest_repo.save_result(save_data)
                    st.success(f"回测结果已保存 (ID: {saved_record.id})")
                except Exception as e:
                    st.error(f"保存失败: {e}")

        with tab2:
            # 净值分析说明
            with st.expander("📖 净值分析说明（VectorBT计算）", expanded=False):
                st.markdown("""
                ### 净值曲线

                **计算逻辑（VectorBT）**：
                - 使用 VectorBT 的 `Portfolio.value()` 方法计算
                - 净值 = 初始资金 + 所有交易收益的累计和
                - 精确模拟每日持仓变化、交易成本、资金流动
                - 可用于观察策略的累积表现和资金使用效率

                ### 回撤曲线

                **计算逻辑（VectorBT）**：
                - 回撤 = (历史最高净值 - 当前净值) / 历史最高净值
                - 使用 VectorBT 的 `drawdown()` 方法计算
                - 衡量从历史最高点到当前点的跌幅
                - 最大回撤是评估策略风险的重要指标

                ### 月度收益热力图

                **计算逻辑**：
                - 将 VectorBT 计算的每日收益率按月汇总
                - 使用月度复利计算：(1 + 日收益率) 累乘 - 1
                - 使用颜色深浅表示收益正负（红色为负，蓝色为正）
                - 可用于识别策略的季节性规律和周期性特征

                **使用建议**：
                - 净值曲线应持续向上，避免大幅波动
                - 回撤应快速恢复，避免长期处于低位
                - 月度收益应保持相对稳定，避免大起大落
                """)

            st.markdown("### 净值曲线")
            plot_equity_curve(result["equity_curve"], title="策略净值曲线")

            st.markdown("### 回撤曲线")
            plot_drawdown_curve(result["equity_curve"], title="策略回撤曲线")

            if "portfolio_returns" in result:
                st.markdown("### 月度收益热力图")
                plot_monthly_returns_heatmap(result["portfolio_returns"], title="月度收益率")

        with tab3:
            # 分层分析说明
            with st.expander("📖 分层分析说明（VectorBT计算）", expanded=False):
                st.markdown("""
                ### 各分层收益对比

                **分层逻辑（VectorBT实现）**：
                - 根据因子值将数据分为N层（通常为5层）
                - 使用滚动窗口（252天）计算因子值的动态分位数排名
                - Q1表示因子值最低的组，Q5表示因子值最高的组
                - 使用 `pd.qcut()` 确保等频分层，每组数据量相同

                **计算逻辑**：
                - 每层分别计算收益，基于 VectorBT 的信号回测结果
                - 使用 `rolling(252).rank(pct=True)` 计算动态排名
                - 每日重新分层，动态调整持仓
                - 用于检验因子的单调性和预测能力

                **解读方法**：
                - **有效因子**：Q1 < Q2 < Q3 < Q4 < Q5（单调递增）
                - **无效因子**：各层收益无明显差异
                - **反向因子**：Q1 > Q2 > Q3 > Q4 > Q5（单调递减）

                ### 各分层统计

                **指标说明**：
                - **日均收益**：该层所有交易日的平均日收益率（VectorBT精确计算）
                - **年化收益**：日均收益 × 252（年化交易日数）
                - **夏普比率**：衡量该层风险调整后的收益表现
                - **胜率**：该层盈利交易日占比

                **使用建议**：
                - 优先选择各层收益差异明显的因子
                - Q5与Q1的收益差越大，因子区分度越好
                - 夏普比率在各层都应该相对稳定
                - 有效因子的分层收益应呈现明显的单调性
                """)

            if "quantile_returns" in result and result["quantile_returns"]:
                st.markdown("### 各分层收益对比")
                plot_quantile_returns_comparison(result["quantile_returns"])

                # 各分层统计
                st.markdown("### 各分层统计")
                from backend.services.statistics_service import StatisticsService
                stats_service = StatisticsService()

                quantile_stats = stats_service.analyze_quantile_returns(result["quantile_returns"])

                stats_data = []
                for quantile_name, stats in quantile_stats.items():
                    stats_data.append({
                        "分层": quantile_name,
                        "日均收益": stats['mean'],
                        "年化收益": stats['annual_return'],
                        "夏普比率": stats['sharpe'],
                        "胜率": stats['win_rate'],
                    })

                st.dataframe(
                    pd.DataFrame(stats_data),
                    column_config={
                        "分层": st.column_config.TextColumn("分层", width="small"),
                        "日均收益": st.column_config.NumberColumn(
                            "日均收益",
                            format="%.4f",
                            help="📖 **日均收益**：该层所有交易日的平均日收益率\n\n- 正值表示平均盈利，负值表示平均亏损\n- 绝对值越大，该层表现越极端\n- 用于比较不同层的平均表现"
                        ),
                        "年化收益": st.column_config.NumberColumn(
                            "年化收益",
                            format="%.2%",
                            help="📖 **年化收益**：将日均收益按年化交易日数推算\n\n- 公式：日均收益 × 252\n- 便于与年度基准比较\n- 年化收益越高，该层长期表现越好"
                        ),
                        "夏普比率": st.column_config.NumberColumn(
                            "夏普比率",
                            format="%.4f",
                            help="📖 **夏普比率**：风险调整后的收益指标\n\n- >1 为良好，>2 为优秀\n- 衡量单位风险的超额收益\n- 公式：(年化收益 - 无风险利率) / 年化波动率"
                        ),
                        "胜率": st.column_config.NumberColumn(
                            "胜率",
                            format="%.2%",
                            help="📖 **胜率**：盈利交易日占比\n\n- 盈利天数 / 总交易天数\n- >50% 表示盈利天数多于亏损天数\n- 衡量该层的胜率稳定性"
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("当前回测结果没有分层数据（股票池横截面回测不显示分层）")

        with tab4:
            st.subheader("交易日志")

            # 交易日志说明
            with st.expander("📖 交易日志说明", expanded=False):
                st.markdown("""
                ### 交易记录详情

                **数据来源**：VectorBT 回测引擎自动记录每笔交易

                **字段说明**：
                - **日期**：交易发生的日期
                - **方向**：买入(Long)或卖出(Short)
                - **价格**：成交价格
                - **数量**：交易数量（股数或单位）
                - **价值**：交易价值（价格×数量）
                - **收益**：该笔交易的盈亏
                - **收益率**：该笔交易的收益率

                **使用建议**：
                - 查看每笔交易的详细记录
                - 分析交易的盈利和亏损模式
                - 识别交易频繁的时段
                """)

            # 绘制行情和指标图表
            if "raw_data" in st.session_state.backtest_result and st.session_state.backtest_result["raw_data"]:
                raw_data = st.session_state.backtest_result["raw_data"]
                factors = config.get("factors", [])

                if len(raw_data) > 0:
                    st.markdown("---")
                    st.markdown("### 📈 行情与指标分析")

                    # 如果有多只股票，为每只股票创建一个tab
                    stock_codes_list = list(raw_data.keys())

                    if len(stock_codes_list) > 1:
                        stock_tabs = st.tabs([f"{code}" for code in stock_codes_list])
                    else:
                        stock_tabs = [st.container()]

                    for idx, stock_code in enumerate(stock_codes_list):
                        stock_data = raw_data[stock_code]

                        with stock_tabs[idx]:
                            # 获取价格数据
                            if "close" in stock_data.columns:
                                prices = stock_data["close"]
                                dates = stock_data.index

                                # 准备交易信号标记
                                buy_signals = []
                                sell_signals = []

                                # 从trades中提取交易信号
                                if "trades" in result and result["trades"] is not None:
                                    trades_df = result["trades"]

                                    # 过滤当前股票的交易记录
                                    # 对于单股票回测，VectorBT的Column字段可能是"close"或索引，不是股票代码
                                    if "股票代码" in trades_df.columns:
                                        # 尝试按股票代码过滤
                                        stock_trades = trades_df[trades_df["股票代码"] == stock_code]
                                        # 如果过滤后没有数据，且只有一只股票，则使用所有交易
                                        if len(stock_trades) == 0 and len(stock_codes_list) == 1:
                                            # 可能是单股票回测，Column字段值为0或"close"
                                            stock_trades = trades_df
                                    else:
                                        stock_trades = trades_df

                                    # 使用iterrows()安全地遍历交易记录
                                    for trade_idx, trade in stock_trades.iterrows():
                                        # trade_idx是入场时间（索引）
                                        if "方向" in trade.index and "入场价格" in trade.index:
                                            entry_price = trade["入场价格"]
                                            exit_price = trade.get("出场价格")
                                            exit_timestamp = trade.get("出场时间")
                                            direction = trade["方向"]

                                            # 跳过NaN值
                                            if pd.isna(entry_price) or pd.isna(direction):
                                                continue

                                            # 添加入场信号
                                            try:
                                                entry_price = float(entry_price)
                                                if direction == "做多":
                                                    buy_signals.append({
                                                        "date": trade_idx,
                                                        "price": entry_price
                                                    })
                                                elif direction == "做空":
                                                    sell_signals.append({
                                                        "date": trade_idx,
                                                        "price": entry_price
                                                    })
                                            except (ValueError, TypeError):
                                                pass

                                            # 添加出场信号（如果存在）
                                            if not pd.isna(exit_timestamp) and not pd.isna(exit_price):
                                                try:
                                                    exit_price = float(exit_price)
                                                    # 确保出场时间是datetime类型
                                                    if not isinstance(exit_timestamp, pd.Timestamp):
                                                        exit_timestamp = pd.to_datetime(exit_timestamp)

                                                    # 做多平仓是卖出，做空平仓是买入
                                                    if direction == "做多":
                                                        sell_signals.append({
                                                            "date": exit_timestamp,
                                                            "price": exit_price
                                                        })
                                                    elif direction == "做空":
                                                        buy_signals.append({
                                                            "date": exit_timestamp,
                                                            "price": exit_price
                                                        })
                                                except (ValueError, TypeError):
                                                    pass

                                # 创建子图，共用x轴
                                from plotly.subplots import make_subplots
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.08,
                                    row_heights=[0.6, 0.4],
                                    subplot_titles=("行情走势与交易信号", "")
                                )

                                # 添加价格线
                                fig.add_trace(
                                    go.Scatter(
                                        x=dates,
                                        y=prices,
                                        mode='lines',
                                        name='价格',
                                        line=dict(color='#1f77b4', width=1.5)
                                    ),
                                    row=1, col=1
                                )

                                # 添加买入信号
                                if buy_signals:
                                    buy_dates = [s["date"] for s in buy_signals]
                                    buy_prices = [s["price"] for s in buy_signals]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=buy_dates,
                                            y=buy_prices,
                                            mode='markers',
                                            name='买入',
                                            marker=dict(
                                                symbol='triangle-up',
                                                size=14,
                                                color='red',
                                                line=dict(width=2, color='darkred')
                                            ),
                                            hovertemplate='买入<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=1
                                    )

                                # 添加卖出信号
                                if sell_signals:
                                    sell_dates = [s["date"] for s in sell_signals]
                                    sell_prices = [s["price"] for s in sell_signals]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=sell_dates,
                                            y=sell_prices,
                                            mode='markers',
                                            name='卖出',
                                            marker=dict(
                                                symbol='triangle-down',
                                                size=14,
                                                color='green',
                                                line=dict(width=2, color='darkgreen')
                                            ),
                                            hovertemplate='卖出<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=1
                                    )

                                # 添加因子线（归一化处理）
                                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                                for factor_idx, factor_name in enumerate(factors):
                                    if factor_name in stock_data.columns:
                                        factor_values = stock_data[factor_name]

                                        # 归一化处理到 [0, 1] 区间
                                        scaler = MinMaxScaler()
                                        factor_values_normalized = scaler.fit_transform(factor_values.values.reshape(-1, 1)).flatten()

                                        # 使用第二个y轴（对应row=2）
                                        fig.add_trace(
                                            go.Scatter(
                                                x=dates,
                                                y=factor_values_normalized,
                                                mode='lines',
                                                name=f"{factor_name} (归一化)",
                                                line=dict(color=colors[factor_idx % len(colors)], width=1.5)
                                            ),
                                            row=2, col=1
                                        )

                                # 更新布局
                                fig.update_layout(
                                    height=700,
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    # 价格曲线不显示x轴标题
                                    xaxis=dict(showticklabels=True),
                                    # 因子曲线不显示标题
                                    xaxis2=dict(showticklabels=True),
                                    yaxis=dict(title='价格', side='left'),
                                    yaxis2=dict(
                                        title='因子值 (归一化)',
                                        side='left',
                                        showgrid=True
                                    ),
                                    margin=dict(l=60, r=60, t=40, b=60)
                                )

                                st.plotly_chart(fig, use_container_width=True)

            # 检查是否有交易记录
            if "trades" in result and result["trades"] is not None:
                trades_df = result["trades"]

                if len(trades_df) > 0:
                    st.markdown("---")
                    # 显示统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总交易次数", len(trades_df))
                    with col2:
                        profit_count = len(trades_df[trades_df["收益"] > 0]) if "收益" in trades_df.columns else 0
                        st.metric("盈利交易", profit_count)
                    with col3:
                        loss_count = len(trades_df[trades_df["收益"] < 0]) if "收益" in trades_df.columns else 0
                        st.metric("亏损交易", loss_count)
                    with col4:
                        win_rate = profit_count / len(trades_df) if len(trades_df) > 0 else 0
                        st.metric("胜率", f"{win_rate:.2%}")

                    st.markdown("---")

                    # 交易记录表格
                    st.markdown("### 交易明细")

                    # 隐藏的列
                    hidden_columns = ['方向', '状态', 'Exit Trade Id', 'Position Id']

                    # 常见字段配置
                    field_configs = {
                        "交易ID": st.column_config.NumberColumn("交易ID", format="%.0f", width="small"),
                        "股票代码": st.column_config.TextColumn("股票代码", width="small"),
                        "数量": st.column_config.NumberColumn("数量", format="%.2f", width="small"),
                        "入场价格": st.column_config.NumberColumn("入场价格", format="%.2f", width="medium"),
                        "入场手续费": st.column_config.NumberColumn("入场手续费", format="%.2f", width="small"),
                        "出场时间": st.column_config.TextColumn("出场时间", width="medium"),
                        "出场价格": st.column_config.NumberColumn("出场价格", format="%.2f", width="medium"),
                        "出场手续费": st.column_config.NumberColumn("出场手续费", format="%.2f", width="small"),
                        "收益": st.column_config.NumberColumn("收益", format="%.2f", width="medium"),
                        "收益率": st.column_config.NumberColumn("收益率", format="%.4f", width="small"),
                        "父ID": st.column_config.NumberColumn("父ID", format="%.0f", width="small"),
                        "价值": st.column_config.NumberColumn("价值", format="%.2f", width="medium"),
                    }

                    # 过滤掉隐藏的列
                    display_columns = [col for col in trades_df.columns if col not in hidden_columns]

                    # 为显示的列创建配置
                    column_config = {}
                    for col in display_columns:
                        column_config[col] = field_configs.get(col, st.column_config.TextColumn(col, width="medium"))

                    st.dataframe(
                        trades_df[display_columns],
                        column_config=column_config if column_config else None,
                        use_container_width=True,
                        height=400,
                    )

                    # 提供下载选项
                    csv = trades_df[display_columns].to_csv()
                    st.download_button(
                        label="📥 下载交易日志 (CSV)",
                        data=csv,
                        file_name=f"trades_{config['start_date']}_to_{config['end_date']}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("回测期间没有产生交易记录")
            else:
                st.info("""
                **交易日志功能说明**：

                当前回测结果未包含详细的交易记录。这可能是因为：
                1. 使用的是 VectorBT 的 stats() 方法，该方法主要返回性能指标
                2. 需要在回测时显式提取交易记录

                **建议**：如需查看详细交易记录，可以：
                - 查看"净值分析"标签了解整体表现
                - 查看"风险分析"标签了解风险指标
                - 联系开发者添加交易记录提取功能
                """)


# ============ 主应用 ============
def main():
    """主应用入口"""

    # 侧边栏导航
    with st.sidebar:
        st.title("FactorFlow")
        st.caption("股票因子分析系统")

        page = st.radio(
            "选择功能",
            ["因子管理", "因子分析", "策略回测"],
            label_visibility="collapsed",
        )

        st.divider()

        # 因子管理页面显示统计
        if page == "因子管理":
            stats = factor_service.get_factor_stats()
            st.metric("预置因子", stats["preset_count"])
            st.metric("用户因子", stats["user_count"])
            st.metric("总因子数", stats["total_count"])

    # 因子管理页面
    if page == "因子管理":
        st.title("因子管理")

        tab1, tab2 = st.tabs(["因子列表", "新增因子"])

        # Tab1: 因子列表
        with tab1:
            # 顶部工具栏
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown("### 因子列表")
            with col2:
                factor_count = len(factor_service.get_all_factors())
                st.metric("因子总数", factor_count)
            with col3:
                if st.button("🔄 刷新", use_container_width=True, key="refresh_factor_list"):
                    st.rerun()

            st.markdown("---")

            factors = factor_service.get_all_factors()

            if not factors:
                st.info("暂无因子，请先在系统初始化时加载预置因子或添加自定义因子")
            else:
                # 按分类分组显示
                factor_df = pd.DataFrame(factors)
                if not factor_df.empty:
                    factor_df["来源"] = factor_df["source"].map({"preset": "预置", "user": "自定义"})

                    # 分类筛选
                    categories = ["全部"] + factor_df["category"].unique().tolist()
                    selected_category = st.selectbox("筛选分类", categories)

                    if selected_category != "全部":
                        display_df = factor_df[factor_df["category"] == selected_category]
                    else:
                        display_df = factor_df

                    # 来源筛选
                    source_filter = st.multiselect(
                        "筛选来源", ["预置", "自定义"], default=["预置", "自定义"]
                    )
                    if source_filter:
                        display_df = display_df[display_df["来源"].isin(source_filter)]

                    # 显示选项
                    show_formula = st.checkbox("在表格中显示公式", value=False, key="show_formula_in_table")

                    # 因子列表数据表格
                    if show_formula:
                        columns_to_show = ["name", "category", "来源", "code", "description"]
                        column_config = {
                            "name": st.column_config.TextColumn("因子名称", width="medium"),
                            "category": st.column_config.TextColumn("分类", width="small"),
                            "来源": st.column_config.TextColumn("来源", width="small"),
                            "code": st.column_config.TextColumn("公式", width="large"),
                            "description": st.column_config.TextColumn("说明", width="large"),
                        }
                    else:
                        columns_to_show = ["name", "category", "来源", "description"]
                        column_config = {
                            "name": st.column_config.TextColumn("因子名称", width="medium"),
                            "category": st.column_config.TextColumn("分类", width="small"),
                            "来源": st.column_config.TextColumn("来源", width="small"),
                            "description": st.column_config.TextColumn("说明", width="large"),
                        }

                    st.dataframe(
                        display_df[columns_to_show],
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                    )

                    # 因子公式查看器
                    with st.expander("📝 查看因子公式", expanded=False):
                        selected_factor_name = st.selectbox(
                            "选择因子查看公式",
                            options=display_df["name"].tolist(),
                            key="view_formula_select",
                        )

                        if selected_factor_name:
                            factor_info = display_df[display_df["name"] == selected_factor_name].iloc[0]
                            st.markdown(f"**因子名称**: {factor_info['name']}")
                            st.markdown(f"**分类**: {factor_info['category']}")
                            st.markdown(f"**说明**: {factor_info['description']}")

                            # 显示公式代码
                            if "code" in factor_info and pd.notna(factor_info["code"]):
                                st.markdown("**计算公式**:")
                                st.code(
                                    factor_info["code"],
                                    language="python",
                                    line_numbers=False,
                                )
                            else:
                                st.warning("该因子没有公式代码")

                    # 删除因子（仅限用户自定义）
                    with st.expander("删除因子", expanded=False):
                        user_factors = factor_df[factor_df["source"] == "user"]
                        if not user_factors.empty:
                            factor_to_delete = st.selectbox(
                                "选择要删除的因子",
                                options=user_factors["name"].tolist(),
                                key="delete_factor_select",
                            )
                            delete_col1, delete_col2 = st.columns(2)
                            with delete_col1:
                                if st.button("确认删除", type="primary", key="btn_delete_factor"):
                                    try:
                                        # 查找因子ID
                                        factor_id = user_factors[user_factors["name"] == factor_to_delete]["id"].iloc[0]
                                        factor_service.delete_factor(factor_id)
                                        # 设置成功消息到session_state
                                        st.session_state["delete_success"] = f"因子 '{factor_to_delete}' 已删除"
                                        st.rerun()
                                    except Exception as e:
                                        st.session_state["delete_error"] = str(e)
                                        st.rerun()

                            # 显示操作结果
                            if "delete_success" in st.session_state:
                                st.success(st.session_state["delete_success"])
                                # 清除消息，避免重复显示
                                del st.session_state["delete_success"]
                            elif "delete_error" in st.session_state:
                                st.error(f"删除失败: {st.session_state['delete_error']}")
                                del st.session_state["delete_error"]
                        else:
                            st.info("没有可删除的用户自定义因子")

        # Tab2: 新增因子
        with tab2:
            st.subheader("创建自定义因子")

            # 代码模式选择
            code_mode = st.radio(
                "代码模式",
                ["简单表达式", "自定义函数"],
                horizontal=True,
                help="简单表达式：单行公式；自定义函数：支持复杂逻辑的Python函数"
            )

            # 自定义函数模式下的模板选择（在form外部）
            if code_mode == "自定义函数":
                st.markdown("**快速加载模板**")
                template_col1, template_col2, template_col3 = st.columns(3)

                # 定义模板代码
                templates = {
                    "trend": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """多因子组合趋势指标"""
    # 计算RSI
    rsi = RSI(df["close"], timeperiod=14)

    # 计算MACD
    macd, signal, hist = MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

    # 计算ADX
    adx = ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # 组合信号：RSI > 50 且 MACD > signal 且 ADX > 25
    trend_score = (
        (rsi > 50).astype(int) +
        (macd > signal).astype(int) +
        (adx > 25).astype(int)
    )

    return trend_score''',
                    "volatility": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """动态波动率因子"""
    # 计算对数收益率
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # 计算短期波动率（10日）
    vol_short = log_ret.rolling(window=10).std()

    # 计算长期波动率（60日）
    vol_long = log_ret.rolling(window=60).std()

    # 波动率相对水平
    vol_ratio = vol_short / vol_long

    # 标记高波动期
    high_vol_threshold = vol_ratio.rolling(window=252).quantile(0.7)
    is_high_vol = (vol_ratio > high_vol_threshold).astype(float)

    return is_high_vol''',
                    "pattern": '''def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """K线形态识别因子"""
    # 计算实体大小
    body = abs(df["close"] - df["open"])

    # 计算上下影线
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    # 识别十字星（实体很小，上下影线较长）
    avg_body = body.rolling(window=20).mean()
    is_doji = (
        (body < 0.3 * avg_body) &
        (upper_shadow > 0.5 * body) &
        (lower_shadow > 0.5 * body)
    ).astype(float)

    # 识别大阳线
    avg_range = (df["high"] - df["low"]).rolling(window=20).mean()
    is_bullish = (
        (df["close"] > df["open"]) &
        (body > 1.5 * avg_body) &
        ((df["close"] - df["low"]) / (df["high"] - df["low"]) > 0.6)
    ).astype(float)

    # 组合信号
    return is_doji * 0.5 + is_bullish * 1.0'''
                }

                with template_col1:
                    if st.button("📈 加载趋势模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["trend"]
                        st.rerun()

                with template_col2:
                    if st.button("📊 加载波动率模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["volatility"]
                        st.rerun()

                with template_col3:
                    if st.button("🕯️ 加载K线形态模板", use_container_width=True):
                        st.session_state["factor_code_template"] = templates["pattern"]
                        st.rerun()

                st.markdown("---")

            with st.form("create_factor"):
                factor_name = st.text_input(
                    "因子名称",
                    placeholder="例如：my_custom_factor",
                    help="请使用英文名称，仅包含字母、数字和下划线",
                )

                if code_mode == "简单表达式":
                    st.markdown("**因子公式代码**")
                    st.markdown("""
                    可用变量：
                    - `open`, `high`, `low`, `close`, `volume`: OHLCV数据
                    - `np`: numpy模块
                    - `SMA`, `EMA`, `RSI`, `MACD`, `ADX`, `CCI`, `ATR`, `BBANDS`, `OBV`: TA-Lib函数

                    示例：
                    ```python
                    # 简单移动平均比率
                    close / SMA(close, timeperiod=20)

                    # 价格动量
                    (close - close.shift(10)) / close.shift(10)

                    # 波动率
                    np.log(close / close.shift(1)).rolling(window=20).std()
                    ```
                    """)

                    factor_code = st.text_area(
                        "因子计算公式",
                        placeholder="close / SMA(close, timeperiod=20)",
                        height=100,
                    )

                else:  # 自定义函数模式
                    st.markdown("**因子计算函数**")
                    st.markdown("""
                    可用变量：
                    - `df`: 包含OHLCV数据的DataFrame，包含列 `open`, `high`, `low`, `close`, `volume`
                    - `np`: numpy模块
                    - `pd`: pandas模块
                    - `SMA`, `EMA`, `RSI`, `MACD`, `ADX`, `CCI`, `ATR`, `BBANDS`, `OBV`: TA-Lib函数

                    **函数签名**：
                    ```python
                    def calculate_factor(df: pd.DataFrame) -> pd.Series:
                        \"\"\"
                        df: 包含OHLCV数据的DataFrame
                        返回: 与df长度相同的因子值Series
                        \"\"\"
                        # 你的计算逻辑
                        return factor_values
                    ```
                    """)

                    # 显示代码编辑器
                    # 使用 session_state 直接控制值
                    if "factor_code_input" not in st.session_state:
                        # 初始默认代码
                        st.session_state.factor_code_input = """def calculate_factor(df: pd.DataFrame) -> pd.Series:
    \"\"\"
    自定义因子计算函数

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        因子值Series，长度与df相同
    \"\"\"
    # 示例：计算价格相对于20日均线的偏离度
    sma20 = SMA(df["close"], timeperiod=20)
    deviation = (df["close"] - sma20) / sma20

    return deviation"""

                    # 如果有模板加载，更新值
                    if "factor_code_template" in st.session_state:
                        st.session_state.factor_code_input = st.session_state.factor_code_template
                        # 清除模板标记，避免重复加载
                        del st.session_state["factor_code_template"]

                    factor_code = st.text_area(
                        "因子计算函数代码",
                        height=300,
                        help="完整的Python函数，必须返回pd.Series对象",
                        key="factor_code_input"
                    )

                factor_description = st.text_area(
                    "因子说明",
                    placeholder="简要描述因子的含义和用途...",
                    height=80,
                )

                col1, col2 = st.columns(2)
                with col1:
                    validate_btn = st.form_submit_button("验证代码", type="secondary")
                with col2:
                    create_btn = st.form_submit_button("创建因子", type="primary")

                # 验证按钮
                if validate_btn:
                    if not factor_name or not factor_code:
                        st.warning("请填写因子名称和计算公式")
                    else:
                        is_valid, message = factor_service.validate_factor_code(factor_code)
                        if is_valid:
                            st.success(message)
                        else:
                            st.error(message)

                # 创建按钮
                if create_btn:
                    if not factor_name or not factor_code:
                        st.warning("请填写因子名称和计算公式")
                    else:
                        try:
                            factor_service.create_factor(
                                name=factor_name,
                                code=factor_code,
                                description=factor_description,
                            )
                            # 设置成功消息到session_state
                            st.session_state["create_success"] = f"因子 '{factor_name}' 创建成功！切换到「因子列表」标签页查看。"
                            st.rerun()
                        except ValueError as e:
                            st.session_state["create_error"] = str(e)
                            st.rerun()

            # 显示创建操作的反馈消息（在form外部）
            if "create_success" in st.session_state:
                st.success(st.session_state["create_success"])
                # 清除消息
                del st.session_state["create_success"]
            elif "create_error" in st.session_state:
                st.error(st.session_state["create_error"])
                del st.session_state["create_error"]

    # 因子分析页面
    elif page == "因子分析":
        st.title("因子分析")

        # 侧边栏配置
        with st.sidebar:
            st.subheader("分析配置")

            # 数据模式
            data_mode = st.radio("数据模式", ["单股票", "股票池"])

            # 股票代码输入
            if data_mode == "单股票":
                stock_codes_input = st.text_input(
                    "股票代码",
                    value="000001",
                    help="支持格式：000001 或 000001.SZ（深圳）, 600000（上海）",
                )
                stock_codes = [stock_codes_input.strip()]
            else:
                stock_codes_input = st.text_area(
                    "股票代码（每行一个）",
                    value="000001\n600000",
                    height=100,
                )
                stock_codes = [code.strip() for code in stock_codes_input.strip().split("\n") if code.strip()]

            # 选择因子
            all_factors = factor_service.get_all_factors()
            factor_options = {f["name"]: f["description"] for f in all_factors}
            selected_factors = st.multiselect(
                "选择因子",
                options=list(factor_options.keys()),
                format_func=lambda x: f"{x} - {factor_options.get(x, '')}",
                default=[],
            )

            # 时间范围
            default_end = datetime.now()
            default_start = default_end - timedelta(days=365)

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", value=default_start)
            with col2:
                end_date = st.date_input("结束日期", value=default_end)

            # 分析按钮
            analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

        # 执行分析
        if analyze_btn:
            if not stock_codes or not stock_codes[0]:
                st.error("请输入股票代码")
            elif not selected_factors:
                st.error("请至少选择一个因子")
            else:
                with st.spinner("正在执行因子分析，请稍候..."):
                    try:
                        results = analysis_service.analyze(
                            stock_codes=stock_codes,
                            factor_names=selected_factors,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            use_cache=True,
                        )

                        # 保存到session state供其他页面使用
                        st.session_state["analysis_results"] = results
                        st.session_state["analysis_stock_codes"] = stock_codes
                        st.session_state["analysis_factors"] = selected_factors

                        st.success("分析完成！")

                    except Exception as e:
                        st.error(f"分析失败: {e}")

        # 显示分析结果
        if "analysis_results" in st.session_state:
            results = st.session_state["analysis_results"]
            factor_data = results.get("factor_data", {})
            stock_codes_result = results["metadata"]["stock_codes"]
            selected_factors_result = results["metadata"]["factor_names"]

            # Tab1: 行情预览
            tab1, tab2, tab3, tab4 = st.tabs(["行情预览", "SHAP分析", "统计分析", "导出报告"])

            with tab1:
                st.subheader("股票行情")

                # 价格走势图
                plot_price_chart(factor_data, stock_codes_result)

                # 因子叠加图（每个股票分tab显示）
                if len(stock_codes_result) > 1:
                    stock_tabs = st.tabs(stock_codes_result)
                    for stock_code, stock_tab in zip(stock_codes_result, stock_tabs):
                        with stock_tab:
                            if stock_code in factor_data:
                                st.write(f"**{stock_code}**")

                                # 因子选择
                                selected_plot_factor = st.selectbox(
                                    "选择要显示的因子",
                                    options=selected_factors_result,
                                    key=f"plot_factor_{stock_code}",
                                )
                                if selected_plot_factor:
                                    plot_factor_with_price(
                                        factor_data[stock_code],
                                        selected_plot_factor,
                                        stock_code,
                                    )

                                # 因子数据表格
                                with st.expander("查看因子数据"):
                                    display_cols = ["close", "open", "high", "low"] + selected_factors_result
                                    available_cols = [col for col in display_cols if col in factor_data[stock_code].columns]
                                    st.dataframe(
                                        factor_data[stock_code][available_cols].tail(100),
                                        use_container_width=True,
                                    )
                else:
                    stock_code = stock_codes_result[0]
                    if stock_code in factor_data:
                        selected_plot_factor = st.selectbox(
                            "选择要显示的因子",
                            options=selected_factors_result,
                            key="plot_factor_single",
                        )
                        if selected_plot_factor:
                            plot_factor_with_price(
                                factor_data[stock_code],
                                selected_plot_factor,
                                stock_code,
                            )

                        with st.expander("查看因子数据"):
                            display_cols = ["close", "open", "high", "low"] + selected_factors_result
                            available_cols = [col for col in display_cols if col in factor_data[stock_code].columns]
                            st.dataframe(
                                factor_data[stock_code][available_cols].tail(100),
                                use_container_width=True,
                            )

            with tab2:
                st.subheader("SHAP分析")

                # SHAP分析说明
                with st.expander("📖 SHAP分析方法说明", expanded=False):
                    st.markdown("""
                    ### SHAP (SHapley Additive exPlanations) 分析逻辑

                    **什么是SHAP？**
                    SHAP 是一种博弈论方法，用于解释机器学习模型的输出。它将预测结果分解为每个特征的贡献度。

                    ---

                    ### 📊 不同场景下的SHAP分析

                    #### 1️⃣ 单股票 × 单因子场景

                    **计算逻辑**：
                    ```
                    时间序列: t=1, 2, ..., T
                    因子值: f₁, f₂, ..., fₜ (已计算)
                    未来收益率: rₜ = close(t+5)/close(t) - 1

                    训练数据:
                    X = [f₁, f₂, ..., fₜ]  # 特征矩阵
                    y = [r₁, r₂, ..., rₜ]  # 目标变量

                    模型: XGBoost(X, y)
                    SHAP值: 贡献度(fᵢ → rₜ)
                    ```

                    **解释**：
                    - SHAP值表示该因子在每个时点对未来5日收益的贡献
                    - 适合分析单个因子对收益的预测能力
                    - 时序依赖性强，适合个股择时策略

                    **适用场景**：
                    - 评估单个因子的有效性
                    - 选择最有效的预测因子
                    - 构建单因子择时策略

                    #### 2️⃣ 单股票 × 多因子场景

                    **计算逻辑**：
                    ```
                    多个因子: f₁, f₂, ..., fₙ (例如：RSI, MACD, ADX)

                    训练数据:
                    X = [f₁, f₂, ..., fₙ]  # 多个因子作为特征
                    y = 未来5日收益率

                    模型: XGBoost(X, y)
                    SHAP值: 每个因子对预测的贡献度
                    ```

                    **解释**：
                    - 比较多个因子的相对重要性
                    - 识别冗余因子（SHAP值接近0）
                    - 发现因子之间的交互作用

                    **适用场景**：
                    - 构建多因子选股模型
                    - 因子组合优化
                    - 识别最佳因子组合

                    #### 3️⃣ 股票池 × 单因子场景

                    **计算逻辑**：
                    ```
                    横截面: t时刻，股票池 {s₁, s₂, ..., sₙ}
                    因子值: f(s₁), f(s₂), ..., f(sₙ)
                    收益率: r(s₁), r(s₂), ..., r(sₙ)

                    训练数据（合并所有股票和时刻）:
                    X = [f(sᵢ, tₖ)]  # (股票×时间) × 1
                    y = r(sᵢ, tₖ)     # (股票×时间) × 1

                    模型: XGBoost(X, y)
                    SHAP值: 该因子在所有股票和时刻的平均贡献度
                    ```

                    **解释**：
                    - 评估因子在横截面上的选股能力
                    - SHAP值高说明该因子能有效区分股票表现
                    - 与时序IC互补，综合评估因子有效性

                    **适用场景**：
                    - 因子的横截面有效性验证
                    - 选股因子筛选
                    - 股票组合构建

                    #### 4️⃣ 股票池 × 多因子场景（最复杂）

                    **计算逻辑**：
                    ```
                    多个因子: f₁, f₂, ..., fₙ
                    股票池: s₁, s₂, ..., sₙ
                    时间: t₁, t₂, ..., tₜ

                    训练数据:
                    X = [f₁(sᵢ, tₖ), f₂(sᵢ, tₖ), ..., fₙ(sᵢ, tₖ)]  # (股票×时间) × 因子数
                    y = r(sᵢ, tₖ)                                       # (股票×时间) × 1

                    模型: XGBoost(X, y)
                    SHAP值: 每个因子在所有股票和时刻的平均贡献度
                    ```

                    **解释**：
                    - 综合评估多因子组合的预测能力
                    - 识别最重要的因子
                    - 发现因子在不同股票和时间的稳定性

                    **适用场景**：
                    - 多因子选股模型
                    - 因子权重配置
                    - 稳健的量化策略构建

                    ---

                    ### 📈 输出指标说明

                    **1. 全局特征重要性**
                    - `importance = mean(|SHAP值|)`
                    - 值越大，因子越重要
                    - 可以识别出最重要的因子

                    **2. 模型R²得分**
                    - R² = 1 - RSS/TSS
                    - 衡量模型对收益率变动的解释比例
                    - R² > 0.05 说明因子有效
                    - R² > 0.1 说明因子很强

                    **3. 使用建议**
                    - **因子筛选**：选择SHAP重要性>阈值的因子
                    - **因子组合**：使用SHAP正相关的因子
                    - **风险评估**：注意SHAP不稳定的因子
                    - **策略构建**：结合IC/IR综合评估
                    """)

                shap_data = results.get("shap", {})

                if "error" in shap_data:
                    st.warning(shap_data["error"])
                else:
                    # 特征重要性
                    if "feature_importance" in shap_data:
                        st.markdown("### 全局特征重要性")
                        plot_shap_importance(shap_data["feature_importance"])

                    # 模型得分
                    if "model_score" in shap_data:
                        st.metric("XGBoost模型R²得分", f"{shap_data['model_score']:.4f}")

            with tab3:
                st.subheader("统计分析")

                # 统计分析说明
                with st.expander("📖 IC/IR统计分析说明", expanded=False):
                    st.markdown("""
                    ### IC (Information Coefficient) 信息系数分析

                    **什么是IC？**
                    IC 是衡量因子值与未来收益率相关性的指标，反映因子的预测能力。

                    ---

                    ### 📊 不同场景下的IC/IR计算

                    #### 1️⃣ 单股票时序IC (Time-Series IC)

                    **计算公式**：
                    ```
                    股票 i，时间 t
                    因子值: fₜ
                    未来收益率: rₜ = close(t+5)/close(t) - 1

                    ICₜ = Corr(fₜ, rₜ)
                          = Cov(fₜ, rₜ) / (σ(fₜ) × σ(rₜ))
                    ```

                    **计算逻辑**：
                    - 使用滚动窗口（默认252天）计算时序IC
                    - 每个时点计算因子值与未来收益率的斯皮尔曼相关系数
                    - 得到IC序列：IC₁, IC₂, ..., ICₜ

                    **统计指标**：
                    - **IC均值**：`mean(IC)` - 衡量因子平均预测能力
                    - **IC标准差**：`std(IC)` - 衡量IC的波动程度
                    - **IR (信息比率)**：`mean(IC) / std(IC)` - 风险调整后的收益能力
                    - **IC>0占比**：`count(IC>0) / count(IC)` - 因子方向性

                    **解释**：
                    - IC均值绝对值 > 0.03 → 因子有效
                    - IC均值绝对值 > 0.05 → 因子很强
                    - IR > 0.5 → 优秀因子
                    - IR > 0.3 → 良好因子

                    **适用场景**：
                    - 评估因子的择时能力
                    - 单因子策略回测
                    - 时序策略因子选择

                    #### 2️⃣ 股票池横截面IC (Cross-Sectional IC)

                    **计算公式**：
                    ```
                    时间 t，股票池 {s₁, s₂, ..., sₙ}
                    因子值: f(s₁), f(s₂), ..., f(sₙ)
                    收益率: r(s₁), r(s₂), ..., r(sₙ)

                    ICₜ = Corr(f(s), r(s))
                          = Cov(f(s), r(s)) / (σ(f(s)) × σ(r(s)))
                    ```

                    **计算逻辑**：
                    - 在每个时点t，计算所有股票因子值与收益率的相关系数
                    - 得到IC序列：IC₁, IC₂, ..., ICₜ

                    **统计指标**：
                    - 计算方式与时序IC相同
                    - 但意义不同：衡量横截面选股能力

                    **解释**：
                    - IC高 → 因子能有效区分股票表现
                    - 适合选股策略，而非择时
                    - 横截面IC与时序IC互补

                    **适用场景**：
                    - 多因子选股模型
                    - 股票组合构建
                    - 因子横截面有效性验证

                    ---

                    ### 🔍 单因子 vs 多因子场景

                    #### 单因子场景
                    ```
                    计算某个因子 f₁ 的 IC/IR

                    时序IC（单股票）：
                      IC = Corr(f₁, future_return)
                      分析：因子f₁的预测能力

                    横截面IC（股票池）：
                      IC = Corr(f₁(s), future_return(s))
                      分析：因子f₁的选股能力
                    ```

                    #### 多因子场景
                    ```
                    计算多个因子 {f₁, f₂, ..., fₙ} 的IC/IR

                    方法1：分别计算每个因子的IC
                      IC(f₁), IC(f₂), ..., IC(fₙ)
                      比较哪个因子预测能力最强

                    方法2：计算因子组合的IC
                      先组合：F = w₁·f₁ + w₂·f₂ + ... + wₙ·fₙ
                      再计算：IC = Corr(F, future_return)
                      分析：因子组合是否比单因子更好
                    ```

                    ---

                    ### 📈 统计图表说明

                    **1. IC统计表格**
                    - 显示每个因子的IC均值、标准差、IR
                    - 用于快速比较因子性能

                    **2. 月度IC热力图**
                    - X轴：月份
                    - Y轴：因子
                    - 颜色：IC值（红色负值，绿色正值）
                    - 用于观察因子在不同时期的表现

                    **3. 滚动IR曲线**
                    - X轴：时间
                    - Y轴：滚动IR值
                    - 用于识别因子稳定性变化

                    **4. 因子相关性矩阵**
                    - 显示因子之间的相关系数
                    - 用于识别冗余因子
                    - 相关性高(>0.7)的因子可考虑去重

                    ---

                    ### 💡 使用建议

                    1. **因子筛选**
                       - 优先选择IC均值高且IR>0.5的因子
                       - IC<0.02的因子建议剔除

                    2. **因子组合**
                       - 选择相关性低、IC都高的因子
                       - 避免使用高度相关的因子

                    3. **稳定性分析**
                       - 观察滚动IR，识别因子失效期
                       - 月度IC热力图观察周期性

                    4. **策略构建**
                       - 时序IC高的因子适合择时策略
                       - 横截面IC高的因子适合选股策略
                       - 结合SHAP分析综合评估
                    """)

                ic_ir_data = results.get("ic_ir", {})

                # 相关性矩阵
                st.markdown("### 因子相关性矩阵")
                plot_correlation_matrix(factor_data, selected_factors_result)

                if not ic_ir_data:
                    st.info("暂无IC/IR统计数据")
                else:
                    # IC统计表格
                    if "ic_stats" in ic_ir_data:
                        st.markdown("### IC统计表格")
                        ic_stats = ic_ir_data["ic_stats"]

                        # 准备表格数据
                        table_data = []
                        for factor_name, stats in ic_stats.items():
                            table_data.append({
                                "因子名称": factor_name,
                                "IC均值": stats['IC均值'],
                                "IC标准差": stats['IC标准差'],
                                "IR": stats['IR'],
                                "IC>0占比": stats['IC>0占比'],
                                "IC绝对值均值": stats['IC绝对值均值'],
                            })

                        st.dataframe(
                            pd.DataFrame(table_data),
                            column_config={
                                "因子名称": st.column_config.TextColumn("因子名称", width="medium"),
                                "IC均值": st.column_config.NumberColumn(
                                    "IC均值",
                                    format="%.4f",
                                    help="📖 **IC均值**：衡量因子预测能力的平均水平\n\n- 绝对值越大，因子预测能力越强\n- >0.03 为有效因子，>0.05 为优秀因子\n- 负值表示反向预测"
                                ),
                                "IC标准差": st.column_config.NumberColumn(
                                    "IC标准差",
                                    format="%.4f",
                                    help="📖 **IC标准差**：IC值的波动程度\n\n- 标准差越小，因子表现越稳定\n- 用于计算IR（信息比率）\n- 过大的波动表明因子不稳定"
                                ),
                                "IR": st.column_config.NumberColumn(
                                    "IR (信息比率)",
                                    format="%.4f",
                                    help="📖 **IR (Information Ratio)**：IC均值与标准差的比值\n\n- 衡量因子的风险调整后收益能力\n- >0.5 为优秀因子，>0.3 为良好因子\n- 公式：IR = IC均值 / IC标准差"
                                ),
                                "IC>0占比": st.column_config.NumberColumn(
                                    "IC>0占比",
                                    format="%.2%",
                                    help="📖 **IC>0占比**：因子预测方向正确的比例\n\n- 衡量因子方向性预测的准确度\n- >50% 表示方向预测准确\n- 接近100%表示几乎总是方向正确"
                                ),
                                "IC绝对值均值": st.column_config.NumberColumn(
                                    "IC绝对值均值",
                                    format="%.4f",
                                    help="📖 **IC绝对值均值**：|IC|的平均值\n\n- 忽略正负方向，只看预测强度\n- 用于评估因子的整体预测能力\n- 比IC均值更能反映因子的真实预测力"
                                ),
                            },
                            use_container_width=True,
                            hide_index=True,
                        )

                    # IC时间序列
                    if "ic_stats" in ic_ir_data:
                        st.markdown("### IC时间序列")
                        ic_series_dict = {
                            name: stats["IC序列"]
                            for name, stats in ic_ir_data["ic_stats"].items()
                        }
                        plot_ic_time_series(ic_series_dict)

                    # 月度IC热力图
                    if "monthly_ic" in ic_ir_data:
                        st.markdown("### 月度IC热力图")
                        monthly_ic = ic_ir_data["monthly_ic"]
                        if not monthly_ic:
                            st.info("暂无月度IC数据")
                        elif len(monthly_ic) == 1:
                            plot_monthly_ic_heatmap(list(monthly_ic.values())[0], list(monthly_ic.keys())[0])
                        else:
                            monthly_tabs = st.tabs(list(monthly_ic.keys()))
                            for factor_name, monthly_tab in zip(monthly_ic.keys(), monthly_tabs):
                                with monthly_tab:
                                    plot_monthly_ic_heatmap(monthly_ic[factor_name], factor_name)

                    # 滚动窗口IR
                    if "rolling_ir" in ic_ir_data:
                        st.markdown("### 滚动窗口IR")
                        plot_rolling_ir(ic_ir_data["rolling_ir"])

            with tab4:
                st.subheader("导出报告")

                report = analysis_service.generate_report(results)

                st.markdown(report)

                if st.button("保存报告到文件", type="primary"):
                    output_path = analysis_service.export_report(results)
                    st.success(f"报告已保存到: {output_path}")

    # 策略回测页面
    elif page == "策略回测":
        backtest_main()


if __name__ == "__main__":
    main()
