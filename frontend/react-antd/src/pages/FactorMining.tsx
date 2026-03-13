import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Card,
  Form,
  Input,
  DatePicker,
  Button,
  Select,
  InputNumber,
  Progress,
  Row,
  Col,
  message,
  Space,
  Divider,
  Tag,
  Spin,
  Alert
} from 'antd'
import {
  PlayCircleOutlined,
  SaveOutlined,
  BarChartOutlined,
  RocketOutlined
} from '@ant-design/icons'
import * as echarts from 'echarts'
import { api } from '@/services/api'
import dayjs from 'dayjs'
import './FactorMining.css'

const { Option } = Select
const { RangePicker } = DatePicker

interface Factor {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
}

interface MinedFactor {
  name: string
  expression: string
  ic: number
  ir: number
  fitness: number
}

interface MiningStatus {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_generation: number
  total_generations: number
  best_fitness: number
  avg_fitness: number
  fitness_history?: {
    best: number[]
    average: number[]
  }
  error?: string
}

interface MiningResult {
  factors: MinedFactor[]
  best_fitness: number
  avg_fitness: number
  generations: number
  fitness_history?: {
    best: number[]
    average: number[]
  }
}

const FactorMining: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const evolutionChartRef = useRef<HTMLDivElement>(null)
  const resultChartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<echarts.ECharts | null>(null)
  const resultChartInstanceRef = useRef<echarts.ECharts | null>(null)

  const [factors, setFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(false)
  const [mining, setMining] = useState(false)
  const [currentStockCode, setCurrentStockCode] = useState<string>('')

  const [miningStatus, setMiningStatus] = useState<MiningStatus | null>(null)
  const [miningResult, setMiningResult] = useState<MiningResult | null>(null)

  // 加载因子列表
  const loadFactors = async () => {
    try {
      const response = await api.getFactors() as any
      if (response.success) {
        setFactors(response.data)
      }
    } catch (error) {
      console.error('加载因子列表失败:', error)
    }
  }

  useEffect(() => {
    loadFactors()

    // 设置默认日期范围
    const endDate = dayjs()
    const startDate = dayjs().subtract(1, 'year')
    form.setFieldsValue({
      dateRange: [startDate, endDate],
      population_size: 50,
      n_generations: 10,
      mutation_rate: 0.2,
      crossover_rate: 0.7,
      elite_size: 5,
      fitness_objective: 'ic_mean',
      ic_threshold: 0.03
    })

    return () => {
      // 清理定时器
      if (window.miningInterval) {
        clearInterval(window.miningInterval)
      }
      // 清理图表
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose()
        chartInstanceRef.current = null
      }
      if (resultChartInstanceRef.current) {
        resultChartInstanceRef.current.dispose()
        resultChartInstanceRef.current = null
      }
    }
  }, [])

  // 开始挖掘
  const startMining = async (values: any) => {
    const selectedFactors = values.base_factors || []

    // 保存当前股票代码，用于后续命名
    const stockCode = values.stock_code.replace('.', '') // 移除股票代码中的点
    setCurrentStockCode(stockCode)

    const [startDate, endDate] = values.dateRange
    const requestData = {
      stock_code: values.stock_code,
      base_factors: selectedFactors,
      start_date: startDate.format('YYYY-MM-DD'),
      end_date: endDate.format('YYYY-MM-DD'),
      population_size: values.population_size,
      n_generations: values.n_generations,
      cx_prob: values.crossover_rate,
      mut_prob: values.mutation_rate,
      elite_size: values.elite_size,
      fitness_objective: values.fitness_objective,
      ic_threshold: values.ic_threshold
    }

    try {
      setLoading(true)
      setMining(true)
      setMiningResult(null)

      const response = await api.startGeneticMining(requestData) as any

      if (response.success) {
        const newTaskId = response.data.task_id

        // 轮询获取进度
        window.miningInterval = setInterval(() => {
          checkMiningProgress(newTaskId)
        }, 2000)

        message.success('挖掘任务已启动')
      }
    } catch (error) {
      console.error('启动挖掘失败:', error)
      message.error('启动挖掘失败')
      setMining(false)
    } finally {
      setLoading(false)
    }
  }

  // 检查挖掘进度
  const checkMiningProgress = async (currentTaskId: string) => {
    try {
      const response = await api.getMiningStatus(currentTaskId) as any

      if (response.success) {
        const statusData = response.data as MiningStatus
        setMiningStatus(statusData)

        console.log('Mining status:', statusData.status, 'Generation:', statusData.current_generation, '/', statusData.total_generations)

        // 更新进化曲线 - 使用完整的历史数据
        if (statusData.fitness_history && statusData.fitness_history.best.length > 0) {
          console.log('Updating evolution chart with history:', statusData.fitness_history)
          updateEvolutionChart(statusData.fitness_history)
        } else if (statusData.current_generation > 0) {
          // 降级方案：如果没有历史数据，用当前值生成
          console.log('Using fallback for evolution chart')
          updateEvolutionChart({
            best: Array(statusData.current_generation).fill(statusData.best_fitness),
            average: Array(statusData.current_generation).fill(statusData.avg_fitness)
          })
        }

        // 检查是否完成
        if (statusData.status === 'completed') {
          console.log('Mining completed, clearing interval and getting results')
          if (window.miningInterval) {
            clearInterval(window.miningInterval)
          }
          setMining(false)
          await getMiningResults(currentTaskId)
        } else if (statusData.status === 'failed') {
          console.log('Mining failed:', statusData.error)
          if (window.miningInterval) {
            clearInterval(window.miningInterval)
          }
          setMining(false)
          message.error(`挖掘失败: ${statusData.error || '未知错误'}`)
        }
      }
    } catch (error) {
      console.error('获取进度失败:', error)
    }
  }

  // 获取挖掘结果
  const getMiningResults = async (currentTaskId: string) => {
    try {
      console.log('Fetching mining results for task:', currentTaskId)
      const response = await api.getMiningResults(currentTaskId) as any

      if (response.success) {
        console.log('Mining results received:', response.data)
        setMiningResult(response.data)

        // 绘制最终进化曲线（使用新的图表实例）
        if (response.data.fitness_history) {
          console.log('Updating result chart with history:', response.data.fitness_history)
          // 延迟一下确保DOM已渲染
          setTimeout(() => {
            updateResultChart(response.data.fitness_history)
          }, 200)
        }
      } else {
        message.error('获取结果失败: ' + (response.message || '未知错误'))
      }
    } catch (error) {
      console.error('获取结果失败:', error)
      message.error('获取结果失败')
    }
  }

  // 更新进化曲线（进度中）
  const updateEvolutionChart = (fitnessHistory: { best: number[]; average: number[] }) => {
    console.log('updateEvolutionChart called, DOM element:', evolutionChartRef.current)

    if (!evolutionChartRef.current) {
      console.error('Progress chart DOM element is null')
      return
    }

    let chart = chartInstanceRef.current
    if (!chart) {
      console.log('Initializing new progress chart instance')
      chart = echarts.init(evolutionChartRef.current)
      chartInstanceRef.current = chart
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1)

    const option = {
      title: {
        text: '进化曲线',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['最优适应度', '平均适应度'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        name: '代数',
        data: generations
      },
      yAxis: {
        type: 'value',
        name: '适应度',
        scale: true
      },
      series: [
        {
          name: '最优适应度',
          type: 'line',
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: '#3b82f6' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
              ]
            }
          }
        },
        {
          name: '平均适应度',
          type: 'line',
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: '#22c55e' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(34, 197, 94, 0.3)' },
                { offset: 1, color: 'rgba(34, 197, 94, 0.05)' }
              ]
            }
          }
        }
      ]
    }

    try {
      chart.setOption(option, true)
      console.log('Progress chart updated successfully')
    } catch (error) {
      console.error('Error updating progress chart:', error)
    }
  }

  // 更新最终结果图表
  const updateResultChart = (fitnessHistory: { best: number[]; average: number[] }) => {
    console.log('updateResultChart called, DOM element:', resultChartRef.current)

    if (!resultChartRef.current) {
      console.error('Result chart DOM element is null')
      return
    }

    let chart = resultChartInstanceRef.current
    if (!chart) {
      console.log('Initializing new result chart instance')
      chart = echarts.init(resultChartRef.current)
      resultChartInstanceRef.current = chart
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1)

    const option = {
      title: {
        text: '完整进化曲线',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['最优适应度', '平均适应度'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        name: '代数',
        data: generations
      },
      yAxis: {
        type: 'value',
        name: '适应度',
        scale: true
      },
      series: [
        {
          name: '最优适应度',
          type: 'line',
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: '#3b82f6' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
              ]
            }
          }
        },
        {
          name: '平均适应度',
          type: 'line',
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: '#22c55e' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(34, 197, 94, 0.3)' },
                { offset: 1, color: 'rgba(34, 197, 94, 0.05)' }
              ]
            }
          }
        }
      ]
    }

    try {
      chart.setOption(option, true)
      console.log('Result chart updated successfully')
    } catch (error) {
      console.error('Error updating result chart:', error)
    }
  }

  // 保存单个因子
  const saveFactor = async (factor: MinedFactor, index: number) => {
    try {
      // 生成因子名称：Mined_Factor_序号_年月日_股票代码
      const today = new Date()
      const dateStr = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, '0'),
        String(today.getDate()).padStart(2, '0')
      ].join('')

      const factorName = `Mined_Factor_${index + 1}_${dateStr}_${currentStockCode}`

      const factorData = {
        name: factorName,
        code: factor.expression,
        category: '遗传挖掘',
        description: `通过遗传算法挖掘的因子\nIC: ${factor.ic?.toFixed(4)}\nIR: ${factor.ir?.toFixed(4)}\n适应度: ${factor.fitness?.toFixed(4)}`,
        formula_type: 'expression'
      }

      console.log('Saving factor:', factorData)
      const response = await api.createFactor(factorData) as any

      if (response.success) {
        message.success(`因子 "${factorName}" 已保存到自定义因子库`)
        // 刷新因子列表
        await loadFactors()
      } else {
        message.error('保存失败: ' + (response.message || '未知错误'))
      }
    } catch (error: any) {
      console.error('保存因子失败:', error)
      message.error('保存因子失败: ' + (error.message || '未知错误'))
    }
  }

  // 保存全部因子
  const saveAllFactors = async () => {
    if (!miningResult || !miningResult.factors || miningResult.factors.length === 0) {
      message.warning('没有可保存的因子')
      return
    }

    // 生成日期字符串
    const today = new Date()
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, '0'),
      String(today.getDate()).padStart(2, '0')
    ].join('')

    let successCount = 0
    let failCount = 0

    for (let i = 0; i < miningResult.factors.length; i++) {
      const factor = miningResult.factors[i]

      try {
        // 生成因子名称：Mined_Factor_序号_年月日_股票代码
        const factorName = `Mined_Factor_${i + 1}_${dateStr}_${currentStockCode}`

        const factorData = {
          name: factorName,
          code: factor.expression,
          category: '遗传挖掘',
          description: `通过遗传算法挖掘的因子\nIC: ${factor.ic?.toFixed(4)}\nIR: ${factor.ir?.toFixed(4)}\n适应度: ${factor.fitness?.toFixed(4)}`,
          formula_type: 'expression'
        }

        console.log(`Saving factor ${i + 1}/${miningResult.factors.length}:`, factorName)
        const response = await api.createFactor(factorData) as any

        if (response.success) {
          successCount++
          console.log(`Factor ${i + 1} saved successfully`)
        } else {
          failCount++
          console.error(`Factor ${i + 1} save failed:`, response.message)
        }
      } catch (error) {
        console.error(`保存因子 ${i + 1} 失败:`, error)
        failCount++
      }
    }

    // 刷新因子列表
    await loadFactors()

    if (failCount === 0) {
      message.success(`成功保存 ${successCount} 个因子到自定义因子库`)
    } else {
      message.warning(`保存完成: 成功 ${successCount} 个, 失败 ${failCount} 个`)
    }
  }

  // 计算进度百分比
  const getProgressPercent = () => {
    if (!miningStatus) return 0
    return Math.round((miningStatus.current_generation / miningStatus.total_generations) * 100)
  }

  return (
    <div className="factor-mining-container">
      {/* 背景 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      <div className="factor-mining-content">
        <div className="page-header">
          <div className="header-content">
            <RocketOutlined className="header-icon" />
            <div>
              <h1 className="page-title">因子挖掘</h1>
              <p className="page-subtitle">使用遗传算法自动发现最优因子表达式</p>
            </div>
          </div>
          <Button onClick={() => navigate('/factor-management')}>
            返回因子管理
          </Button>
        </div>

        <Row gutter={[24, 24]}>
          {/* 左侧配置面板 */}
          <Col xs={24} lg={8}>
            <Card title="遗传算法配置" className="config-card">
              <Form
                form={form}
                layout="vertical"
                onFinish={startMining}
              >
                {/* 基础配置 */}
                <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">基础配置</Divider>

                <Form.Item
                  label="股票代码"
                  name="stock_code"
                  initialValue="000001"
                  rules={[{ required: true, message: '请输入股票代码' }]}
                >
                  <Input placeholder="例如：000001、600000" />
                </Form.Item>

                <Form.Item
                  label="日期范围"
                  name="dateRange"
                  rules={[{ required: true, message: '请选择日期范围' }]}
                >
                  <RangePicker style={{ width: '100%' }} />
                </Form.Item>

                {/* 基础因子选择 */}
                <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">基础因子选择</Divider>
                <p className="text-hint">选择作为遗传算法输入的基础因子（可搜索因子名称）</p>

                <Form.Item
                  name="base_factors"
                  rules={[{ required: true, message: '请至少选择一个基础因子' }]}
                >
                  <Select
                    mode="multiple"
                    placeholder="输入因子名称搜索，如：RSI、MACD、SMA"
                    style={{ width: '100%' }}
                    showSearch
                    filterOption={(input, option) => {
                      const label = String(option?.label ?? '')
                      const value = String(option?.value ?? '')
                      return label.toLowerCase().includes(input.toLowerCase()) ||
                             value.toLowerCase().includes(input.toLowerCase())
                    }}
                    optionLabelProp="label"
                    maxTagCount="responsive"
                    size="large"
                    classNames={{ popup: "factor-select-dropdown" }}
                    listHeight={400}
                  >
                    {factors.map(factor => (
                      <Option
                        key={factor.id}
                        value={factor.name}
                        label={factor.name}
                      >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontWeight: 500 }}>{factor.name}</span>
                            <Tag color={factor.source === 'preset' ? 'success' : 'warning'}>
                              {factor.source === 'preset' ? '预置' : '自定义'}
                            </Tag>
                            <Tag color="blue">{factor.category}</Tag>
                          </div>
                          <div style={{ fontSize: 12, color: '#64748b', fontFamily: 'monospace' }}>
                            {factor.code}
                          </div>
                          {factor.description && (
                            <div style={{ fontSize: 12, color: '#94a3b8' }}>
                              {factor.description}
                            </div>
                          )}
                        </div>
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate>
                  {() => {
                    const selectedCount = form.getFieldValue('base_factors')?.length || 0
                    return (
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                        <span className="text-hint">
                          已选择 <strong style={{ color: '#3b82f6' }}>{selectedCount}</strong> 个因子
                        </span>
                        <Space size="small">
                          <Button
                            type="link"
                            size="small"
                            onClick={() => {
                              form.setFieldsValue({ base_factors: factors.map(f => f.name) })
                            }}
                          >
                            全选
                          </Button>
                          <Button
                            type="link"
                            size="small"
                            onClick={() => {
                              form.setFieldsValue({ base_factors: [] })
                            }}
                          >
                            清空
                          </Button>
                        </Space>
                      </div>
                    )
                  }}
                </Form.Item>

                {/* 算法参数 */}
                <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">算法参数</Divider>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      label="种群大小"
                      name="population_size"
                      tooltip="每一代的个体数量"
                    >
                      <InputNumber min={10} max={200} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="迭代次数"
                      name="n_generations"
                      tooltip="进化代数"
                    >
                      <InputNumber min={1} max={100} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      label="变异率"
                      name="mutation_rate"
                    >
                      <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="交叉率"
                      name="crossover_rate"
                    >
                      <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item
                  label="精英保留数量"
                  name="elite_size"
                  tooltip="每代保留的最优个体数"
                >
                  <InputNumber min={0} max={20} style={{ width: '100%' }} />
                </Form.Item>

                {/* 适应度函数 */}
                <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">适应度函数</Divider>

                <Form.Item
                  label="优化目标"
                  name="fitness_objective"
                >
                  <Select>
                    <Option value="ic_mean">IC均值</Option>
                    <Option value="ir_ratio">IR比率</Option>
                    <Option value="sharpe">夏普比率</Option>
                    <Option value="combined">综合得分</Option>
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate>
                  {() => {
                    const objective = form.getFieldValue('fitness_objective') || 'ic_mean'

                    let thresholdLabel = '阈值'
                    let thresholdPlaceholder = '0.03'

                    if (objective === 'ic_mean') {
                      thresholdLabel = 'IC阈值'
                      thresholdPlaceholder = '例如：0.03'
                    } else if (objective === 'ir_ratio') {
                      thresholdLabel = 'IR阈值'
                      thresholdPlaceholder = '例如：0.5'
                    } else if (objective === 'sharpe') {
                      thresholdLabel = '夏普阈值'
                      thresholdPlaceholder = '例如：1.0'
                    } else if (objective === 'combined') {
                      thresholdLabel = '综合阈值'
                      thresholdPlaceholder = '例如：0.5'
                    }

                    return (
                      <Form.Item
                        label={thresholdLabel}
                        name="ic_threshold"
                        tooltip={`筛选因子的${thresholdLabel}`}
                      >
                        <InputNumber
                          min={0}
                          step={0.01}
                          style={{ width: '100%' }}
                          placeholder={thresholdPlaceholder}
                        />
                      </Form.Item>
                    )
                  }}
                </Form.Item>

                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    icon={<PlayCircleOutlined />}
                    loading={loading}
                    block
                    size="large"
                    disabled={mining}
                  >
                    {mining ? '挖掘中...' : '开始挖掘'}
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* 右侧结果展示 */}
          <Col xs={24} lg={16}>
            <Card title="挖掘结果" className="result-card">
              {/* 等待提示 */}
              {!mining && !miningStatus && !miningResult && (
                <div className="placeholder-content">
                  <BarChartOutlined className="placeholder-icon" />
                  <p className="placeholder-text">配置参数后点击"开始挖掘"按钮</p>
                  <p className="placeholder-hint">遗传算法将自动搜索最优因子表达式</p>
                </div>
              )}

              {/* 挖掘进度和完成状态 */}
              {miningStatus && !miningResult && (
                <div className="mining-progress">
                  <div className="progress-section">
                    <div className="progress-header">
                      <span className="progress-label">挖掘进度</span>
                      <span className="progress-value">{getProgressPercent()}%</span>
                    </div>
                    <Progress
                      percent={getProgressPercent()}
                      status={mining ? "active" : "success"}
                      strokeColor={{
                        '0%': '#3b82f6',
                        '100%': '#22c55e'
                      }}
                    />
                  </div>

                  <Row gutter={16} style={{ marginTop: 24 }}>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">当前代数</p>
                        <p className="stat-value">{miningStatus.current_generation}/{miningStatus.total_generations}</p>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">最优适应度</p>
                        <p className="stat-value stat-primary">
                          {miningStatus.best_fitness?.toFixed(4) || '-'}
                        </p>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">平均适应度</p>
                        <p className="stat-value">
                          {miningStatus.avg_fitness?.toFixed(4) || '-'}
                        </p>
                      </div>
                    </Col>
                  </Row>

                  <div className="chart-section" style={{ marginTop: 24 }}>
                    <h4 className="chart-title">进化曲线（实时）</h4>
                    <div ref={evolutionChartRef} className="chart-container" style={{ height: '300px' }}></div>
                  </div>
                </div>
              )}

              {/* 挖掘完成提示 */}
              {miningStatus && miningStatus.status === 'completed' && !miningResult && (
                <div style={{ textAlign: 'center', padding: '24px' }}>
                  <Spin size="large" tip="正在加载挖掘结果..." />
                </div>
              )}

              {/* 最终结果 */}
              {miningResult && (
                <div className="mining-result">
                  {/* 挖掘摘要 */}
                  <div className="result-summary" style={{ marginBottom: 24 }}>
                    <Row gutter={16}>
                      <Col span={8}>
                        <div className="stat-item">
                          <p className="stat-label">总代数</p>
                          <p className="stat-value">{miningResult.generations}</p>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div className="stat-item">
                          <p className="stat-label">最优适应度</p>
                          <p className="stat-value stat-primary">
                            {miningResult.best_fitness?.toFixed(4)}
                          </p>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div className="stat-item">
                          <p className="stat-label">发现因子数</p>
                          <p className="stat-value">
                            {miningResult.factors?.length || 0}
                          </p>
                        </div>
                      </Col>
                    </Row>
                  </div>

                  {/* 最终进化曲线 */}
                  <div className="chart-section" style={{ marginBottom: 24 }}>
                    <h4 className="chart-title">完整进化曲线</h4>
                    <div ref={resultChartRef} className="chart-container" style={{ height: '300px' }}></div>
                  </div>

                  <Divider />

                  <h3 className="result-title">发现的因子</h3>

                  {!miningResult.factors || miningResult.factors.length === 0 ? (
                    <Alert
                      message="未发现符合条件的因子"
                      type="info"
                      showIcon
                      style={{ marginTop: 16 }}
                    />
                  ) : (
                    <div className="factors-list">
                      {miningResult.factors.map((factor, index) => (
                        <Card key={index} className="factor-card" size="small">
                          <div className="factor-header">
                            <div className="factor-info">
                              <Space>
                                <Tag color="blue">Top {index + 1}</Tag>
                                <span className="factor-name">
                                  {factor.name || `Factor_${index + 1}`}
                                </span>
                              </Space>
                              <div className="factor-expression">
                                {factor.expression}
                              </div>
                            </div>
                            <div className="factor-stats">
                              <div className="stat-row">
                                <span className="stat-label">IC:</span>
                                <span className={`stat-value ${factor.ic > 0 ? 'positive' : 'negative'}`}>
                                  {factor.ic?.toFixed(4)}
                                </span>
                              </div>
                              <div className="stat-row">
                                <span className="stat-label">IR:</span>
                                <span className={`stat-value ${factor.ir > 0 ? 'positive' : 'negative'}`}>
                                  {factor.ir?.toFixed(4)}
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="factor-actions">
                            <Button
                              type="primary"
                              size="small"
                              icon={<SaveOutlined />}
                              onClick={() => saveFactor(factor, index)}
                            >
                              保存到因子库
                            </Button>
                          </div>
                        </Card>
                      ))}
                    </div>
                  )}

                  <div className="result-actions" style={{ marginTop: 24 }}>
                    <Space>
                      <Button
                        type="primary"
                        icon={<SaveOutlined />}
                        onClick={saveAllFactors}
                      >
                        全部保存到因子库
                      </Button>
                    </Space>
                  </div>
                </div>
              )}
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  )
}

export default FactorMining
