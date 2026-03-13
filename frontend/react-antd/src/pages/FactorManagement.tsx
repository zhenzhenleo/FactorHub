import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Button,
  Input,
  Select,
  Space,
  Modal,
  Form,
  InputNumber,
  message,
  Tag,
  Card,
  Checkbox,
  Row,
  Col,
  Tabs
} from 'antd'
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table'
import {
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  DeleteOutlined,
  EyeOutlined,
  CheckOutlined,
  CopyOutlined
} from '@ant-design/icons'
import { api } from '@/services/api'
import './FactorManagement.css'

const { TextArea } = Input
const { Option } = Select

interface Factor {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
  formula_type?: string
}


const FactorManagement: React.FC = () => {
  const navigate = useNavigate()
  const [factors, setFactors] = useState<Factor[]>([])
  const [filteredFactors, setFilteredFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(false)
  const [categories, setCategories] = useState<string[]>([])

  // Tab状态
  const [activeTab, setActiveTab] = useState<'user' | 'preset'>('user')

  // 筛选状态
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [sourceFilter, setSourceFilter] = useState<string>('')
  const [searchText, setSearchText] = useState<string>('')

  // 分页状态
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0
  })

  // 弹窗状态
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [batchModalVisible, setBatchModalVisible] = useState(false)
  const [form] = Form.useForm()
  const [batchForm] = Form.useForm()

  // 加载因子列表
  const loadFactors = async () => {
    setLoading(true)
    try {
      const response = await api.getFactors() as any
      if (response.success) {
        setFactors(response.data)
        // 提取分类列表
        const cats = [...new Set((response.data as Factor[]).map((f: Factor) => f.category).filter(Boolean))] as string[]
        setCategories(cats)
      }
    } catch (error) {
      message.error('加载因子列表失败')
    } finally {
      setLoading(false)
    }
  }

  // Tab切换时自动设置sourceFilter
  useEffect(() => {
    setSourceFilter(activeTab)
    setCategoryFilter('')  // 重置分类筛选
  }, [activeTab])

  // 筛选因子
  useEffect(() => {
    let filtered = [...factors]

    if (categoryFilter) {
      filtered = filtered.filter(f => f.category === categoryFilter)
    }

    if (sourceFilter) {
      filtered = filtered.filter(f => f.source === sourceFilter)
    }

    if (searchText) {
      const searchLower = searchText.toLowerCase()
      filtered = filtered.filter(f =>
        f.name.toLowerCase().includes(searchLower) ||
        (f.description && f.description.toLowerCase().includes(searchLower)) ||
        f.code.toLowerCase().includes(searchLower)
      )
    }

    setFilteredFactors(filtered)
    setPagination(prev => ({ ...prev, total: filtered.length, current: 1 }))
  }, [factors, categoryFilter, sourceFilter, searchText])

  // 创建因子
  const handleCreateFactor = async (values: any) => {
    try {
      const response = await api.createFactor(values) as any
      if (response.success) {
        message.success('因子创建成功')
        setCreateModalVisible(false)
        form.resetFields()
        loadFactors()
      } else {
        message.error(response.message || '创建失败')
      }
    } catch (error) {
      message.error('创建因子失败')
    }
  }

  // 验证公式
  const handleValidateFormula = async () => {
    const code = form.getFieldValue('code')
    const formulaType = form.getFieldValue('formula_type')

    if (!code) {
      message.warning('请先输入因子代码')
      return
    }

    try {
      const response = await api.validateFactor({
        code,
        formula_type: formulaType
      } as any) as any
      if (response.success) {
        message.success('公式验证通过')
      } else {
        message.error(response.message || '公式验证失败')
      }
    } catch (error) {
      message.error('验证失败')
    }
  }

  // 删除因子
  const handleDeleteFactor = async (id: number, name: string) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除因子 "${name}" 吗？`,
      onOk: async () => {
        try {
          const response = await api.deleteFactor(id) as any
          if (response.success) {
            message.success('删除成功')
            loadFactors()
          } else {
            message.error(response.message || '删除失败')
          }
        } catch (error) {
          message.error('删除失败')
        }
      }
    })
  }

  // 复制因子
  const handleCopyFactor = async (id: number, name: string) => {
    try {
      const response = await api.copyFactor(id) as any
      if (response.success) {
        message.success(`因子已复制为 "${response.data.name}"`)
        loadFactors()
      } else {
        message.error(response.message || '复制失败')
      }
    } catch (error) {
      message.error('复制失败')
    }
  }

  // 批量生成因子
  const handleBatchGenerate = async (values: any) => {
    const { base_factors, methods, ic_threshold, ir_threshold, min_valid_ratio } = values

    if (!base_factors || base_factors.length === 0) {
      message.warning('请至少选择一个基础因子')
      return
    }

    if (!methods || methods.length === 0) {
      message.warning('请至少选择一种生成方法')
      return
    }

    try {
      message.loading({ content: '批量生成中，请稍候...', key: 'batch' })
      const response = await api.batchGenerateFactors({
        base_factors,
        generate_methods: methods,
        ic_threshold,
        ir_threshold,
        min_valid_ratio
      } as any) as any

      if (response.success) {
        message.success({ content: `生成完成！生成因子数：${response.data.generated_count || 0}，通过筛选：${response.data.passed_count || 0}`, key: 'batch' })
        setBatchModalVisible(false)
        batchForm.resetFields()
        loadFactors()
      } else {
        message.error({ content: response.message || '批量生成失败', key: 'batch' })
      }
    } catch (error) {
      message.error({ content: '批量生成失败', key: 'batch' })
    }
  }

  // 表格列定义
  const columns: ColumnsType<Factor> = [
    {
      title: '因子名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: Factor) => (
        <div>
          <div className="factor-name">{text}</div>
          <div className="factor-code">{record.code}</div>
        </div>
      )
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (text: string) => <Tag color="blue">{text || '-'}</Tag>
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 100,
      render: (text: string) => (
        <Tag color={text === 'preset' ? 'success' : 'warning'}>
          {text === 'preset' ? '预置' : '自定义'}
        </Tag>
      )
    },
    {
      title: '说明',
      dataIndex: 'description',
      key: 'description',
      width: 300,
      ellipsis: true,
      render: (text: string) => text || '-'
    },
    {
      title: '操作',
      key: 'action',
      width: 200,
      fixed: 'right',
      render: (_: any, record: Factor) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => navigate(`/factor-detail?id=${record.id}`)}
          >
            查看
          </Button>
          <Button
            type="link"
            size="small"
            icon={<CopyOutlined />}
            onClick={() => handleCopyFactor(record.id, record.name)}
          >
            复制
          </Button>
          {record.source === 'user' && (
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteFactor(record.id, record.name)}
            >
              删除
            </Button>
          )}
        </Space>
      )
    }
  ]

  // 分页配置
  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination({
      current: newPagination.current || 1,
      pageSize: newPagination.pageSize || 20,
      total: filteredFactors.length
    })
  }

  useEffect(() => {
    loadFactors()
  }, [])

  return (
    <div className="factor-management-container">
      {/* 背景装饰 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      {/* 页面标题 */}
      <div className="page-header">
        <h1 className="page-title">因子管理</h1>
        <p className="page-subtitle">创建、管理和分析量化因子</p>
      </div>

      {/* Tab分类 */}
      <Card className="tab-card" bordered={false} style={{ marginBottom: '16px' }}>
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as 'user' | 'preset')}
        >
          <Tabs.TabPane tab={`自定义因子 (${factors.filter(f => f.source === 'user').length})`} key="user">
            {null}
          </Tabs.TabPane>
          <Tabs.TabPane tab={`系统预置因子 (${factors.filter(f => f.source === 'preset').length})`} key="preset">
            {null}
          </Tabs.TabPane>
        </Tabs>
      </Card>

      {/* 工具栏 */}
      <Card className="toolbar-card" bordered={false}>
        <div className="toolbar">
          <div className="filters">
            <Space size="middle" wrap>
              <div>
                <div className="filter-label">分类筛选</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 150 }}
                  value={categoryFilter || undefined}
                  onChange={setCategoryFilter}
                >
                  {categories.map(cat => (
                    <Option key={cat} value={cat}>{cat}</Option>
                  ))}
                </Select>
              </div>
              {/* 来源筛选已由Tab替代 */}
              {/* <div>
                <div className="filter-label">来源筛选</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 120 }}
                  value={sourceFilter || undefined}
                  onChange={setSourceFilter}
                >
                  <Option value="preset">预置</Option>
                  <Option value="user">自定义</Option>
                </Select>
              </div> */}
              <div>
                <div className="filter-label">搜索</div>
                <Input
                  placeholder="搜索因子名称..."
                  prefix={<SearchOutlined />}
                  style={{ width: 200 }}
                  value={searchText}
                  onChange={e => setSearchText(e.target.value)}
                  allowClear
                />
              </div>
            </Space>
          </div>
          <div className="actions">
            <Space size="small">
              <Button
                icon={<ReloadOutlined />}
                onClick={loadFactors}
                loading={loading}
              >
                刷新
              </Button>
              {activeTab === 'user' && (
                <>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => {
                      setCreateModalVisible(true)
                    }}
                  >
                    新增因子
                  </Button>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => setBatchModalVisible(true)}
                  >
                    批量生成
                  </Button>
                </>
              )}
            </Space>
          </div>
        </div>
      </Card>

      {/* 因子列表表格 */}
      <Card className="table-card" bordered={false}>
        <Table
          columns={columns}
          dataSource={filteredFactors}
          rowKey="id"
          loading={loading}
          pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            total: pagination.total,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `显示第 ${range[0]} 到 ${range[1]} 条，共 ${total} 条`,
            pageSizeOptions: ['10', '20', '50', '100']
          }}
          onChange={handleTableChange}
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* 新增因子弹窗 */}
      <Modal
        title="创建新因子"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateFactor}
        >
          <Form.Item
            label="因子名称"
            name="name"
            rules={[{ required: true, message: '请输入因子名称' }]}
          >
            <Input placeholder="例如：RSI指标" />
          </Form.Item>

          <Form.Item
            label="分类"
            name="category"
            rules={[{ required: true, message: '请选择分类' }]}
          >
            <Select placeholder="请选择">
              <Option value="技术指标">技术指标</Option>
              <Option value="价格动量">价格动量</Option>
              <Option value="成交量">成交量</Option>
              <Option value="波动率">波动率</Option>
              <Option value="自定义">自定义</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="说明"
            name="description"
          >
            <TextArea rows={3} placeholder="简要描述因子的含义和用途" />
          </Form.Item>

          <Form.Item
            label="公式类型"
            name="formula_type"
            initialValue="expression"
          >
            <Select>
              <Option value="expression">表达式</Option>
              <Option value="function">函数</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="因子代码"
            name="code"
            rules={[{ required: true, message: '请输入因子代码' }]}
            extra="支持 Python 表达式，可使用 close, open, high, low, volume 等字段"
          >
            <TextArea
              rows={6}
              placeholder="例如：RSI(close, 14)"
              className="font-mono"
              style={{
                backgroundColor: '#f6f8fa',
                fontSize: '14px',
                fontFamily: 'Consolas, Monaco, monospace',
                borderRadius: '6px',
                padding: '12px',
                minHeight: '150px',
                maxHeight: '300px',
                overflowY: 'auto'
              }}
            />
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Button type="primary" htmlType="submit" style={{ flex: 1 }}>
                创建因子
              </Button>
              <Button onClick={handleValidateFormula}>
                验证公式
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 批量生成因子弹窗 */}
      <Modal
        title="批量生成因子"
        open={batchModalVisible}
        onCancel={() => {
          setBatchModalVisible(false)
          batchForm.resetFields()
        }}
        footer={null}
        width={800}
        destroyOnClose
      >
        <Form
          form={batchForm}
          layout="vertical"
          onFinish={handleBatchGenerate}
        >
          <Form.Item label="选择基础因子" name="base_factors">
            <Checkbox.Group style={{ width: '100%' }}>
              <Row gutter={[16, 16]}>
                {factors.map(factor => (
                  <Col span={8} key={factor.id}>
                    <Checkbox value={factor.code}>
                      <span title={factor.code}>{factor.name}</span>
                    </Checkbox>
                  </Col>
                ))}
              </Row>
            </Checkbox.Group>
          </Form.Item>

          <Form.Item label="生成方法" name="methods">
            <Checkbox.Group>
              <Row gutter={16}>
                <Col span={8}>
                  <Checkbox value="arithmetic">算术运算</Checkbox>
                </Col>
                <Col span={8}>
                  <Checkbox value="statistics">统计变换</Checkbox>
                </Col>
                <Col span={8}>
                  <Checkbox value="technical">技术指标</Checkbox>
                </Col>
              </Row>
            </Checkbox.Group>
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                label="IC阈值"
                name="ic_threshold"
                initialValue={0.03}
              >
                <InputNumber step={0.01} min={0} max={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="IR阈值"
                name="ir_threshold"
                initialValue={0.5}
              >
                <InputNumber step={0.1} min={0} max={10} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="最小有效率"
                name="min_valid_ratio"
                initialValue={0.7}
              >
                <InputNumber step={0.1} min={0} max={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Button type="primary" htmlType="submit" block icon={<CheckOutlined />}>
              开始批量生成
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default FactorManagement
