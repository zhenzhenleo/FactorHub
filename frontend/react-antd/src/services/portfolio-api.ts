/**
 * 组合分析页面 - API 服务扩展
 * 添加组合分析相关的 API 调用
 */

import request from './api'

// 组合分析 API
export const portfolioApi = {
  // 权重优化
  optimizeWeights(data: {
    factors: string[]
    start_date: string
    end_date: string
    method: string
    rebalance_frequency: string
    risk_free_rate: number
    min_weight: number
    max_weight: number
    target_return?: number
  }) {
    return request.post('/portfolio/optimize-weights', data)
  },

  // 计算综合得分
  calculateCompositeScore(data: {
    factors: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/portfolio/composite-score', data)
  },

  // 对比权重方法
  compareWeightMethods(data: {
    factors: string[]
    start_date: string
    end_date: string
    methods: string[]
  }) {
    return request.post('/portfolio/compare-methods', data)
  }
}

export default portfolioApi
