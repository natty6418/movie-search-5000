import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchKeywords = async (query, limit = 10, enhance_mode = 'none', bm25_type = 'combined') => {
  const response = await api.post('/search/keyword', { query, limit, enhance_mode, bm25_type });
  return response.data;
};

export const searchSemantic = async (query, limit = 10, enhance_mode = 'none') => {
  const response = await api.post('/search/semantic', { query, limit, enhance_mode });
  return response.data;
};

export const searchHybrid = async (query, mode = 'weighted', alpha = 0.5, k = 60, limit = 10, enhance_mode = 'none', bm25_type = 'combined') => {
  const response = await api.post('/search/hybrid', { query, mode, alpha, k, limit, enhance_mode, bm25_type });
  return response.data;
};

export const performRag = async (query, mode = 'rag', limit = 5, enhance_mode = 'none', bm25_type = 'combined') => {
  const response = await api.post('/rag', { query, mode, limit, enhance_mode, bm25_type });
  return response.data;
};

export const incrementVisit = async () => {
    const response = await api.post('/visit');
    return response.data;
};
