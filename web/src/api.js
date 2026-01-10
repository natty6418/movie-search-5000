import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchKeywords = async (query, limit = 10) => {
  const response = await api.post('/search/keyword', { query, limit });
  return response.data;
};

export const searchSemantic = async (query, limit = 10) => {
  const response = await api.post('/search/semantic', { query, limit });
  return response.data;
};

export const searchHybrid = async (query, mode = 'weighted', alpha = 0.5, k = 60, limit = 10) => {
  const response = await api.post('/search/hybrid', { query, mode, alpha, k, limit });
  return response.data;
};

export const performRag = async (query, mode = 'rag', limit = 5) => {
  const response = await api.post('/rag', { query, mode, limit });
  return response.data;
};
