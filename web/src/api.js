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

export const performAgentAction = async (query, chat_history = []) => {
    const response = await api.post('/agent', { query, chat_history });
    return response.data;
};

export const streamAgentAction = (query, sessionId, onStatus, onUpdate, onComplete, onError) => {
    const url = `${API_BASE_URL}/agent/stream`;

    // Use fetch with POST for streaming (EventSource doesn't support POST directly)
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            session_id: sessionId,
            chat_history: []
        })
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        const readStream = () => {
            reader.read().then(({ done, value }) => {
                if (done) return;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'status') {
                                onStatus(data.message, data.node);
                            } else if (data.type === 'update') {
                                onUpdate(data);
                            } else if (data.type === 'complete') {
                                onComplete(data);
                                return;
                            } else if (data.type === 'error') {
                                onError(data.message);
                                return;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE:', e);
                        }
                    }
                }
                
                readStream();
            }).catch(onError);
        };
        
        readStream();
    }).catch(onError);
};

export const incrementVisit = async () => {
    const response = await api.post('/visit');
    return response.data;
};
