import ReactMarkdown from 'react-markdown';
import { useState } from 'react';
import { Button } from './components/Button';
import { Card } from './components/Card';
import { Input, Select } from './components/Input';
import { RetroMarquee } from './components/Marquee';
import { HitCounter } from './components/HitCounter';
import { SearchResult } from './components/SearchResult';
import { searchKeywords, searchSemantic, searchHybrid, performRag, performAgentAction } from './api';
import { Search, Zap, Cpu, MessageSquare, Terminal, HelpCircle, Bot } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('keyword'); // keyword, semantic, hybrid, rag, agent
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ragAnswer, setRagAnswer] = useState(null);
  
  // New state for enhanced query display
  const [enhancedQuery, setEnhancedQuery] = useState(null);
  const [enhanceMode, setEnhanceMode] = useState('none'); // none, fix_spelling, rewrite, expand
  const [bm25Type, setBm25Type] = useState('combined'); // unigram, bigram, combined
  
  // RAG options
  const [ragMode, setRagMode] = useState('rag'); // rag, summarize, citation, question

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setResults([]);
    setRagAnswer(null);
    setEnhancedQuery(null);

    try {
      if (mode === 'keyword') {
        const data = await searchKeywords(query, 10, enhanceMode, bm25Type);
        setResults(data.results);
        if (data.query_used && data.query_used !== query) setEnhancedQuery(data.query_used);
      } else if (mode === 'semantic') {
        const data = await searchSemantic(query, 10, enhanceMode);
        setResults(data.results);
        if (data.query_used && data.query_used !== query) setEnhancedQuery(data.query_used);
      } else if (mode === 'hybrid') {
        const data = await searchHybrid(query, 'rrf', 0.5, 60, 10, enhanceMode, bm25Type);
        setResults(data.results);
        if (data.query_used && data.query_used !== query) setEnhancedQuery(data.query_used);
      } else if (mode === 'rag') {
        const data = await performRag(query, ragMode, 5, enhanceMode, bm25Type);
        if (data.query_used && data.query_used !== query) setEnhancedQuery(data.query_used);
        
        if (ragMode === 'rag') {
            setRagAnswer(data.answer);
            setResults(data.docs.map(d => ({ ...d, score: 'N/A' })));
        } else if (ragMode === 'summarize') {
            setRagAnswer(data.summary);
            setResults(data.docs.map(d => ({ ...d, score: 'N/A' })));
        } else if (ragMode === 'citation') {
            setRagAnswer(data.citations);
            setResults(data.docs.map(d => ({ ...d, score: 'N/A' })));
        } else if (ragMode === 'question') {
            setRagAnswer(data.answer);
            setResults(data.docs.map(d => ({ ...d, score: 'N/A' })));
        }
      } else if (mode === 'agent') {
        const data = await performAgentAction(query);
        setRagAnswer(data.answer);
        setResults([]); // Agent doesn't return raw results in this simplified view
      }
    } catch (err) {
      console.error(err);
      alert("Error performing search! Check console.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  }

  return (
    <div className="min-h-screen pb-16">
      {/* ... (rest of header) ... */}
      <RetroMarquee 
        text={[
            "WELCOME TO THE FUTURE OF MOVIE SEARCH", 
            "POWERED BY RAG TECHNOLOGY", 
            "UNDER CONSTRUCTION", 
            "BEST VIEWED IN NETSCAPE NAVIGATOR 4.0",
            "SIGN OUR GUESTBOOK!"
        ]} 
      />

      <div className="max-w-5xl mx-auto p-4 space-y-8">
        
        {/* Header Section */}
        <header className="text-center py-8 space-y-4">
           <h1 className="text-5xl md:text-7xl animate-rainbow drop-shadow-md select-none">
              MOVIE_SEARCH_5000
           </h1>
           <div className="inline-block bg-construction px-4 py-1 border-2 border-black rotate-[-2deg]">
              <span className="bg-black text-yellow-300 font-bold px-2 py-1 text-xl uppercase tracking-widest">
                  Beta v0.1
              </span>
           </div>
        </header>

        {/* Main Interface Window */}
        <Card title="SearchControl.exe" className="w-full">
            <div className="space-y-6">
                
                {/* Tabs */}
                <div className="flex flex-wrap gap-2 border-b-2 border-[#808080] pb-2">
                    {[
                        { id: 'keyword', label: 'Keyword Search', icon: Search },
                        { id: 'semantic', label: 'Semantic AI', icon: Zap },
                        { id: 'hybrid', label: 'Hybrid Fusion', icon: Cpu },
                        { id: 'rag', label: 'RAG Assistant', icon: MessageSquare },
                        { id: 'agent', label: 'Auto Agent', icon: Bot },
                    ].map(tab => (
                        <Button 
                            key={tab.id}
                            variant={mode === tab.id ? 'primary' : 'default'}
                            onClick={() => setMode(tab.id)}
                            className="flex items-center gap-2"
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </Button>
                    ))}
                </div>

                {/* Search Bar */}
                <div className="flex flex-col lg:flex-row gap-4 items-end lg:items-center bg-[#E8E8E8] p-4 border-2 border-[#808080] border-t-black border-l-black">
                    <div className="flex-1 w-full space-y-2">
                        <label className="font-bold uppercase text-sm">Search Query:</label>
                        <Input 
                            value={query} 
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={mode === 'rag' || mode === 'agent' ? "Ask a question..." : "Enter keywords..."}
                            autoFocus
                        />
                    </div>

                    <div className="flex flex-wrap items-end gap-2 w-full lg:w-auto">
                        <div className="space-y-2 flex-1 lg:flex-none">
                             <label className="font-bold uppercase text-xs block">Enhance</label>
                             <Select value={enhanceMode} onChange={(e) => setEnhanceMode(e.target.value)} className="w-full lg:w-32 text-sm">
                                <option value="none">None</option>
                                <option value="fix_spelling">Fix Spelling</option>
                                <option value="rewrite">Rewrite</option>
                                <option value="expand">Expand</option>
                            </Select>
                        </div>

                        {mode !== 'semantic' && mode !== 'agent' && (
                            <div className="space-y-2 flex-1 lg:flex-none">
                                <label className="font-bold uppercase text-xs block">Keywords</label>
                                <Select value={bm25Type} onChange={(e) => setBm25Type(e.target.value)} className="w-full lg:w-28 text-sm">
                                    <option value="combined">Auto</option>
                                    <option value="unigram">Standard</option>
                                    <option value="bigram">Phrases</option>
                                </Select>
                            </div>
                        )}

                        {mode === 'rag' && (
                            <div className="space-y-2 flex-1 lg:flex-none">
                                <label className="font-bold uppercase text-xs block">Mode</label>
                                <Select value={ragMode} onChange={(e) => setRagMode(e.target.value)} className="w-full lg:w-32 text-sm">
                                    <option value="rag">Q&A</option>
                                    <option value="summarize">Summarize</option>
                                    <option value="citation">Citations</option>
                                    <option value="question">Chat</option>
                                </Select>
                            </div>
                        )}

                        <Button variant="success" size="lg" onClick={handleSearch} disabled={loading} className="w-full lg:w-auto h-[34px] flex items-center justify-center">
                            {loading ? 'Processing...' : 'Search'}
                        </Button>
                    </div>
                </div>
                
                {/* Enhanced Query Display */}
                {enhancedQuery && (
                    <div className="bg-black text-[#00FF00] font-mono text-sm p-2 border-2 border-[#808080] bevel-inset">
                        <span className="font-bold mr-2">root@system:~$</span>
                        <span className="opacity-70">Enhanced Query: </span>
                        "{enhancedQuery}"
                    </div>
                )}

            </div>
        </Card>

        {/* Content Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Left Column: Results */}
            <div className={clsx("space-y-6", ragAnswer ? "md:col-span-1" : "md:col-span-3")}>
                <Card title={`Results (${results.length})`} contentClassName="p-0 bg-[#C0C0C0]">
                    {loading && (
                        <div className="p-8 text-center bg-white">
                            <div className="animate-spin inline-block mb-2">⏳</div>
                            <p className="font-bold animate-pulse">Searching the World Wide Web...</p>
                        </div>
                    )}
                    
                    {!loading && results.length === 0 && (
                        <div className="p-8 text-center bg-white">
                            <p className="text-retro-muted italic">Ready to search.</p>
                        </div>
                    )}

                    {!loading && results.length > 0 && (
                        <div className="max-h-[600px] overflow-y-auto border-t-2 border-[#808080]">
                            {results.map((result, idx) => (
                                <SearchResult key={result.id || idx} result={result} index={idx} />
                            ))}
                        </div>
                    )}
                </Card>
            </div>

            {/* Right Column: RAG Answer (Only visible in RAG mode with answer) */}
            {ragAnswer && (
                <div className="md:col-span-2 space-y-6">
                    <Card title="AI_Assistant_Response.txt">
                        <div className="prose prose-sm max-w-none font-sans bg-white p-2">
                             <div className="flex items-center gap-2 mb-4 border-b-2 border-[#E8E8E8] pb-2">
                                <span className="bg-retro-secondary text-white text-xs px-1 font-bold animate-pulse-glow">NEW!</span>
                                <h3 className="m-0 text-lg">Generated Response</h3>
                             </div>
                             <div className="font-medium leading-relaxed">
                                <ReactMarkdown>{ragAnswer}</ReactMarkdown>
                             </div>
                        </div>
                        <div className="mt-4 flex justify-end">
                            <Button size="sm" onClick={() => {navigator.clipboard.writeText(ragAnswer); alert("Copied to clipboard!")}}>
                                Copy Text
                            </Button>
                        </div>
                    </Card>

                    <div className="bg-retro-panel-yellow border-2 border-black p-4 shadow-[4px_4px_0_rgba(0,0,0,0.5)]">
                        <h4 className="flex items-center gap-2 font-bold mb-2">
                            <HelpCircle size={16} />
                            Did you know?
                        </h4>
                        <p className="text-sm">
                            This response was generated by a local Large Language Model (LLM) running on your machine. 
                            It analyzed the top {results.length} search results to synthesize this answer.
                        </p>
                    </div>
                </div>
            )}
        </div>

        <hr className="hr-groove my-12" />

        {/* Footer */}
        <footer className="text-center space-y-6 pb-8">
            <div className="flex justify-center gap-8">
                 <div className="text-center">
                    <p className="font-bold text-xs uppercase mb-1">Created By</p>
                    <a href="https://github.com/natty6418" target="_blank" className="font-heading text-xl">Natty6416</a>
                 </div>
                 <div className="text-center">
                    <p className="font-bold text-xs uppercase mb-1">Visitor Count</p>
                    <HitCounter />
                 </div>
            </div>
            <div className="flex justify-center gap-2 text-xs font-mono text-retro-muted">
                <span>[Home]</span>
                <span>[Email Us]</span>
                <span>[Links]</span>
                <span>[Web Ring]</span>
            </div>

            <p className="text-xs">
                Copyright © 1997-2026 MovieSearch 5000 Corp. All rights reserved.<br/>
                Optimized for 800x600 resolution.
            </p>
            
            <div className="flex justify-center mt-4">
                 <img src="https://gifcities.org/assets/construction.gif" alt="" className="h-8 opacity-0" /> {/* Placeholder for layout */}
            </div>
        </footer>

      </div>
    </div>
  );
}

export default App;