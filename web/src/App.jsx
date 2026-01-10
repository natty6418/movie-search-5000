import { useState } from 'react';
import { Button } from './components/Button';
import { Card } from './components/Card';
import { Input, Select } from './components/Input';
import { RetroMarquee } from './components/Marquee';
import { HitCounter } from './components/HitCounter';
import { SearchResult } from './components/SearchResult';
import { searchKeywords, searchSemantic, searchHybrid, performRag } from './api';
import { Search, Zap, Cpu, MessageSquare, Terminal, HelpCircle } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('keyword'); // keyword, semantic, hybrid, rag
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ragAnswer, setRagAnswer] = useState(null);
  
  // Hybrid options
  const [hybridAlpha, setHybridAlpha] = useState(0.5);
  
  // RAG options
  const [ragMode, setRagMode] = useState('rag'); // rag, summarize, citation, question

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setResults([]);
    setRagAnswer(null);

    try {
      let res;
      if (mode === 'keyword') {
        const data = await searchKeywords(query);
        setResults(data.results);
      } else if (mode === 'semantic') {
        const data = await searchSemantic(query);
        setResults(data.results);
      } else if (mode === 'hybrid') {
        const data = await searchHybrid(query, 'weighted', parseFloat(hybridAlpha));
        setResults(data.results);
      } else if (mode === 'rag') {
        const data = await performRag(query, ragMode);
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
      {/* Marquee Banner */}
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
              MOVIE_SEARCH_2000
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
                <div className="flex flex-col md:flex-row gap-4 items-end bg-[#E8E8E8] p-4 border-2 border-[#808080] border-t-black border-l-black">
                    <div className="flex-1 w-full space-y-2">
                        <label className="font-bold uppercase text-sm">Search Query:</label>
                        <Input 
                            value={query} 
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={mode === 'rag' ? "Ask a question..." : "Enter keywords..."}
                            autoFocus
                        />
                    </div>
                    
                    {/* Contextual Options */}
                    {mode === 'hybrid' && (
                        <div className="w-full md:w-32 space-y-2">
                            <label className="font-bold uppercase text-sm" title="0 = Semantic, 1 = Keyword">Alpha ({hybridAlpha})</label>
                            <input 
                                type="range" 
                                min="0" 
                                max="1" 
                                step="0.1"
                                value={hybridAlpha}
                                onChange={(e) => setHybridAlpha(e.target.value)}
                                className="w-full accent-blue-700"
                            />
                        </div>
                    )}

                    {mode === 'rag' && (
                        <div className="w-full md:w-48 space-y-2">
                            <label className="font-bold uppercase text-sm">Mode</label>
                            <Select value={ragMode} onChange={(e) => setRagMode(e.target.value)}>
                                <option value="rag">Standard Q&A</option>
                                <option value="summarize">Summarize</option>
                                <option value="citation">Citations</option>
                                <option value="question">Chat</option>
                            </Select>
                        </div>
                    )}

                    <Button variant="success" size="lg" onClick={handleSearch} disabled={loading}>
                        {loading ? 'Processing...' : 'Search'}
                    </Button>
                </div>

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
                             <div className="whitespace-pre-wrap font-medium leading-relaxed">
                                {ragAnswer}
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
                    <a href="https://github.com/bootdotdev" target="_blank" className="font-heading text-xl">BOOT.DEV</a>
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
                Copyright © 1997-2026 RAG Search Engine Corp. All rights reserved.<br/>
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