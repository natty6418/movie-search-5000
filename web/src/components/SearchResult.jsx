import { useState } from 'react';
import { cn } from './Button';
import { PlusSquare, MinusSquare } from 'lucide-react';

export function SearchResult({ result, index }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div 
      className={cn(
        "p-2 border-b-2 border-r-2 border-[#808080] flex flex-col gap-1 cursor-pointer transition-colors",
        index % 2 === 0 ? "bg-white" : "bg-[#E8E8E8]",
        expanded ? "bg-blue-50" : "hover:bg-blue-50"
      )}
      onClick={() => setExpanded(!expanded)}
    >
       <div className="flex items-start justify-between">
         <div className="flex items-start gap-2">
            <span className="mt-1 text-retro-muted">
                {expanded ? <MinusSquare size={16} /> : <PlusSquare size={16} />}
            </span>
            <h4 className="font-bold text-retro-title-bar underline hover:text-retro-hover visited:text-retro-visited text-lg leading-tight select-none">
                {result.title}
            </h4>
         </div>
         <span className="font-mono text-xs bg-retro-panel-yellow px-1 border border-black shrink-0 ml-2">
           ID: {result.id}
         </span>
       </div>
       
       <p className={cn("text-sm leading-snug", !expanded && "line-clamp-2")}>
         {result.description}
       </p>
       
       {result.score !== undefined && (
          <div className="text-xs font-mono text-retro-muted mt-1">
            Relevance Score: <span className="text-black font-bold">{typeof result.score === 'number' ? result.score.toFixed(4) : result.score}</span>
          </div>
       )}
       
       {expanded && result.details && (
         <div className="mt-2 p-2 border-2 border-[#808080] bevel-inset bg-[#C0C0C0] font-mono text-xs">
            <div className="font-bold border-b border-[#808080] mb-1 pb-1">Debug Info:</div>
            <pre className="whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(result.details, null, 2)}
            </pre>
         </div>
       )}
    </div>
  );
}
