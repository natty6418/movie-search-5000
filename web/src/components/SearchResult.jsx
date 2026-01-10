import { cn } from './Button';

export function SearchResult({ result, index }) {
  return (
    <div className={cn(
      "p-2 border-b-2 border-r-2 border-[#808080] flex flex-col gap-1",
      index % 2 === 0 ? "bg-white" : "bg-[#E8E8E8]"
    )}>
       <div className="flex items-start justify-between">
         <h4 className="font-bold text-retro-title-bar underline cursor-pointer hover:text-retro-hover visited:text-retro-visited text-lg leading-tight">
            {result.title}
         </h4>
         <span className="font-mono text-xs bg-retro-panel-yellow px-1 border border-black">
           ID: {result.id}
         </span>
       </div>
       
       <p className="text-sm line-clamp-2 leading-snug">
         {result.description}
       </p>
       
       {result.score !== undefined && (
          <div className="text-xs font-mono text-retro-muted mt-1">
            Relevance Score: <span className="text-black font-bold">{typeof result.score === 'number' ? result.score.toFixed(4) : result.score}</span>
          </div>
       )}
       
       {result.details && (
         <div className="text-xs mt-1 text-retro-muted">
           Match: {Object.entries(result.details).map(([k,v]) => `${k}=${typeof v === 'number' ? v.toFixed(2) : v}`).join(', ')}
         </div>
       )}
    </div>
  );
}
