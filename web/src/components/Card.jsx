import { cn } from './Button';

export function Card({ title, children, className, contentClassName }) {
  return (
    <div className={cn("bg-retro-bg bevel-outset p-1", className)}>
      {title && (
        <div className="bg-gradient-to-r from-retro-title-bar to-retro-title-bar-end text-white px-2 py-1 font-bold flex items-center justify-between mb-1 select-none">
          <span className="truncate">{title}</span>
          <div className="flex gap-1">
             <div className="w-4 h-4 bg-retro-bg bevel-outset flex items-center justify-center text-black text-xs leading-none font-sans font-bold cursor-pointer active:bevel-inset">_</div>
             <div className="w-4 h-4 bg-retro-bg bevel-outset flex items-center justify-center text-black text-xs leading-none font-sans font-bold cursor-pointer active:bevel-inset">X</div>
          </div>
        </div>
      )}
      <div className={cn("bevel-inset bg-white p-4", contentClassName)}>
        {children}
      </div>
    </div>
  );
}
