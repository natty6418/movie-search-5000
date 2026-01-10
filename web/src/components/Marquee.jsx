import Marquee from 'react-fast-marquee';
import { cn } from './Button';

export function RetroMarquee({ text, className, speed = 40 }) {
  return (
    <div className={cn("bg-black text-[#00FF00] font-mono py-1 border-y-2 border-retro-border-dark", className)}>
      <Marquee speed={speed} gradient={false}>
         {text.map((item, idx) => (
             <span key={idx} className="mx-8 font-bold uppercase tracking-widest">
                {item}
             </span>
         ))}
      </Marquee>
    </div>
  );
}
