import { useState, useEffect } from 'react';

export function HitCounter() {
    const [count, setCount] = useState(1234);

    useEffect(() => {
        // Fake increment
        const interval = setInterval(() => {
            if (Math.random() > 0.7) {
                setCount(c => c + 1);
            }
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const paddedCount = count.toString().padStart(7, '0');

    return (
        <div className="bg-black border-[3px] border-[#808080] border-r-white border-b-white inline-block px-3 py-1">
            <span className="font-mono text-[#00FF00] font-bold text-xl tracking-[0.2em] drop-shadow-[0_0_5px_rgba(0,255,0,0.7)]">
                {paddedCount}
            </span>
        </div>
    );
}
