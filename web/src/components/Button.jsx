import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function Button({ 
  className, 
  variant = 'default', 
  size = 'md', 
  children, 
  ...props 
}) {
  const variants = {
    default: 'bg-retro-bg text-black bevel-outset hover:bg-[#d0d0d0] active:bevel-inset active:translate-x-[1px] active:translate-y-[1px]',
    primary: 'bg-retro-accent text-white [border-color:#5555ff_#000080_#000080_#5555ff] bevel-outset hover:brightness-110 active:bevel-inset active:translate-x-[1px] active:translate-y-[1px] active:[border-color:#000080_#5555ff_#5555ff_#000080]',
    danger: 'bg-retro-secondary text-white [border-color:#ff5555_#800000_#800000_#ff5555] bevel-outset hover:brightness-110 active:bevel-inset active:translate-x-[1px] active:translate-y-[1px] active:[border-color:#800000_#ff5555_#ff5555_#800000]',
    success: 'bg-retro-success-dark text-white [border-color:#00ff00_#006600_#006600_#00ff00] bevel-outset hover:brightness-110 active:bevel-inset active:translate-x-[1px] active:translate-y-[1px] active:[border-color:#006600_#00ff00_#00ff00_#006600]',
    outline: 'bg-white text-black bevel-outset hover:bg-gray-100 active:bevel-inset active:translate-x-[1px] active:translate-y-[1px]',
  };

  const sizes = {
    sm: 'px-2 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
  };

  return (
    <button 
      className={cn(
        'font-bold uppercase tracking-wide transition-none select-none',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
