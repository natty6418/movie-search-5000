import { cn } from './Button';

export function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        "bevel-inset bg-white px-2 py-1 outline-none focus:outline-none focus:ring-0 w-full font-sans",
        className
      )}
      {...props}
    />
  );
}

export function Select({ className, children, ...props }) {
  return (
    <select
      className={cn(
        "bevel-inset bg-white px-2 py-1 outline-none focus:outline-none focus:ring-0 font-sans cursor-pointer",
        className
      )}
      {...props}
    >
      {children}
    </select>
  );
}
