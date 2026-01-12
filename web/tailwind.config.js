/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'retro-bg': '#C0C0C0',
        'retro-fg': '#000000',
        'retro-muted': '#808080',
        'retro-accent': '#0000FF',
        'retro-secondary': '#FF0000',
        'retro-tertiary': '#FFFF00',
        'retro-success': '#00FF00',
        'retro-success-dark': '#00AA00',
        'retro-border-light': '#FFFFFF',
        'retro-border-dark': '#808080',
        'retro-title-bar': '#000080',
        'retro-title-bar-end': '#1084D0',
        'retro-panel-yellow': '#FFFFCC',
        'retro-visited': '#800080',
        'retro-hover': '#FF0000',
      },
      fontFamily: {
        sans: ['"MS Sans Serif"', '"Segoe UI"', 'Tahoma', 'Geneva', 'Verdana', 'sans-serif'],
        heading: ['"Arial Black"', 'Impact', 'Haettenschweiler', 'sans-serif'],
        mono: ['"Courier New"', 'Courier', 'monospace'],
        comic: ['"Comic Sans MS"', 'cursive'],
      },
      boxShadow: {
        'outset': 'inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf',
        'outset-deep': 'inset -2px -2px 0 #808080, inset 2px 2px 0 #fff, inset -4px -4px 0 #404040, inset 4px 4px 0 #dfdfdf',
        'inset': 'inset 1px 1px 0 #404040, inset -1px -1px 0 #dfdfdf',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
