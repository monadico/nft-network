import { createContext, useContext, useEffect, useState } from 'react';

const DarkModeContext = createContext();

export function DarkModeProvider({ children }) {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage first, then system preference
    const saved = localStorage.getItem('darkMode');
    if (saved !== null) {
      return JSON.parse(saved);
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Initialize document CSS custom property on mount
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.style.setProperty('--color-scheme', 'dark');
    } else {
      document.documentElement.style.setProperty('--color-scheme', 'light');
    }
  }, []);

  useEffect(() => {
    // Update localStorage when dark mode changes
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
    
    // Update CSS custom property for Tailwind v4
    if (isDarkMode) {
      document.documentElement.style.setProperty('--color-scheme', 'dark');
    } else {
      document.documentElement.style.setProperty('--color-scheme', 'light');
    }
  }, [isDarkMode]);

  // Listen for system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => {
      if (localStorage.getItem('darkMode') === null) {
        setIsDarkMode(e.matches);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  return (
    <DarkModeContext.Provider value={{ isDarkMode, toggleDarkMode }}>
      {children}
    </DarkModeContext.Provider>
  );
}

export function useDarkMode() {
  const context = useContext(DarkModeContext);
  if (context === undefined) {
    throw new Error('useDarkMode must be used within a DarkModeProvider');
  }
  return context;
}
