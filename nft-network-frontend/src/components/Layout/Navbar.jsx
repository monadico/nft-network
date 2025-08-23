import { Link } from 'react-router-dom';
import DarkModeToggle from '../UI/DarkModeToggle';

export default function Navbar() {
  return (
    <nav className="w-full border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-gray-900/60">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          <Link to="/" className="font-semibold tracking-tight text-gray-900 dark:text-white">
            nft-network
          </Link>
          <div className="flex items-center gap-6 text-sm">
            <Link to="/graph" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400">Graph</Link>
            <Link to="/bubbles" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400">Bubble Map</Link>
            <Link to="/about" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400">About</Link>
            <DarkModeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}


