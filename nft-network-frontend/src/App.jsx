import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Layout/Navbar';
import Home from './pages/Home';
import GraphViewer from './pages/GraphViewer';
import BubbleViewer from './pages/BubbleViewer';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/graph" element={<GraphViewer />} />
          <Route path="/bubbles" element={<BubbleViewer />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App
