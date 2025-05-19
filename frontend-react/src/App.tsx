import React, { useState, useEffect } from 'react';
import DataIngestionComponent from './components/DataIngestionComponent';
import ChatInterfaceComponent from './components/ChatInterfaceComponent';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';

// Access Vite environment variables for API base URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'; // Fallback

const App: React.FC = () => {
  const [ingestionStatus, setIngestionStatus] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  
  // Auto-clear ingestion status message after a delay
  useEffect(() => {
    if (ingestionStatus) {
      const timer = setTimeout(() => setIngestionStatus(null), 7000); // Clear status after 7 seconds
      return () => clearTimeout(timer); // Cleanup timer on component unmount or if status changes
    }
  }, [ingestionStatus]);

  const handleIngestionStatus = (type: 'success' | 'error' | 'info', message: string) => {
    setIngestionStatus({ type, message });
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-4 sm:p-6 lg:p-8 selection:bg-sky-500 selection:text-white">
      <div className="w-full max-w-3xl">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold text-sky-500">AI Customer Feedback Explorer</h1>
          <p className="text-gray-400 mt-2">Upload CSV feedback data and chat with AI to get insights.</p>
        </header>

        {ingestionStatus && (
          <div
            role="alert"
            className={`p-4 mb-6 rounded-lg text-sm flex items-center shadow-lg transition-all duration-500 ease-in-out transform ${ingestionStatus ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}
              ${ingestionStatus.type === 'success' ? 'bg-green-600 text-white' : ''}
              ${ingestionStatus.type === 'error' ? 'bg-red-600 text-white' : ''}
              ${ingestionStatus.type === 'info' ? 'bg-blue-600 text-white' : ''}
            `}
          >
            {ingestionStatus.type === 'success' && <CheckCircle2 size={20} className="mr-3 flex-shrink-0" />}
            {ingestionStatus.type === 'error' && <AlertCircle size={20} className="mr-3 flex-shrink-0" />}
            {ingestionStatus.type === 'info' && <Loader2 size={20} className="mr-3 flex-shrink-0 animate-spin" />}
            <span className="flex-grow">{ingestionStatus.message}</span>
            <button 
                onClick={() => setIngestionStatus(null)} 
                className="ml-4 p-1 rounded-full hover:bg-black/20 focus:outline-none"
                aria-label="Dismiss message"
            >
                &times;
            </button>
          </div>
        )}

        <DataIngestionComponent apiBaseUrl={API_BASE_URL} onIngestionStatus={handleIngestionStatus} />
        
        <ChatInterfaceComponent apiBaseUrl={API_BASE_URL} />

        <footer className="text-center mt-12 py-4 text-sm text-gray-500">
            Feedback Explorer v1.0.0 &copy; {new Date().getFullYear()}
        </footer>
      </div>
    </div>
  );
};

export default App;