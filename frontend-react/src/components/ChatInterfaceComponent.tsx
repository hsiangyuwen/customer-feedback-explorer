import React, { useState, useEffect, FormEvent, useRef } from 'react';
import { Send, Loader2, Bot } from 'lucide-react';
import ChatMessageItem, { ChatMessage, FeedbackItem } from './ChatMessageItem'; // Import shared types

interface QueryApiResponse {
  relevant_feedback: FeedbackItem[];
  summary: string;
}

interface ChatInterfaceComponentProps {
  apiBaseUrl: string;
}

const ChatInterfaceComponent: React.FC<ChatInterfaceComponentProps> = ({ apiBaseUrl }) => {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);


  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleSendMessage = async (event?: FormEvent<HTMLFormElement>) => {
    if (event) event.preventDefault();
    const query = currentQuery.trim();
    if (!query) return;

    const newUserMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: query,
    };
    setChatMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setCurrentQuery('');
    setIsQuerying(true);

    try {
      const response = await fetch(`${apiBaseUrl}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, top_k: 3 }), // top_k can be dynamic
      });

      let assistantMessage: ChatMessage;
      if (response.ok) {
        const data: QueryApiResponse = await response.json();
        assistantMessage = {
          id: `assistant-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: data.summary,
          relevantFeedback: data.relevant_feedback,
        };
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from query API.' }));
        assistantMessage = {
          id: `assistant-error-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: `Error (Status ${response.status}): ${errorData.detail || response.statusText}`,
          isError: true,
        };
      }
      setChatMessages((prevMessages) => [...prevMessages, assistantMessage]);
    } catch (error) {
      console.error('Query API error:', error);
      const errorMessage: ChatMessage = {
        id: `assistant-error-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: `An error occurred while querying: ${error instanceof Error ? error.message : String(error)}`,
        isError: true,
      };
      setChatMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsQuerying(false);
      inputRef.current?.focus(); // Re-focus input after sending
    }
  };

  return (
    <div className="mt-8 p-6 bg-gray-800 rounded-xl shadow-2xl">
      <h2 className="text-2xl font-semibold mb-6 text-sky-400 flex items-center">
        <Bot size={28} className="mr-3" />
        Chat with AI
      </h2>
      
      <div ref={chatContainerRef} className="h-96 overflow-y-auto mb-4 p-4 bg-gray-700 rounded-lg border border-gray-600 custom-scrollbar">
        {chatMessages.length === 0 && (
          <p className="text-center text-gray-400">
            No messages yet. After ingesting data, ask a question about the feedback!
          </p>
        )}
        {chatMessages.map((msg) => (
          <ChatMessageItem key={msg.id} message={msg} />
        ))}
        {isQuerying && chatMessages[chatMessages.length -1]?.role === 'user' && ( // Show thinking only if last message was user
          <div className="flex justify-start mb-4">
              <div className="max-w-xl px-5 py-3 rounded-2xl shadow-md bg-slate-700 text-gray-200 rounded-bl-none">
                  <div className="flex items-center">
                      <Bot size={18} className="mr-2 text-teal-400 flex-shrink-0" />
                      <span className="font-semibold text-sm">AI Assistant</span>
                  </div>
                  <div className="flex items-center mt-2">
                      <Loader2 size={16} className="animate-spin mr-2 text-gray-400" />
                      <span className="text-sm italic text-gray-400">Thinking...</span>
                  </div>
              </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="flex items-center space-x-3">
        <input
          ref={inputRef}
          type="text"
          value={currentQuery}
          onChange={(e) => setCurrentQuery(e.target.value)}
          placeholder="Ask about feedback..."
          className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none transition-shadow"
          disabled={isQuerying}
          aria-label="Chat input"
        />
        <button
          type="submit"
          disabled={isQuerying || !currentQuery.trim()}
          className="px-4 sm:px-6 py-3 bg-sky-600 hover:bg-sky-700 text-white font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center transition-colors duration-150"
          aria-label={isQuerying ? 'Sending message' : 'Send message'}
        >
          {isQuerying ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          <span className="ml-2 hidden sm:inline">{isQuerying ? 'Sending...' : 'Send'}</span>
        </button>
      </form>
    </div>
  );
};

export default ChatInterfaceComponent;