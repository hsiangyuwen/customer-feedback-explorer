import React, { useState } from 'react';
import { Bot, User, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';

// Define types for clarity, assuming they might be shared or expanded later
export interface FeedbackItem {
  id: number | string;
  original_id?: string | null;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  relevantFeedback?: FeedbackItem[];
  isError?: boolean;
}

interface ChatMessageItemProps {
  message: ChatMessage;
}

const ChatMessageItem: React.FC<ChatMessageItemProps> = ({ message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-xl lg:max-w-2xl px-5 py-3 rounded-2xl shadow-md ${
          isUser ? 'bg-sky-600 text-white rounded-br-none' : `bg-slate-700 text-gray-200 rounded-bl-none ${message.isError ? 'border border-red-500' : ''}`
        }`}
      >
        <div className="flex items-center mb-1">
          {isUser ? (
            <User size={18} className="mr-2 text-sky-300 flex-shrink-0" />
          ) : (
             message.isError 
             ? <AlertCircle size={18} className="mr-2 text-red-400 flex-shrink-0" /> 
             : <Bot size={18} className="mr-2 text-teal-400 flex-shrink-0" />
          )}
          <span className="font-semibold text-sm">{isUser ? 'You' : 'AI Assistant'}</span>
        </div>
        <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
        {message.relevantFeedback && message.relevantFeedback.length > 0 && (
          <div className="mt-3 border-t border-slate-600 pt-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs text-sky-300 hover:text-sky-200 flex items-center font-medium focus:outline-none"
              aria-expanded={isExpanded}
            >
              {isExpanded ? 'Hide Relevant Feedback' : 'Show Relevant Feedback'}
              {isExpanded ? <ChevronUp size={14} className="ml-1" /> : <ChevronDown size={14} className="ml-1" />}
            </button>
            {isExpanded && (
              <div className="mt-2 space-y-2 max-h-48 overflow-y-auto pr-1 custom-scrollbar"> {/* Added custom-scrollbar */}
                {message.relevantFeedback.map((item) => (
                  <div key={item.id} className="p-2 bg-slate-600 rounded-md">
                    <p className="text-xs text-gray-300">
                      <span className="font-semibold text-sky-400">
                        ID {item.original_id || item.id}:
                      </span>
                      {' '}{item.text}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessageItem;