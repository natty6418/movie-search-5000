import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Button } from './Button';
import { Input } from './Input';

const ChatInterface = ({ onSendMessage, messages, isLoading, statusMessage }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, statusMessage]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !isLoading) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    return (
        <div className="flex flex-col h-[600px] bg-white rounded-lg border border-gray-200 shadow-sm">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="flex items-center justify-center h-full text-gray-400">
                        <div className="text-center">
                            <Bot className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <p>Start a conversation with Movie Search 5000</p>
                            <p className="text-sm mt-1">Ask about movies, get recommendations, or compare films</p>
                        </div>
                    </div>
                )}
                
                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
                    >
                        <div className={`flex items-start max-w-[80%] gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                            {/* Avatar */}
                            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                                msg.role === 'user' 
                                    ? 'bg-blue-500 text-white' 
                                    : 'bg-purple-500 text-white'
                            }`}>
                                {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                            </div>
                            
                            {/* Message Bubble */}
                            <div className={`rounded-lg px-4 py-2 ${
                                msg.role === 'user'
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-gray-100 text-gray-800'
                            }`}>
                                {msg.role === 'user' ? (
                                    <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                                ) : (
                                    <div className="prose prose-sm max-w-none">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    </div>
                                )}
                                
                                {/* Show docs if available */}
                                {msg.docs && msg.docs.length > 0 && (
                                    <div className="mt-2 pt-2 border-t border-gray-300 space-y-1">
                                        <p className="text-xs font-semibold opacity-70">Referenced Movies:</p>
                                        {msg.docs.map((doc, i) => (
                                            <div key={i} className="text-xs opacity-80">
                                                • {doc.title}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                
                {/* Status Message (agent thinking) */}
                {statusMessage && (
                    <div className="flex justify-start animate-pulse">
                        <div className="flex items-start gap-2">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center">
                                <Bot className="w-4 h-4" />
                            </div>
                            <div className="bg-gray-100 rounded-lg px-4 py-2 text-gray-600">
                                {statusMessage}
                            </div>
                        </div>
                    </div>
                )}
                
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 p-4 bg-gray-50">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about movies..."
                        disabled={isLoading}
                        className="flex-1"
                    />
                    <Button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="flex-shrink-0"
                    >
                        <Send className="w-4 h-4" />
                    </Button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;
