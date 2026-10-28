import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  RotateCcw,
  Sparkles,
  BookOpen,
  Database,
  Cpu,
  Globe,
  ShieldCheck,
  Clock,
  FileText,
  Layers,
  ChevronRight,
  Menu,
  X,
  Sun,
  Moon,
} from 'lucide-react';
import { ChatMessage, HealthStatus, SamplePrompt } from './types';
import {
  sendChatMessage,
  resetChatSession,
  fetchHealthStatus,
  extractCitations,
} from './api';

const NUST_LOGO_URL =
  'https://images.seeklogo.com/logo-png/36/1/namibia-university-of-science-and-technology-logo-png_seeklogo-360137.png';

const SAMPLE_PROMPTS: SamplePrompt[] = [
  {
    category: 'Fees & Finance',
    title: 'Registration & Hostel Fees',
    query: 'How much is the registration fee and hostel deposit at NUST?',
  },
  {
    category: 'Course Prerequisites',
    title: 'Module Prerequisite Lookup',
    query: 'What are the prerequisites and NQF credits for PRG510S?',
  },
  {
    category: 'Admissions & Programmes',
    title: 'Admissions + Tuition Combined',
    query:
      'What are the undergraduate admission requirements and how much is the registration fee?',
  },
  {
    category: 'Academic Calendar',
    title: 'Semester & Exam Dates',
    query: 'When does the first semester start and when are examinations?',
  },
  {
    category: 'Academic Regulations',
    title: 'Attendance & Pass Mark Rules',
    query:
      'What is the minimum lecture attendance percentage required to write exams at NUST?',
  },
  {
    category: 'Multilingual (Oshiwambo & Afrikaans)',
    title: 'Waapenduka / Goeie môre Greeting',
    query:
      'Waapenduka! Ongeipi? Tell me about NUST computer science programmes.',
  },
];

const LANGUAGE_LABELS: Record<string, string> = {
  en: 'English (EN)',
  kj: 'Oshiwambo (KJ)',
  af: 'Afrikaans (AF)',
  de: 'German (DE)',
};

function formatMessageText(text: string): React.ReactNode {
  const lines = text.split('\n');
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return <div key={idx} className="msg-spacer" />;
    }

    const isBullet = /^[-•*]\s+/.test(trimmed);
    const cleanLine = isBullet ? trimmed.replace(/^[-•*]\s+/, '') : line;

    // Parse **bold** and `code` inline segments
    const parts = cleanLine.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    const renderedParts = parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={pIdx}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={pIdx} className="inline-code">
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });

    if (isBullet) {
      return (
        <div key={idx} className="msg-bullet-item">
          <span className="bullet-dot">•</span>
          <span>{renderedParts}</span>
        </div>
      );
    }

    return (
      <p key={idx} className="msg-paragraph">
        {renderedParts}
      </p>
    );
  });
}

export default function App() {
  const [userId] = useState<string>(() => {
    const saved = sessionStorage.getItem('nust_chat_user_id');
    if (saved) return saved;
    const generated = `student_${Math.random().toString(36).slice(2, 9)}`;
    sessionStorage.setItem('nust_chat_user_id', generated);
    return generated;
  });

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('nust_chat_theme');
    return saved === 'dark' ? 'dark' : 'light';
  });

  useEffect(() => {
    localStorage.setItem('nust_chat_theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-0',
      role: 'assistant',
      content:
        'Welcome to **CampusConverse**, the official AI academic assistant for the **Namibia University of Science and Technology (NUST)**.\n\n' +
        'I can help you with:\n' +
        '- **Admissions & Programmes** (NQF levels, minimum points, faculties)\n' +
        '- **Tuition, Hostel & Banking Details** (verified fee tables & discounts)\n' +
        '- **Module Prerequisites** (e.g., `PRG510S`, `DSA611S`, `CSA511S`)\n' +
        '- **Academic Calendar & Exam Regulations** (grounded with official PDF citations)\n\n' +
        'Feel free to greet me in **English**, **Oshiwambo**, **Afrikaans**, or **German**!',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
      status: 'ready',
      intent: 'system_welcome',
      confidence: 1.0,
      language: 'en',
      citations: [],
    },
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealthStatus().then(setHealth);
    const interval = setInterval(() => {
      fetchHealthStatus().then(setHealth);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async (overrideQuery?: string) => {
    const queryText = (overrideQuery ?? input).trim();
    if (!queryText || isLoading) return;

    if (!overrideQuery) {
      setInput('');
    }
    setSidebarOpen(false);

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: queryText,
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    const startTime = performance.now();

    try {
      const response = await sendChatMessage(queryText, userId);
      const elapsedMs = Math.round(performance.now() - startTime);
      const { cleanText, citations } = extractCitations(response.reply);

      const botMsg: ChatMessage = {
        id: `bot-${Date.now()}`,
        role: 'assistant',
        content: cleanText || response.reply,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
        status: response.status,
        intent: response.intent,
        confidence: response.confidence,
        language: response.language,
        citations,
        latencyMs: elapsedMs,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content:
          `Unable to reach the CampusConverse backend service. ` +
          `Please ensure the backend container is running.\n\n` +
          `Details: \`${err?.message || 'Connection error'}\``,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
        status: 'error',
        intent: 'connection_error',
        confidence: 0,
        language: 'en',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    await resetChatSession(userId);
    setMessages((prev) => [prev[0]]);
    setSidebarOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <div className="app-shell" data-theme={theme}>
      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Left Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand-badge">
            <img
              src={NUST_LOGO_URL}
              alt="NUST Logo"
              className="brand-logo-img"
            />
          </div>
          <div className="brand-titles">
            <h1>CampusConverse</h1>
            <span>NUST AI Academic Assistant</span>
          </div>
          <button
            className="mobile-close-btn"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>

        {/* Live Architecture Status Card */}
        <div className="stack-card">
          <div className="stack-card-header">
            <span className="stack-title">
              <Layers size={14} /> Active Architecture
            </span>
            <span
              className={`status-dot ${health?.status === 'healthy' ? 'online' : 'offline'}`}
            >
              {health?.status === 'healthy' ? 'Online' : 'Connecting...'}
            </span>
          </div>
          <div className="stack-grid">
            <div className="stack-item">
              <Cpu size={13} />
              <div>
                <small>LLM Synthesis</small>
                <strong>
                  {health?.llm_provider
                    ? `${health.llm_provider} (${health.llm_model})`
                    : 'Hybrid Deterministic'}
                </strong>
              </div>
            </div>
            <div className="stack-item">
              <Database size={13} />
              <div>
                <small>Structured SQL</small>
                <strong>
                  {health?.database === 'postgresql'
                    ? 'PostgreSQL / AlloyDB'
                    : 'SQLite (nust_facts.db)'}
                </strong>
              </div>
            </div>
            <div className="stack-item">
              <Sparkles size={13} />
              <div>
                <small>Hybrid Retrieval</small>
                <strong>ChromaDB + BM25 + TF-IDF</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Prompts */}
        <div className="prompts-section">
          <div className="prompts-header">
            <BookOpen size={14} />
            <span>Try Official NUST Queries</span>
          </div>
          <div className="prompts-list">
            {SAMPLE_PROMPTS.map((item, idx) => (
              <button
                key={idx}
                className="prompt-card"
                onClick={() => handleSend(item.query)}
                disabled={isLoading}
              >
                <div className="prompt-card-top">
                  <span className="prompt-category">{item.category}</span>
                </div>
                <div className="prompt-title">
                  <span>{item.title}</span>
                  <ChevronRight size={14} />
                </div>
                <p className="prompt-preview">{item.query}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Reset Conversation Footer */}
        <div className="sidebar-footer">
          <button
            className="reset-btn"
            onClick={handleReset}
            disabled={isLoading}
          >
            <RotateCcw size={15} />
            <span>Reset Conversation Memory</span>
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Top Bar */}
        <header className="chat-topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={20} />
            </button>
            <div>
              <h2>Namibia University of Science and Technology</h2>
              <p>
                Official AI Student Support • Admissions, Fees, Prerequisites &
                Academic Policies
              </p>
            </div>
          </div>
          <div className="topbar-right">
            <span className="guardrail-pill">
              <ShieldCheck size={14} /> Safety Guardrails Active
            </span>
            <button
              className="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <>
                  <Moon size={16} />
                  <span>Dark</span>
                </>
              ) : (
                <>
                  <Sun size={16} />
                  <span>Light</span>
                </>
              )}
            </button>
          </div>
        </header>

        {/* Messages Feed */}
        <section className="messages-container">
          {messages.map((msg) => {
            const intents = msg.intent
              ? msg.intent.split('+').map((i) => i.trim())
              : [];

            return (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'assistant' ? (
                    <img
                      src={NUST_LOGO_URL}
                      alt="NUST"
                      className="avatar-logo-img"
                    />
                  ) : (
                    <span>YOU</span>
                  )}
                </div>

                <div className="message-bubble-wrapper">
                  <div className={`message-bubble ${msg.role}`}>
                    <div className="message-content">
                      {formatMessageText(msg.content)}
                    </div>

                    {/* Extracted Policy Citations */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-box">
                        <div className="citations-title">
                          <FileText size={13} />
                          <span>Verified NUST Source Citations</span>
                        </div>
                        <div className="citations-list">
                          {msg.citations.map((cit, cIdx) => (
                            <div key={cIdx} className="citation-chip">
                              <span className="cit-doc">{cit.source}</span>
                              <span className="cit-sep">•</span>
                              <span className="cit-sec">{cit.section}</span>
                              <span className="cit-sep">•</span>
                              <span className="cit-page">p. {cit.page}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Telemetry Metadata Bar under Assistant Messages */}
                  <div className="message-meta">
                    <span className="meta-time">{msg.timestamp}</span>

                    {msg.role === 'assistant' && msg.id !== 'welcome-0' && (
                      <>
                        {msg.language && (
                          <span className="meta-badge lang-badge">
                            <Globe size={11} />
                            {LANGUAGE_LABELS[msg.language] ||
                              msg.language.toUpperCase()}
                          </span>
                        )}

                        {intents.map((intentName, iIdx) => (
                          <span key={iIdx} className="meta-badge intent-badge">
                            {intentName}
                          </span>
                        ))}

                        {typeof msg.confidence === 'number' &&
                          msg.confidence > 0 && (
                            <span className="meta-badge conf-badge">
                              {Math.round(msg.confidence * 100)}% conf
                            </span>
                          )}

                        {typeof msg.latencyMs === 'number' && (
                          <span className="meta-badge latency-badge">
                            <Clock size={11} />
                            {msg.latencyMs} ms
                          </span>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {isLoading && (
            <div className="message-row assistant">
              <div className="message-avatar">
                <img
                  src={NUST_LOGO_URL}
                  alt="NUST"
                  className="avatar-logo-img"
                />
              </div>
              <div className="message-bubble assistant loading-bubble">
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <span className="loading-text">
                  Searching NUST records & handbooks...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </section>

        {/* Bottom Composer */}
        <footer className="composer-area">
          <div className="composer-box">
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about NUST admissions, fees, module prerequisites (e.g. PRG510S), or exam rules..."
              disabled={isLoading}
            />
            <button
              className="send-btn"
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
          <div className="composer-footnote">
            <span>
              Answers are grounded in official NUST Yearbooks, General Rules,
              and Fee Schedules. Press <strong>Enter</strong> to send.
            </span>
          </div>
        </footer>
      </main>
    </div>
  );
}
