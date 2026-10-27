export interface Citation {
  source: string;
  section: string;
  page: string;
}

export interface ChatResponse {
  reply: string;
  status: string;
  intent: string | null;
  confidence: number | null;
  language: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  status?: string;
  intent?: string | null;
  confidence?: number | null;
  language?: string | null;
  citations?: Citation[];
  latencyMs?: number;
}

export interface HealthStatus {
  status: string;
  llm_provider?: string;
  llm_model?: string;
  database?: string;
  vector_store?: string;
}

export interface SamplePrompt {
  category: string;
  title: string;
  query: string;
  tag: string;
}
