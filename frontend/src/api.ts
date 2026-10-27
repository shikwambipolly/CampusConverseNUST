import { ChatResponse, Citation, HealthStatus } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export async function sendChatMessage(
  message: string,
  userId: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, user_id: userId }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function resetChatSession(userId: string): Promise<void> {
  await fetch(`${API_BASE}/api/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId }),
  });
}

export async function fetchHealthStatus(): Promise<HealthStatus | null> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Extracts [Source: <doc> | Section: <sec> | Page: <p>] tags from assistant replies
 * and returns both the cleaned display text and deduplicated Citation objects.
 */
export function extractCitations(rawText: string): {
  cleanText: string;
  citations: Citation[];
} {
  const citationRegex =
    /\[Source:\s*([^|\]]+?)\s*\|\s*Section:\s*([^|\]]+?)\s*\|\s*Page:\s*([^\]]+?)\]/gi;
  const citations: Citation[] = [];
  const seen = new Set<string>();

  let match: RegExpExecArray | null;
  while ((match = citationRegex.exec(rawText)) !== null) {
    const source = match[1].trim();
    const section = match[2].trim();
    const page = match[3].trim();
    const key = `${source}|${section}|${page}`;
    if (!seen.has(key)) {
      seen.add(key);
      citations.push({ source, section, page });
    }
  }

  const cleanText = rawText
    .replace(citationRegex, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return { cleanText, citations };
}
