import OpenAI from "openai";
import { getConfig } from "../config.js";

// OpenAI client singleton
let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) {
    const config = getConfig();
    _client = new OpenAI({ apiKey: config.openaiApiKey });
  }
  return _client;
}

/**
 * Default models
 */
const REWRITE_MODEL = "gpt-4o-mini";
const DEFAULT_COMPLETION_MODEL = "gpt-4o";

/**
 * Chat message type
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * Options for chat completion
 */
export interface ChatCompletionOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

/**
 * Result from a non-streaming chat completion
 */
export interface ChatCompletionResult {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * Rewrite a user query into optimized search keywords.
 * Uses a fast, cheap model for minimal latency.
 *
 * @param query - The original user question
 * @returns Expanded search keywords
 */
export async function rewriteQuery(query: string): Promise<string> {
  const client = getClient();

  const response = await client.chat.completions.create({
    model: REWRITE_MODEL,
    temperature: 0,
    max_tokens: 200,
    messages: [
      {
        role: "system",
        content: `You are a search query optimizer. Rewrite the user's question as search keywords that will help find relevant documents.

Rules:
- Output ONLY the keywords, no explanations
- Include synonyms and related terms
- Keep it concise (max 15-20 words)
- Preserve important specific terms (names, numbers, technical terms)

Example:
Question: "What's the return policy for electronics?"
Keywords: return policy electronics refund exchange warranty time limit conditions requirements`,
      },
      {
        role: "user",
        content: query,
      },
    ],
  });

  const rewritten = response.choices[0]?.message?.content?.trim();
  if (!rewritten) {
    // Fallback to original query if rewriting fails
    return query;
  }

  return rewritten;
}

/**
 * Stream a chat completion response.
 * Yields content chunks as they arrive.
 *
 * @param messages - The conversation messages
 * @param options - Completion options
 * @yields Content chunks from the response
 */
export async function* streamChatCompletion(
  messages: ChatMessage[],
  options: ChatCompletionOptions = {}
): AsyncGenerator<string, ChatCompletionResult, unknown> {
  const client = getClient();
  const {
    model = DEFAULT_COMPLETION_MODEL,
    temperature = 0.3,
    maxTokens = 2048,
  } = options;

  const stream = await client.chat.completions.create({
    model,
    temperature,
    max_tokens: maxTokens,
    stream: true,
    stream_options: { include_usage: true },
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
  });

  let fullContent = "";
  let usage = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };

  for await (const chunk of stream) {
    // Extract content delta
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) {
      fullContent += delta;
      yield delta;
    }

    // Capture usage from the final chunk
    if (chunk.usage) {
      usage = {
        promptTokens: chunk.usage.prompt_tokens,
        completionTokens: chunk.usage.completion_tokens,
        totalTokens: chunk.usage.total_tokens,
      };
    }
  }

  return {
    content: fullContent,
    usage,
  };
}

/**
 * Get a non-streaming chat completion.
 *
 * @param messages - The conversation messages
 * @param options - Completion options
 * @returns The completion result with content and usage
 */
export async function chatCompletion(
  messages: ChatMessage[],
  options: ChatCompletionOptions = {}
): Promise<ChatCompletionResult> {
  const client = getClient();
  const {
    model = DEFAULT_COMPLETION_MODEL,
    temperature = 0.3,
    maxTokens = 2048,
  } = options;

  const response = await client.chat.completions.create({
    model,
    temperature,
    max_tokens: maxTokens,
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
  });

  const content = response.choices[0]?.message?.content || "";
  const usage = response.usage || {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  };

  return {
    content,
    usage: {
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
    },
  };
}

/**
 * Build the system prompt for RAG with context chunks.
 *
 * @param contextChunks - Array of context objects with source, headings, and text
 * @returns The formatted system prompt
 */
export function buildRAGSystemPrompt(
  contextChunks: Array<{
    source: string;
    headings: string[];
    text: string;
  }>
): string {
  if (contextChunks.length === 0) {
    return `You are a knowledgeable assistant. The user is asking a question but no relevant context was found in the knowledge base.

Please respond with: "I couldn't find any relevant information in the knowledge base to answer your question. Please make sure the relevant documents have been indexed, or try rephrasing your question."`;
  }

  const contextParts = contextChunks.map((chunk, index) => {
    const headingStr =
      chunk.headings.length > 0 ? ` | Section: ${chunk.headings.join(" > ")}` : "";
    return `[${index + 1}] Source: ${chunk.source}${headingStr}
${chunk.text}`;
  });

  return `You are a knowledgeable assistant. Answer questions using ONLY the provided context.

Rules:
- Cite sources using [1], [2], etc. when referencing information from the context
- If the context doesn't contain enough information to fully answer the question, say "Based on the available context, I cannot fully answer this question." and explain what information is missing
- Be concise and direct
- Do not make up information that is not in the context

Context:
${contextParts.join("\n\n")}`;
}
