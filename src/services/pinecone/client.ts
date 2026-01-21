import { Pinecone } from "@pinecone-database/pinecone";
import { getConfig } from "../../config.js";

/**
 * Pinecone client singleton
 */
let _client: Pinecone | null = null;

export function getClient(): Pinecone {
  if (!_client) {
    const config = getConfig();
    _client = new Pinecone({ apiKey: config.pineconeApiKey });
  }
  return _client;
}

/**
 * Embedding models to use
 */
export const DENSE_MODEL = "llama-text-embed-v2";
export const SPARSE_MODEL = "pinecone-sparse-english-v0";

/**
 * Default reranking model
 */
export const RERANK_MODEL = "bge-reranker-v2-m3";
