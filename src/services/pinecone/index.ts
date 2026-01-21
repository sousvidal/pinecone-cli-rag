/**
 * Pinecone service - main entry point
 * 
 * This module provides a complete RAG solution using Pinecone's
 * dense and sparse indexes with hybrid search capabilities.
 */

// Re-export all types
export type {
  DenseEmbedding,
  SparseEmbedding,
  DenseRecord,
  SparseRecord,
  ChunkMetadata,
  SearchResult,
  RerankOptions,
  HybridSearchOptions,
} from "./types.js";

// Re-export client and constants
export { getClient, DENSE_MODEL, SPARSE_MODEL, RERANK_MODEL } from "./client.js";

// Re-export embedding functions
export {
  generateDenseEmbeddings,
  generateSparseEmbeddings,
  generateDenseQueryEmbedding,
  generateSparseQueryEmbedding,
} from "./embeddings.js";

// Re-export indexing functions
export {
  upsertToDenseIndex,
  upsertToSparseIndex,
  createRecordsFromChunks,
  indexChunks,
  clearAllIndexes,
  getIndexStats,
} from "./indexing.js";

// Re-export search functions
export { searchDenseIndex, searchSparseIndex } from "./search.js";

// Re-export reranking functions
export { reciprocalRankFusion, rerank } from "./rerank.js";

// Re-export hybrid search (main search function)
export { hybridSearch } from "./hybrid.js";
