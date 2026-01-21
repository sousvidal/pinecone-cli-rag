/**
 * Type definitions for Pinecone service
 */

/**
 * Dense embedding result
 */
export interface DenseEmbedding {
  values: number[];
}

/**
 * Sparse embedding result
 */
export interface SparseEmbedding {
  indices: number[];
  values: number[];
}

/**
 * Record to upsert to dense index
 */
export interface DenseRecord {
  id: string;
  values: number[];
  metadata: Record<string, string | number | string[] | boolean>;
}

/**
 * Record to upsert to sparse index
 */
export interface SparseRecord {
  id: string;
  sparseValues: {
    indices: number[];
    values: number[];
  };
  metadata: Record<string, string | number | string[] | boolean>;
}

/**
 * Metadata structure for indexed chunks
 */
export interface ChunkMetadata {
  source: string;
  chunk_index: number;
  chunk_text: string;
  headings: string[];
  item_types: string[];
  total_chunks: number;
  [key: string]: string | number | string[] | boolean;
}

/**
 * Search result from Pinecone
 */
export interface SearchResult {
  id: string;
  score: number;
  metadata: ChunkMetadata;
  source: "dense" | "sparse" | "hybrid";
}

/**
 * Options for reranking search results
 */
export interface RerankOptions {
  model?: string;
  topN?: number;
  rankFields?: string[];
}

/**
 * Options for hybrid search
 */
export interface HybridSearchOptions {
  topK?: number;
  namespace?: string;
  rrfK?: number; // RRF constant (default: 60)
  rerank?: boolean; // Enable reranking (default: true)
  rerankModel?: string; // Reranking model (default: bge-reranker-v2-m3)
  rerankTopN?: number; // Number of reranked results (defaults to topK)
}

/**
 * Reranked result from Pinecone inference API (internal)
 */
export interface RerankResultItem {
  index: number;
  score: number;
  document?: {
    id?: string;
    [key: string]: unknown;
  };
}
