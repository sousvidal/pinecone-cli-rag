import { Pinecone } from "@pinecone-database/pinecone";
import { getConfig } from "../config.js";
import type { Chunk } from "../utils/chunker.js";

// Pinecone client singleton
let _client: Pinecone | null = null;

function getClient(): Pinecone {
  if (!_client) {
    const config = getConfig();
    _client = new Pinecone({ apiKey: config.pineconeApiKey });
  }
  return _client;
}

/**
 * Embedding models to use
 */
const DENSE_MODEL = "llama-text-embed-v2";
const SPARSE_MODEL = "pinecone-sparse-english-v0";

/**
 * Default reranking model
 */
const RERANK_MODEL = "bge-reranker-v2-m3";

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
 * Generate dense embeddings using Pinecone's inference API
 */
export async function generateDenseEmbeddings(
  texts: string[]
): Promise<DenseEmbedding[]> {
  if (texts.length === 0) return [];

  const client = getClient();

  // Process in batches of 96 (Pinecone's max batch size)
  const batchSize = 96;
  const allEmbeddings: DenseEmbedding[] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    const response = await client.inference.embed(DENSE_MODEL, batch, {
      inputType: "passage",
      truncate: "END",
    });

    for (const embedding of response.data) {
      // Dense embeddings have a 'values' property
      const embeddingAny = embedding as { values?: number[] };
      if (embeddingAny.values) {
        allEmbeddings.push({ values: embeddingAny.values });
      }
    }
  }

  return allEmbeddings;
}

/**
 * Generate sparse embeddings using Pinecone's inference API
 */
export async function generateSparseEmbeddings(
  texts: string[]
): Promise<SparseEmbedding[]> {
  if (texts.length === 0) return [];

  const client = getClient();

  // Process in batches of 96
  const batchSize = 96;
  const allEmbeddings: SparseEmbedding[] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    const response = await client.inference.embed(SPARSE_MODEL, batch, {
      inputType: "passage",
      truncate: "END",
    });

    for (const embedding of response.data) {
      // Sparse embeddings have 'sparseIndices' and 'sparseValues' properties (camelCase from SDK)
      const embeddingAny = embedding as {
        sparseIndices?: number[];
        sparseValues?: number[];
      };
      if (embeddingAny.sparseIndices && embeddingAny.sparseValues) {
        allEmbeddings.push({
          indices: embeddingAny.sparseIndices,
          values: embeddingAny.sparseValues,
        });
      } else {
        // Push empty embedding to maintain array length consistency
        allEmbeddings.push({
          indices: [],
          values: [],
        });
      }
    }
  }

  return allEmbeddings;
}

/**
 * Generate a single dense embedding for a query using Pinecone's inference API.
 * Uses inputType: "query" for optimal search performance.
 */
export async function generateDenseQueryEmbedding(
  text: string
): Promise<DenseEmbedding> {
  const client = getClient();

  const response = await client.inference.embed(DENSE_MODEL, [text], {
    inputType: "query",
    truncate: "END",
  });

  const embedding = response.data[0] as { values?: number[] };
  if (!embedding.values) {
    throw new Error("Failed to generate dense query embedding");
  }

  return { values: embedding.values };
}

/**
 * Generate a single sparse embedding for a query using Pinecone's inference API.
 * Uses inputType: "query" for optimal search performance.
 */
export async function generateSparseQueryEmbedding(
  text: string
): Promise<SparseEmbedding> {
  const client = getClient();

  const response = await client.inference.embed(SPARSE_MODEL, [text], {
    inputType: "query",
    truncate: "END",
  });

  const embedding = response.data[0] as {
    sparseIndices?: number[];
    sparseValues?: number[];
  };

  if (!embedding.sparseIndices || !embedding.sparseValues) {
    throw new Error("Failed to generate sparse query embedding");
  }

  return {
    indices: embedding.sparseIndices,
    values: embedding.sparseValues,
  };
}

/**
 * Upsert records to the dense index
 */
export async function upsertToDenseIndex(
  records: DenseRecord[],
  namespace?: string
): Promise<void> {
  if (records.length === 0) return;

  const config = getConfig();
  const client = getClient();
  const index = client.index(config.pineconeDenseIndex);

  // Upsert in batches of 100
  const batchSize = 100;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    if (namespace) {
      await index.namespace(namespace).upsert(batch);
    } else {
      await index.upsert(batch);
    }
  }
}

/**
 * Upsert records to the sparse index
 */
export async function upsertToSparseIndex(
  records: SparseRecord[],
  namespace?: string
): Promise<void> {
  if (records.length === 0) return;

  const config = getConfig();
  const client = getClient();
  const index = client.index(config.pineconeSparseIndex);

  // Filter out records with empty sparse values (defensive validation)
  const validRecords = records.filter(
    (r) => r.sparseValues.values.length > 0 && r.sparseValues.indices.length > 0
  );

  if (validRecords.length === 0) {
    console.warn("No valid sparse vectors to upsert (all vectors were empty)");
    return;
  }

  // Transform to the format expected by Pinecone sparse index
  // Sparse indexes only need id, sparseValues, and metadata (no dense values)
  const pineconeRecords = validRecords.map((r) => ({
    id: r.id,
    sparseValues: r.sparseValues,
    metadata: r.metadata,
  }));

  // Upsert in batches of 100
  const batchSize = 100;
  for (let i = 0; i < pineconeRecords.length; i += batchSize) {
    const batch = pineconeRecords.slice(i, i + batchSize);
    if (namespace) {
      await index.namespace(namespace).upsert(batch);
    } else {
      await index.upsert(batch);
    }
  }
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
 * Create records from chunks for both dense and sparse indexes.
 * Both indexes will have records with the same IDs for easy merging during search.
 */
export function createRecordsFromChunks(
  chunks: Chunk[],
  sourceFile: string,
  denseEmbeddings: DenseEmbedding[],
  sparseEmbeddings: SparseEmbedding[]
): { denseRecords: DenseRecord[]; sparseRecords: SparseRecord[] } {
  if (
    chunks.length !== denseEmbeddings.length ||
    chunks.length !== sparseEmbeddings.length
  ) {
    throw new Error("Chunks and embeddings arrays must have the same length");
  }

  const denseRecords: DenseRecord[] = [];
  const sparseRecords: SparseRecord[] = [];

  // Create a simple hash of the filename for ID prefix
  let hash = 0;
  for (let i = 0; i < sourceFile.length; i++) {
    const char = sourceFile.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  const fileHash = Math.abs(hash).toString(36);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const id = `${fileHash}-${i.toString().padStart(4, "0")}`;

    const metadata: ChunkMetadata = {
      source: sourceFile,
      chunk_index: i,
      chunk_text: chunk.text.slice(0, 40000), // Pinecone metadata limit
      headings: chunk.headings,
      item_types: chunk.itemTypes,
      total_chunks: chunks.length,
    };

    denseRecords.push({
      id,
      values: denseEmbeddings[i].values,
      metadata,
    });

    // Only create sparse record if the embedding has values
    if (sparseEmbeddings[i].values.length > 0) {
      sparseRecords.push({
        id,
        sparseValues: sparseEmbeddings[i],
        metadata,
      });
    }
  }

  return { denseRecords, sparseRecords };
}

/**
 * Index chunks to both dense and sparse Pinecone indexes.
 * Note: Sparse count may be less than dense count if some chunks produce empty sparse embeddings.
 */
export async function indexChunks(
  chunks: Chunk[],
  sourceFile: string,
  namespace?: string,
  onProgress?: (stage: string, progress: number) => void
): Promise<{ denseCount: number; sparseCount: number }> {
  if (chunks.length === 0) {
    return { denseCount: 0, sparseCount: 0 };
  }

  const texts = chunks.map((c) => c.text);

  // Generate embeddings
  onProgress?.("Generating dense embeddings", 0);
  const denseEmbeddings = await generateDenseEmbeddings(texts);
  onProgress?.("Generating dense embeddings", 100);

  onProgress?.("Generating sparse embeddings", 0);
  const sparseEmbeddings = await generateSparseEmbeddings(texts);
  onProgress?.("Generating sparse embeddings", 100);

  // Create records
  const { denseRecords, sparseRecords } = createRecordsFromChunks(
    chunks,
    sourceFile,
    denseEmbeddings,
    sparseEmbeddings
  );

  // Upsert to indexes
  onProgress?.("Upserting to dense index", 0);
  await upsertToDenseIndex(denseRecords, namespace);
  onProgress?.("Upserting to dense index", 100);

  onProgress?.("Upserting to sparse index", 0);
  await upsertToSparseIndex(sparseRecords, namespace);
  onProgress?.("Upserting to sparse index", 100);

  return {
    denseCount: denseRecords.length,
    sparseCount: sparseRecords.length,
  };
}

/**
 * Clear all vectors from both dense and sparse indexes
 */
export async function clearAllIndexes(namespace?: string): Promise<void> {
  const config = getConfig();
  const client = getClient();

  try {
    const denseIndex = client.index(config.pineconeDenseIndex);
    if (namespace) {
      await denseIndex.namespace(namespace).deleteAll();
    } else {
      await denseIndex.deleteAll();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`Failed to clear dense index: ${message}`);
  }

  try {
    const sparseIndex = client.index(config.pineconeSparseIndex);
    if (namespace) {
      await sparseIndex.namespace(namespace).deleteAll();
    } else {
      await sparseIndex.deleteAll();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`Failed to clear sparse index: ${message}`);
  }
}

/**
 * Get index statistics
 */
export async function getIndexStats(): Promise<{
  dense: { vectorCount: number; dimension: number } | null;
  sparse: { vectorCount: number } | null;
}> {
  const config = getConfig();
  const client = getClient();

  let dense = null;
  let sparse = null;

  try {
    const denseIndex = client.index(config.pineconeDenseIndex);
    const denseStats = await denseIndex.describeIndexStats();
    dense = {
      vectorCount: denseStats.totalRecordCount || 0,
      dimension: denseStats.dimension || 0,
    };
  } catch {
    // Index might not exist
  }

  try {
    const sparseIndex = client.index(config.pineconeSparseIndex);
    const sparseStats = await sparseIndex.describeIndexStats();
    sparse = {
      vectorCount: sparseStats.totalRecordCount || 0,
    };
  } catch {
    // Index might not exist
  }

  return { dense, sparse };
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
 * Search the dense index
 */
export async function searchDenseIndex(
  embedding: DenseEmbedding,
  topK: number,
  namespace?: string
): Promise<SearchResult[]> {
  const config = getConfig();
  const client = getClient();
  const index = client.index(config.pineconeDenseIndex);

  const queryOptions = {
    vector: embedding.values,
    topK,
    includeMetadata: true,
  };

  const response = namespace
    ? await index.namespace(namespace).query(queryOptions)
    : await index.query(queryOptions);

  return (response.matches || []).map((match) => ({
    id: match.id,
    score: match.score || 0,
    metadata: match.metadata as ChunkMetadata,
    source: "dense" as const,
  }));
}

/**
 * Search the sparse index.
 * Note: The Pinecone API supports sparse-only queries for sparse indexes,
 * but the SDK types don't include this case, so we use a type assertion.
 */
export async function searchSparseIndex(
  embedding: SparseEmbedding,
  topK: number,
  namespace?: string
): Promise<SearchResult[]> {
  const config = getConfig();
  const client = getClient();
  const index = client.index(config.pineconeSparseIndex);

  // Sparse-only query - the Pinecone API supports this for sparse indexes,
  // even though the SDK types require a vector field. We use unknown to
  // work around this TypeScript limitation.
  const queryOptions = {
    sparseVector: {
      indices: embedding.indices,
      values: embedding.values,
    },
    topK,
    includeMetadata: true,
  } as unknown as Parameters<typeof index.query>[0];

  const response = namespace
    ? await index.namespace(namespace).query(queryOptions)
    : await index.query(queryOptions);

  return (response.matches || []).map((match) => ({
    id: match.id,
    score: match.score || 0,
    metadata: match.metadata as ChunkMetadata,
    source: "sparse" as const,
  }));
}

/**
 * Reciprocal Rank Fusion (RRF) to combine results from dense and sparse searches.
 * RRF score = sum(1 / (k + rank)) for each result list where the item appears.
 * This is scale-agnostic and handles different score ranges well.
 *
 * @param denseResults - Results from dense index search
 * @param sparseResults - Results from sparse index search
 * @param k - Constant to prevent division by zero and control ranking smoothness (default: 60)
 * @returns Merged and re-ranked results
 */
export function reciprocalRankFusion(
  denseResults: SearchResult[],
  sparseResults: SearchResult[],
  k: number = 60
): SearchResult[] {
  // Map to accumulate RRF scores and track metadata
  const scoreMap = new Map<
    string,
    { score: number; metadata: ChunkMetadata; sources: Set<string> }
  >();

  // Process dense results
  denseResults.forEach((result, rank) => {
    const rrfScore = 1 / (k + rank + 1); // rank is 0-indexed, so add 1
    const existing = scoreMap.get(result.id);
    if (existing) {
      existing.score += rrfScore;
      existing.sources.add("dense");
    } else {
      scoreMap.set(result.id, {
        score: rrfScore,
        metadata: result.metadata,
        sources: new Set(["dense"]),
      });
    }
  });

  // Process sparse results
  sparseResults.forEach((result, rank) => {
    const rrfScore = 1 / (k + rank + 1);
    const existing = scoreMap.get(result.id);
    if (existing) {
      existing.score += rrfScore;
      existing.sources.add("sparse");
    } else {
      scoreMap.set(result.id, {
        score: rrfScore,
        metadata: result.metadata,
        sources: new Set(["sparse"]),
      });
    }
  });

  // Convert to array and sort by RRF score descending
  const merged: SearchResult[] = Array.from(scoreMap.entries())
    .map(([id, data]) => ({
      id,
      score: data.score,
      metadata: data.metadata,
      source: (data.sources.size === 2
        ? "hybrid"
        : data.sources.has("dense")
          ? "dense"
          : "sparse") as "dense" | "sparse" | "hybrid",
    }))
    .sort((a, b) => b.score - a.score);

  return merged;
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
 * Reranked result from Pinecone inference API
 */
interface RerankResultItem {
  index: number;
  score: number;
  document?: {
    id?: string;
    [key: string]: unknown;
  };
}

/**
 * Rerank search results using Pinecone's inference API.
 * Uses a cross-encoder model to score documents based on their semantic relevance to the query.
 *
 * @param query - The search query text
 * @param results - Search results to rerank (must have chunk_text in metadata)
 * @param options - Reranking options
 * @returns Reranked search results with updated scores (normalized 0-1)
 */
export async function rerank(
  query: string,
  results: SearchResult[],
  options: RerankOptions = {}
): Promise<SearchResult[]> {
  if (results.length === 0) return [];

  const {
    model = RERANK_MODEL,
    topN = results.length,
    rankFields = ["chunk_text"],
  } = options;

  const client = getClient();

  // Prepare documents for reranking - use chunk_text from metadata
  const documents = results.map((result) => ({
    id: result.id,
    chunk_text: result.metadata.chunk_text,
  }));

  // Call Pinecone's rerank API
  const response = await client.inference.rerank(model, query, documents, {
    topN,
    rankFields,
    returnDocuments: false, // We already have the metadata
    parameters: {
      truncate: "END",
    },
  });

  // Build reranked results using the rerank scores
  const rerankedResults: SearchResult[] = [];
  for (const item of response.data as RerankResultItem[]) {
    const originalResult = results[item.index];
    if (originalResult) {
      rerankedResults.push({
        ...originalResult,
        score: item.score,
        source: "hybrid", // Mark as hybrid since it went through reranking
      });
    }
  }

  return rerankedResults;
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
 * Perform hybrid search combining dense (semantic) and sparse (lexical) search.
 * Uses Reciprocal Rank Fusion to merge results from both indexes.
 * By default, results are reranked using Pinecone's inference API for improved relevance.
 *
 * @param query - The search query text
 * @param options - Search options
 * @returns Merged and ranked search results
 */
export async function hybridSearch(
  query: string,
  options: HybridSearchOptions = {}
): Promise<SearchResult[]> {
  const {
    topK = 10,
    namespace,
    rrfK = 60,
    rerank: shouldRerank = true,
    rerankModel,
    rerankTopN,
  } = options;

  // We fetch more results from each index to have better fusion results
  // After RRF (and optionally reranking), we'll return only topK results
  const fetchK = Math.max(topK * 2, 20);

  // Generate embeddings for the query (in parallel)
  const [denseEmbedding, sparseEmbedding] = await Promise.all([
    generateDenseQueryEmbedding(query),
    generateSparseQueryEmbedding(query),
  ]);

  // Search both indexes in parallel
  const [denseResults, sparseResults] = await Promise.all([
    searchDenseIndex(denseEmbedding, fetchK, namespace),
    searchSparseIndex(sparseEmbedding, fetchK, namespace),
  ]);

  // Merge results using Reciprocal Rank Fusion
  const mergedResults = reciprocalRankFusion(denseResults, sparseResults, rrfK);

  // If reranking is disabled, return RRF results directly
  if (!shouldRerank) {
    return mergedResults.slice(0, topK);
  }

  // Rerank the merged results for improved relevance
  // Pass more candidates to the reranker than we need, then take topN
  const candidatesForRerank = mergedResults.slice(
    0,
    Math.max(topK * 2, fetchK)
  );

  const rerankedResults = await rerank(query, candidatesForRerank, {
    model: rerankModel,
    topN: rerankTopN ?? topK,
  });

  return rerankedResults;
}
