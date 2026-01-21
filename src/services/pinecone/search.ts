import { getClient } from "./client.js";
import { getConfig } from "../../config.js";
import type {
  DenseEmbedding,
  SparseEmbedding,
  SearchResult,
  ChunkMetadata,
} from "./types.js";

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
