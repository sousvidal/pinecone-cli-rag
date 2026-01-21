import {
  generateDenseQueryEmbedding,
  generateSparseQueryEmbedding,
} from "./embeddings.js";
import { searchDenseIndex, searchSparseIndex } from "./search.js";
import { reciprocalRankFusion, rerank } from "./rerank.js";
import type { HybridSearchOptions, SearchResult } from "./types.js";

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
