import { getClient, RERANK_MODEL } from "./client.js";
import type {
  SearchResult,
  RerankOptions,
  RerankResultItem,
} from "./types.js";

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
    { score: number; metadata: SearchResult["metadata"]; sources: Set<string> }
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
