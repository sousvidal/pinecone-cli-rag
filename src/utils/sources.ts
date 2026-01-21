import type { SearchResult } from "../services/pinecone.js";

/**
 * Grouped source with cited indices that reference this document
 */
export interface GroupedSource {
  indices: number[]; // 1-based indices that were actually cited in the response
  result: SearchResult; // The highest-scoring result for this document
}

/**
 * Filter results by minimum score threshold
 */
export function filterByScore(
  results: SearchResult[],
  minScore: number
): SearchResult[] {
  return results.filter((r) => r.score >= minScore);
}

/**
 * Extract cited source indices from the response text.
 * Matches patterns like [1], [2], [1, 2], etc.
 */
export function extractCitedIndices(responseText: string): Set<number> {
  const cited = new Set<number>();
  // Match [1], [2], [1, 2], [1,2,3], etc.
  const matches = responseText.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g);

  for (const match of matches) {
    const numbers = match[1].split(",").map((s) => parseInt(s.trim(), 10));
    for (const num of numbers) {
      if (!isNaN(num)) {
        cited.add(num);
      }
    }
  }

  return cited;
}

/**
 * Group sources by document_id, keeping only indices that were actually cited in the response.
 * Returns groups ordered by first appearance (highest score), filtering out uncited documents.
 */
export function groupSourcesByDocument(
  results: SearchResult[],
  citedIndices: Set<number>
): GroupedSource[] {
  const groups = new Map<string, GroupedSource>();

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    const key = result.metadata.document_id;
    const index = i + 1; // 1-based index

    // Only include if this index was cited in the response
    if (!citedIndices.has(index)) {
      continue;
    }

    const existing = groups.get(key);
    if (existing) {
      existing.indices.push(index);
    } else {
      groups.set(key, {
        indices: [index],
        result,
      });
    }
  }

  // Sort indices within each group
  for (const group of groups.values()) {
    group.indices.sort((a, b) => a - b);
  }

  return Array.from(groups.values());
}

/**
 * Format source citations for display, showing cited indices that reference each document
 */
export function formatSourceCitations(groups: GroupedSource[]): string {
  const lines = groups.map((group) => {
    const indicesStr = group.indices.join(", ");
    const r = group.result;
    const headings =
      r.metadata.headings && r.metadata.headings.length > 0
        ? ` > ${r.metadata.headings.join(" > ")}`
        : "";
    return `  [${indicesStr}] ${r.metadata.source}${headings} (score: ${r.score.toFixed(2)})`;
  });
  return lines.join("\n");
}

/**
 * Build context chunks from search results for the RAG prompt
 */
export function buildContextChunks(
  results: SearchResult[]
): Array<{ source: string; headings: string[]; text: string }> {
  return results.map((r) => ({
    source: r.metadata.source,
    headings: r.metadata.headings || [],
    text: r.metadata.chunk_text,
  }));
}
