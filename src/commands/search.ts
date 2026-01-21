import { Command } from "commander";
import { hybridSearch, type SearchResult } from "../services/pinecone.js";

interface SearchOptions {
  topK: string;
  namespace?: string;
  verbose?: boolean;
  rerank?: boolean;
  rerankModel?: string;
}

/**
 * Format a search result for display
 */
function formatResult(result: SearchResult, index: number, verbose: boolean): string {
  const lines: string[] = [];
  const scoreStr = result.score.toFixed(4);
  const sourceTag = verbose ? ` [${result.source}]` : "";
  
  lines.push(`${index + 1}. ${result.metadata.source}${sourceTag} (score: ${scoreStr})`);
  
  if (result.metadata.headings && result.metadata.headings.length > 0) {
    lines.push(`   📑 ${result.metadata.headings.join(" > ")}`);
  }
  
  // Show text preview (truncate to ~200 chars)
  const textPreview = result.metadata.chunk_text
    .replace(/\n+/g, " ")
    .slice(0, 200)
    .trim();
  lines.push(`   ${textPreview}${result.metadata.chunk_text.length > 200 ? "..." : ""}`);
  
  return lines.join("\n");
}

export const searchCommand = new Command("search")
  .description("Search indexed documents using hybrid search (semantic + lexical)")
  .argument("<query>", "The search query")
  .option("-k, --top-k <number>", "Number of results to return", "10")
  .option("-n, --namespace <name>", "Pinecone namespace to search in")
  .option("-v, --verbose", "Show additional details (result source: dense/sparse/hybrid)")
  .option("--no-rerank", "Disable reranking (reranking is enabled by default)")
  .option("--rerank-model <model>", "Reranking model to use (default: bge-reranker-v2-m3)")
  .action(async (query: string, options: SearchOptions) => {
    console.log("🔍 Searching documents...\n");
    console.log(`   Query: "${query}"`);
    
    const topK = parseInt(options.topK, 10);
    if (isNaN(topK) || topK < 1) {
      console.error("❌ Invalid --top-k value. Must be a positive integer.");
      process.exit(1);
    }
    
    console.log(`   Top K: ${topK}`);
    if (options.namespace) {
      console.log(`   Namespace: ${options.namespace}`);
    }
    console.log(`   Rerank: ${options.rerank !== false ? "enabled" : "disabled"}`);
    if (options.rerankModel) {
      console.log(`   Rerank model: ${options.rerankModel}`);
    }
    console.log("");

    try {
      const rerankEnabled = options.rerank !== false;
      console.log(`⏳ Generating embeddings and searching indexes${rerankEnabled ? " (with reranking)" : ""}...\n`);
      
      const results = await hybridSearch(query, {
        topK,
        namespace: options.namespace,
        rerank: rerankEnabled,
        rerankModel: options.rerankModel,
      });

      if (results.length === 0) {
        console.log("📭 No results found.\n");
        console.log("   Make sure you have indexed some documents first:");
        console.log("   $ rag index --source ./your-documents");
        return;
      }

      console.log(`📋 Found ${results.length} result(s):\n`);
      console.log("─".repeat(60));
      
      for (let i = 0; i < results.length; i++) {
        console.log(formatResult(results[i], i, options.verbose || false));
        if (i < results.length - 1) {
          console.log("");
        }
      }
      
      console.log("─".repeat(60));
      
      if (options.verbose) {
        // Show source distribution
        const sources = results.reduce(
          (acc, r) => {
            acc[r.source]++;
            return acc;
          },
          { dense: 0, sparse: 0, hybrid: 0 } as Record<string, number>
        );
        
        const parts: string[] = [];
        if (sources.hybrid > 0) parts.push(`${sources.hybrid} hybrid`);
        if (sources.dense > 0) parts.push(`${sources.dense} dense-only`);
        if (sources.sparse > 0) parts.push(`${sources.sparse} sparse-only`);
        
        console.log(`\n📊 Result sources: ${parts.join(", ")}`);
      }
      
      console.log("\n✨ Search complete!");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`\n❌ Search failed: ${message}`);
      
      if (options.verbose && error instanceof Error && error.stack) {
        console.error(`\n${error.stack}`);
      }
      
      process.exit(1);
    }
  });
