import { Command } from "commander";
import { hybridSearch, type SearchResult } from "../services/pinecone.js";
import {
  rewriteQuery,
  streamChatCompletion,
  chatCompletion,
  buildRAGSystemPrompt,
  type ChatMessage,
  type ChatCompletionResult,
} from "../services/openai.js";

interface QueryOptions {
  topK: string;
  namespace?: string;
  model?: string;
  temperature: string;
  minScore: string;
  rewrite?: boolean;
  stream?: boolean;
  verbose?: boolean;
}

/**
 * Filter results by minimum score threshold
 */
function filterByScore(
  results: SearchResult[],
  minScore: number
): SearchResult[] {
  return results.filter((r) => r.score >= minScore);
}

/**
 * Format source citations for display
 */
function formatSourceCitations(results: SearchResult[]): string {
  const lines = results.map((r, i) => {
    const headings =
      r.metadata.headings && r.metadata.headings.length > 0
        ? ` > ${r.metadata.headings.join(" > ")}`
        : "";
    return `  [${i + 1}] ${r.metadata.source}${headings} (score: ${r.score.toFixed(2)})`;
  });
  return lines.join("\n");
}

/**
 * Build context chunks from search results
 */
function buildContextChunks(
  results: SearchResult[]
): Array<{ source: string; headings: string[]; text: string }> {
  return results.map((r) => ({
    source: r.metadata.source,
    headings: r.metadata.headings || [],
    text: r.metadata.chunk_text,
  }));
}

export const queryCommand = new Command("query")
  .description(
    "Answer a question using RAG (Retrieval-Augmented Generation) with OpenAI"
  )
  .argument("<question>", "The question to answer")
  .option("-k, --top-k <number>", "Maximum number of context chunks", "5")
  .option("-n, --namespace <name>", "Pinecone namespace to search in")
  .option("-m, --model <name>", "OpenAI model to use", "gpt-4o")
  .option(
    "-t, --temperature <number>",
    "Response temperature (0-2, lower = more focused)",
    "0.3"
  )
  .option(
    "--min-score <number>",
    "Minimum relevance score threshold (0-1)",
    "0.3"
  )
  .option("--no-rewrite", "Skip query rewriting optimization")
  .option("--no-stream", "Disable streaming output")
  .option(
    "-v, --verbose",
    "Show rewritten query, context details, and token usage"
  )
  .action(async (question: string, options: QueryOptions) => {
    const topK = parseInt(options.topK, 10);
    const temperature = parseFloat(options.temperature);
    const minScore = parseFloat(options.minScore);

    // Validate options
    if (isNaN(topK) || topK < 1) {
      console.error("❌ Invalid --top-k value. Must be a positive integer.");
      process.exit(1);
    }

    if (isNaN(temperature) || temperature < 0 || temperature > 2) {
      console.error("❌ Invalid --temperature value. Must be between 0 and 2.");
      process.exit(1);
    }

    if (isNaN(minScore) || minScore < 0 || minScore > 1) {
      console.error("❌ Invalid --min-score value. Must be between 0 and 1.");
      process.exit(1);
    }

    const shouldRewrite = options.rewrite !== false;
    const shouldStream = options.stream !== false;

    try {
      // Step 1: Query rewriting (optional)
      let searchQuery = question;
      if (shouldRewrite) {
        if (options.verbose) {
          console.log("🔄 Rewriting query for better search...");
        }
        searchQuery = await rewriteQuery(question);
        if (options.verbose) {
          console.log(`   Original: "${question}"`);
          console.log(`   Rewritten: "${searchQuery}"\n`);
        }
      }

      // Step 2: Hybrid search with reranking
      console.log("🔍 Searching for relevant context...");
      const allResults = await hybridSearch(searchQuery, {
        topK: topK * 2, // Fetch extra to account for filtering
        namespace: options.namespace,
        rerank: true,
      });

      // Step 3: Filter by score threshold
      const filteredResults = filterByScore(allResults, minScore);
      const relevantResults = filteredResults.slice(0, topK);

      if (allResults.length === 0) {
        console.log("\n📭 No results found in the knowledge base.");
        console.log("   Make sure you have indexed documents first:");
        console.log("   $ rag index --source ./your-documents\n");
        return;
      }

      const filterInfo =
        filteredResults.length < allResults.length
          ? ` (filtered from ${allResults.length})`
          : "";
      console.log(
        `✓ Found ${relevantResults.length} relevant chunk(s)${filterInfo}\n`
      );

      if (options.verbose && relevantResults.length > 0) {
        console.log("📑 Context chunks:");
        for (let i = 0; i < relevantResults.length; i++) {
          const r = relevantResults[i];
          const preview = r.metadata.chunk_text
            .replace(/\n+/g, " ")
            .slice(0, 100)
            .trim();
          console.log(
            `   [${i + 1}] ${r.metadata.source} (${r.score.toFixed(2)})`
          );
          console.log(`       ${preview}...`);
        }
        console.log("");
      }

      // Step 4: Build prompt with context
      const contextChunks = buildContextChunks(relevantResults);
      const systemPrompt = buildRAGSystemPrompt(contextChunks);

      const messages: ChatMessage[] = [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ];

      // Step 5: Generate response
      console.log("💬 Generating answer...\n");
      console.log("─".repeat(60));

      let result: ChatCompletionResult;

      if (shouldStream) {
        // Streaming response
        const generator = streamChatCompletion(messages, {
          model: options.model,
          temperature,
        });

        // Process stream chunks
        let iteratorResult = await generator.next();
        while (!iteratorResult.done) {
          process.stdout.write(iteratorResult.value);
          iteratorResult = await generator.next();
        }

        // Get the final result with usage info
        result = iteratorResult.value;
        console.log("\n");
      } else {
        // Non-streaming response
        result = await chatCompletion(messages, {
          model: options.model,
          temperature,
        });
        console.log(result.content);
        console.log("\n");
      }

      console.log("─".repeat(60));

      // Step 6: Show source citations
      if (relevantResults.length > 0) {
        console.log("\n📚 Sources:");
        console.log(formatSourceCitations(relevantResults));
      }

      // Show token usage in verbose mode
      if (options.verbose && result.usage.totalTokens > 0) {
        console.log(
          `\n📊 Token usage: ${result.usage.promptTokens} prompt + ${result.usage.completionTokens} completion = ${result.usage.totalTokens} total`
        );
      }

      console.log("");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);

      // Provide helpful error messages for common issues
      if (message.includes("API key")) {
        console.error("\n❌ OpenAI API key error.");
        console.error("   Make sure OPENAI_API_KEY is set in your .env file.");
      } else if (message.includes("rate limit")) {
        console.error("\n❌ Rate limit exceeded.");
        console.error("   Please wait a moment and try again.");
      } else if (message.includes("model")) {
        console.error(`\n❌ Model error: ${message}`);
        console.error(
          '   Try using a different model with --model (e.g., "gpt-4o-mini").'
        );
      } else {
        console.error(`\n❌ Query failed: ${message}`);
      }

      if (options.verbose && error instanceof Error && error.stack) {
        console.error(`\n${error.stack}`);
      }

      process.exit(1);
    }
  });
