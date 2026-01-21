import { Command } from "commander";
import * as path from "path";
import {
  convertDocument,
  findSupportedFiles,
  SUPPORTED_EXTENSIONS,
} from "../services/docling.js";
import { chunkDocument } from "../utils/chunker.js";
import {
  indexChunks,
  getIndexStats,
  clearAllIndexes,
} from "../services/pinecone.js";

interface IndexOptions {
  source: string;
  namespace?: string;
  verbose?: boolean;
}

/**
 * Format a number with thousand separators
 */
function formatNumber(n: number): string {
  return n.toLocaleString();
}

/**
 * Create a simple progress bar
 */
function progressBar(current: number, total: number, width = 30): string {
  const progress = Math.min(current / total, 1);
  const filled = Math.round(width * progress);
  const empty = width - filled;
  return `[${"█".repeat(filled)}${"░".repeat(empty)}] ${Math.round(progress * 100)}%`;
}

export const indexCommand = new Command("index")
  .description("Index documents from a source directory for RAG")
  .option(
    "-s, --source <path>",
    "Source directory or file containing documents to index",
    "./docs"
  )
  .option(
    "-n, --namespace <name>",
    "Pinecone namespace to use (optional, for organizing data)"
  )
  .option("-v, --verbose", "Show detailed progress information")
  .action(async (options: IndexOptions) => {
    console.log("📂 Starting document indexing...\n");
    console.log(`   Source: ${path.resolve(options.source)}`);
    if (options.namespace) {
      console.log(`   Namespace: ${options.namespace}`);
    }
    console.log(
      `   Supported formats: ${SUPPORTED_EXTENSIONS.join(", ")}\n`
    );

    try {
      // Find all supported files
      console.log("🔍 Scanning for documents...");
      const files = findSupportedFiles(options.source);

      if (files.length === 0) {
        console.log(
          "\n⚠️  No supported documents found in the specified path."
        );
        console.log(
          `   Supported formats: ${SUPPORTED_EXTENSIONS.join(", ")}`
        );
        return;
      }

      console.log(`   Found ${files.length} document(s)\n`);

      if (options.verbose) {
        for (const file of files) {
          console.log(`   - ${path.basename(file)}`);
        }
        console.log("");
      }

      // Clear existing indexes before starting
      console.log("🗑️  Clearing existing indexes...");
      await clearAllIndexes(options.namespace);
      console.log("   ✅ Indexes cleared\n");

      // Process each file
      let totalChunks = 0;
      let successCount = 0;
      let errorCount = 0;

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileName = path.basename(file);
        const fileProgress = `[${i + 1}/${files.length}]`;

        console.log(`\n${fileProgress} Processing: ${fileName}`);

        try {
          // Convert document via Docling
          if (options.verbose) {
            console.log("   📄 Converting with Docling...");
          }
          const doclingDoc = await convertDocument(file);

          // Chunk the document
          if (options.verbose) {
            console.log("   ✂️  Chunking document...");
          }
          const chunks = chunkDocument(doclingDoc);

          if (chunks.length === 0) {
            console.log("   ⚠️  No content extracted from document");
            continue;
          }

          if (options.verbose) {
            console.log(`   📊 Created ${chunks.length} chunk(s)`);

            // Show sample chunk info
            if (chunks.length > 0) {
              const firstChunk = chunks[0];
              console.log(
                `      First chunk: ${firstChunk.text.slice(0, 100).replace(/\n/g, " ")}...`
              );
              if (firstChunk.headings.length > 0) {
                console.log(
                  `      Headings: ${firstChunk.headings.join(" > ")}`
                );
              }
            }
          }

          // Index chunks to Pinecone
          const progressCallback = options.verbose
            ? (stage: string, progress: number) => {
                process.stdout.write(
                  `\r   🔄 ${stage}... ${progressBar(progress, 100)}`
                );
                if (progress === 100) {
                  process.stdout.write("\n");
                }
              }
            : undefined;

          if (options.verbose) {
            console.log("   🚀 Indexing to Pinecone...");
          }

          const result = await indexChunks(
            chunks,
            fileName,
            options.namespace,
            progressCallback
          );

          totalChunks += chunks.length;
          successCount++;

          if (result.denseCount === result.sparseCount) {
            console.log(
              `   ✅ Indexed ${result.denseCount} chunks (dense + sparse)`
            );
          } else {
            console.log(
              `   ✅ Indexed ${result.denseCount} dense, ${result.sparseCount} sparse chunks`
            );
          }
        } catch (error) {
          errorCount++;
          const message =
            error instanceof Error ? error.message : String(error);
          console.log(`   ❌ Error: ${message}`);

          if (options.verbose && error instanceof Error && error.stack) {
            console.log(`      ${error.stack.split("\n").slice(1, 3).join("\n      ")}`);
          }
        }
      }

      // Summary
      console.log("\n" + "─".repeat(50));
      console.log("📊 Indexing Summary\n");
      console.log(`   Documents processed: ${successCount}/${files.length}`);
      console.log(`   Total chunks indexed: ${formatNumber(totalChunks)}`);
      if (errorCount > 0) {
        console.log(`   Errors: ${errorCount}`);
      }

      // Show index stats
      try {
        const stats = await getIndexStats();
        console.log("\n📈 Index Statistics:");
        if (stats.dense) {
          console.log(
            `   Dense index: ${formatNumber(stats.dense.vectorCount)} vectors (${stats.dense.dimension}d)`
          );
        }
        if (stats.sparse) {
          console.log(
            `   Sparse index: ${formatNumber(stats.sparse.vectorCount)} vectors`
          );
        }
      } catch {
        // Stats retrieval failed, skip
      }

      console.log("\n✨ Indexing complete!");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`\n❌ Fatal error: ${message}`);

      if (options.verbose && error instanceof Error && error.stack) {
        console.error(`\n${error.stack}`);
      }

      process.exit(1);
    }
  });
