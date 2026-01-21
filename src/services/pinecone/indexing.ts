import { getClient } from "./client.js";
import { getConfig } from "../../config.js";
import type { Chunk } from "../../utils/chunker.js";
import type {
  DenseRecord,
  SparseRecord,
  DenseEmbedding,
  SparseEmbedding,
  ChunkMetadata,
} from "./types.js";
import {
  generateDenseEmbeddings,
  generateSparseEmbeddings,
} from "./embeddings.js";

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
