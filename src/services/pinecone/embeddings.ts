import { getClient, DENSE_MODEL, SPARSE_MODEL } from "./client.js";
import type { DenseEmbedding, SparseEmbedding } from "./types.js";

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
