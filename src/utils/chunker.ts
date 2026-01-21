import type {
  DoclingDocument,
  DoclingTextItem,
  DoclingTableItem,
} from "../services/docling.js";

/**
 * A chunk of document content with metadata for RAG
 */
export interface Chunk {
  /** The text content of the chunk */
  text: string;
  /** Parent heading hierarchy (e.g., ["Chapter 1", "Introduction"]) */
  headings: string[];
  /** Types of items in this chunk (paragraph, table, list_item, etc.) */
  itemTypes: string[];
  /** Position in the document (chunk index) */
  chunkIndex: number;
}

/**
 * Options for chunking
 */
export interface ChunkOptions {
  /** Target chunk size in characters (approximate) */
  maxChunkSize?: number;
  /** Minimum chunk size before merging with next */
  minChunkSize?: number;
}

const DEFAULT_OPTIONS: Required<ChunkOptions> = {
  maxChunkSize: 2000, // ~500 tokens
  minChunkSize: 200, // ~50 tokens
};

/**
 * Resolve a JSON pointer reference (e.g., "#/texts/0") to get the index and type
 */
function parseRef(ref: string): { type: string; index: number } | null {
  const match = ref.match(/^#\/(\w+)\/(\d+)$/);
  if (!match) return null;
  return { type: match[1], index: parseInt(match[2], 10) };
}

/**
 * Get text content from a table, formatted as a readable string
 */
function tableToText(table: DoclingTableItem): string {
  if (!table.data?.table_cells) {
    return "[Table]";
  }

  const cells = table.data.table_cells;
  const numRows = table.data.num_rows || 0;
  const numCols = table.data.num_cols || 0;

  if (numRows === 0 || numCols === 0) {
    return cells.map((c) => c.text).join(" | ");
  }

  // Build a 2D grid
  const grid: string[][] = Array(numRows)
    .fill(null)
    .map(() => Array(numCols).fill(""));

  for (const cell of cells) {
    const row = cell.start_row_offset_idx ?? 0;
    const col = cell.start_col_offset_idx ?? 0;
    if (row < numRows && col < numCols) {
      grid[row][col] = cell.text;
    }
  }

  // Format as a simple text table
  return grid.map((row) => row.join(" | ")).join("\n");
}

/**
 * Check if an item is a heading/section header
 */
function isHeading(label: string): boolean {
  return (
    label === "section_header" ||
    label === "title" ||
    label === "chapter" ||
    label.includes("heading")
  );
}

/**
 * Internal representation of a document item for chunking
 */
interface DocItem {
  text: string;
  label: string;
  isHeading: boolean;
  ref: string;
}

/**
 * Walk the document tree in reading order and extract items
 */
function extractItemsInOrder(doc: DoclingDocument): DocItem[] {
  const items: DocItem[] = [];
  const textsMap = new Map<string, DoclingTextItem>();
  const tablesMap = new Map<string, DoclingTableItem>();

  // Build lookup maps
  if (doc.texts) {
    for (const text of doc.texts) {
      textsMap.set(text.self_ref, text);
    }
  }
  if (doc.tables) {
    for (const table of doc.tables) {
      tablesMap.set(table.self_ref, table);
    }
  }

  // Track visited refs to avoid duplicates
  const visited = new Set<string>();

  function processRef(ref: string) {
    if (visited.has(ref)) return;
    visited.add(ref);

    const parsed = parseRef(ref);
    if (!parsed) return;

    if (parsed.type === "texts" && doc.texts?.[parsed.index]) {
      const textItem = doc.texts[parsed.index];
      items.push({
        text: textItem.text,
        label: textItem.label,
        isHeading: isHeading(textItem.label),
        ref: ref,
      });

      // Process children if any
      if (textItem.children) {
        for (const child of textItem.children) {
          processRef(child.$ref);
        }
      }
    } else if (parsed.type === "tables" && doc.tables?.[parsed.index]) {
      const tableItem = doc.tables[parsed.index];
      items.push({
        text: tableToText(tableItem),
        label: "table",
        isHeading: false,
        ref: ref,
      });
    } else if (parsed.type === "body" || parsed.type === "groups") {
      // These are container nodes, process children
      // For now we rely on the body structure
    }
  }

  // Start from body if available
  if (doc.body?.children) {
    for (const child of doc.body.children) {
      processRef(child.$ref);
    }
  }

  // If body traversal didn't find items, fall back to linear order
  if (items.length === 0) {
    if (doc.texts) {
      for (const text of doc.texts) {
        items.push({
          text: text.text,
          label: text.label,
          isHeading: isHeading(text.label),
          ref: text.self_ref,
        });
      }
    }
    if (doc.tables) {
      for (const table of doc.tables) {
        items.push({
          text: tableToText(table),
          label: "table",
          isHeading: false,
          ref: table.self_ref,
        });
      }
    }
  }

  return items;
}

/**
 * Split text at sentence boundaries
 */
function splitAtSentences(text: string, maxSize: number): string[] {
  if (text.length <= maxSize) {
    return [text];
  }

  const chunks: string[] = [];
  const sentences = text.match(/[^.!?]+[.!?]+\s*/g) || [text];

  let currentChunk = "";

  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > maxSize && currentChunk) {
      chunks.push(currentChunk.trim());
      currentChunk = sentence;
    } else {
      currentChunk += sentence;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  // If we still have chunks that are too large, split by words
  const result: string[] = [];
  for (const chunk of chunks) {
    if (chunk.length > maxSize) {
      const words = chunk.split(/\s+/);
      let current = "";
      for (const word of words) {
        if (current.length + word.length + 1 > maxSize && current) {
          result.push(current.trim());
          current = word;
        } else {
          current += (current ? " " : "") + word;
        }
      }
      if (current) {
        result.push(current.trim());
      }
    } else {
      result.push(chunk);
    }
  }

  return result;
}

/**
 * Chunk a DoclingDocument into semantically meaningful pieces.
 *
 * This chunker:
 * - Walks the document tree in reading order
 * - Groups content by sections (keeps headings with their paragraphs)
 * - Never splits tables - each table becomes its own chunk
 * - Merges small consecutive items until reaching target size
 * - Splits oversized paragraphs at sentence boundaries
 * - Tracks parent headings for each chunk (context for retrieval)
 */
export function chunkDocument(
  doc: DoclingDocument,
  options: ChunkOptions = {}
): Chunk[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const items = extractItemsInOrder(doc);

  if (items.length === 0) {
    return [];
  }

  const chunks: Chunk[] = [];
  let currentHeadings: string[] = [];
  let currentText = "";
  let currentTypes: Set<string> = new Set();

  function flushChunk() {
    if (currentText.trim()) {
      // Check if we need to split the chunk
      if (currentText.length > opts.maxChunkSize) {
        const parts = splitAtSentences(currentText.trim(), opts.maxChunkSize);
        for (const part of parts) {
          chunks.push({
            text: part,
            headings: [...currentHeadings],
            itemTypes: Array.from(currentTypes),
            chunkIndex: chunks.length,
          });
        }
      } else {
        chunks.push({
          text: currentText.trim(),
          headings: [...currentHeadings],
          itemTypes: Array.from(currentTypes),
          chunkIndex: chunks.length,
        });
      }
    }
    currentText = "";
    currentTypes = new Set();
  }

  for (const item of items) {
    // Handle headings - they update context and may trigger a new chunk
    if (item.isHeading) {
      // Flush current chunk before starting new section
      if (currentText.length >= opts.minChunkSize) {
        flushChunk();
      }

      // Update heading hierarchy based on label
      if (item.label === "title") {
        currentHeadings = [item.text];
      } else if (item.label === "chapter") {
        currentHeadings = [item.text];
      } else {
        // Section headers add to hierarchy, but we simplify to keep last few
        if (currentHeadings.length >= 3) {
          currentHeadings = currentHeadings.slice(-2);
        }
        currentHeadings.push(item.text);
      }

      // Include heading text in the next chunk
      currentText += item.text + "\n\n";
      currentTypes.add(item.label);
      continue;
    }

    // Tables are kept as separate chunks (never split)
    if (item.label === "table") {
      // Flush any accumulated text first
      if (currentText.trim()) {
        flushChunk();
      }

      // Create table chunk
      chunks.push({
        text: item.text,
        headings: [...currentHeadings],
        itemTypes: ["table"],
        chunkIndex: chunks.length,
      });
      continue;
    }

    // Regular content - accumulate until we reach target size
    const newText = currentText + item.text + "\n\n";

    if (newText.length > opts.maxChunkSize) {
      // Flush current and start new chunk with this item
      flushChunk();
      currentText = item.text + "\n\n";
      currentTypes.add(item.label);
    } else {
      currentText = newText;
      currentTypes.add(item.label);
    }
  }

  // Flush remaining content
  flushChunk();

  // Re-index chunks
  for (let i = 0; i < chunks.length; i++) {
    chunks[i].chunkIndex = i;
  }

  return chunks;
}

