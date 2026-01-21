import { getConfig } from "../config.js";
import * as fs from "fs";
import * as path from "path";

/**
 * DoclingDocument represents the structured output from Docling.
 * This is a simplified type covering the fields we need for chunking.
 */
export interface DoclingDocument {
  name?: string;
  body?: DoclingBody;
  texts?: DoclingTextItem[];
  tables?: DoclingTableItem[];
  pictures?: DoclingPictureItem[];
}

export interface DoclingBody {
  children?: DoclingRef[];
  name?: string;
  label?: string;
}

export interface DoclingRef {
  $ref: string;
}

export interface DoclingTextItem {
  self_ref: string;
  parent?: DoclingRef;
  children?: DoclingRef[];
  label: string; // "paragraph", "section_header", "title", "list_item", etc.
  text: string;
  prov?: DoclingProvenance[];
}

export interface DoclingTableItem {
  self_ref: string;
  parent?: DoclingRef;
  label: string;
  data?: DoclingTableData;
  prov?: DoclingProvenance[];
}

export interface DoclingTableData {
  table_cells?: DoclingTableCell[];
  num_rows?: number;
  num_cols?: number;
}

export interface DoclingTableCell {
  text: string;
  row_span?: number;
  col_span?: number;
  start_row_offset_idx?: number;
  start_col_offset_idx?: number;
}

export interface DoclingPictureItem {
  self_ref: string;
  parent?: DoclingRef;
  label: string;
  prov?: DoclingProvenance[];
}

export interface DoclingProvenance {
  page_no?: number;
  bbox?: number[];
}

export interface DoclingConvertResponse {
  document: {
    md_content?: string;
    json_content?: DoclingDocument;
    html_content?: string;
    text_content?: string;
    doctags_content?: string;
  };
  status: "success" | "partial_success" | "skipped" | "failure";
  processing_time: number;
  timings?: Record<string, number>;
  errors: string[];
}

/**
 * Supported file extensions for document conversion
 */
export const SUPPORTED_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".doc",
  ".pptx",
  ".html",
  ".htm",
  ".md",
  ".txt",
  ".xlsx",
  ".csv",
];

/**
 * Check if a file extension is supported by Docling
 */
export function isSupportedFile(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return SUPPORTED_EXTENSIONS.includes(ext);
}

/**
 * Get the MIME type for a file based on its extension
 */
function getMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes: Record<string, string> = {
    ".pdf": "application/pdf",
    ".docx":
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx":
      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".xlsx":
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
  };
  return mimeTypes[ext] || "application/octet-stream";
}

/**
 * Log the full Docling response to a file for debugging/analysis
 */
function logDoclingOutput(
  fileName: string,
  response: DoclingConvertResponse
): void {
  const logsDir = path.join(process.cwd(), "docling-logs");

  // Create logs directory if it doesn't exist
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }

  // Create a timestamp-based filename
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const baseFileName = path.basename(fileName, path.extname(fileName));
  const logFileName = `${timestamp}_${baseFileName}.json`;
  const logFilePath = path.join(logsDir, logFileName);

  // Write the full response to the log file
  const logData = {
    source_file: fileName,
    timestamp: new Date().toISOString(),
    docling_response: response,
  };

  fs.writeFileSync(logFilePath, JSON.stringify(logData, null, 2), "utf-8");
  console.log(`   📝 Logged full Docling output to: ${logFilePath}`);
}

/**
 * Convert a document using the Docling API.
 * Returns the structured DoclingDocument for chunking.
 */
export async function convertDocument(
  filePath: string
): Promise<DoclingDocument> {
  const config = getConfig();
  const absolutePath = path.resolve(filePath);

  if (!fs.existsSync(absolutePath)) {
    throw new Error(`File not found: ${absolutePath}`);
  }

  if (!isSupportedFile(absolutePath)) {
    throw new Error(
      `Unsupported file type: ${path.extname(absolutePath)}. Supported: ${SUPPORTED_EXTENSIONS.join(", ")}`
    );
  }

  const fileBuffer = fs.readFileSync(absolutePath);
  const fileName = path.basename(absolutePath);
  const mimeType = getMimeType(absolutePath);

  // Create multipart form data
  const boundary = `----FormBoundary${Date.now()}`;
  const formParts: Buffer[] = [];

  // Add file part
  formParts.push(
    Buffer.from(
      `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="files"; filename="${fileName}"\r\n` +
        `Content-Type: ${mimeType}\r\n\r\n`
    )
  );
  formParts.push(fileBuffer);
  formParts.push(Buffer.from("\r\n"));

  // Add to_formats parameter - request JSON for structured output
  formParts.push(
    Buffer.from(
      `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="to_formats"\r\n\r\n` +
        `json\r\n`
    )
  );

  // Add do_ocr parameter
  formParts.push(
    Buffer.from(
      `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="do_ocr"\r\n\r\n` +
        `true\r\n`
    )
  );

  // Add do_table_structure parameter
  formParts.push(
    Buffer.from(
      `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="do_table_structure"\r\n\r\n` +
        `true\r\n`
    )
  );

  // Close boundary
  formParts.push(Buffer.from(`--${boundary}--\r\n`));

  const body = Buffer.concat(formParts);

  const url = `${config.doclingUrl}/convert/file`;

  // Use AbortController for timeout (5 minutes for large documents)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": `multipart/form-data; boundary=${boundary}`,
        Accept: "application/json",
        Authorization: `Bearer ${config.doclingApiKey}`,
      },
      body: body,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Docling conversion timed out after 5 minutes");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Docling API error (${response.status}): ${errorText.slice(0, 500)}`
    );
  }

  const result = (await response.json()) as DoclingConvertResponse;

  // Log the full Docling output to a file
  logDoclingOutput(fileName, result);

  if (result.status === "failure") {
    throw new Error(`Docling conversion failed: ${result.errors.join(", ")}`);
  }

  if (!result.document.json_content) {
    throw new Error("Docling did not return JSON content");
  }

  // Add the filename to the document for reference
  const doc = result.document.json_content;
  doc.name = fileName;

  return doc;
}

/**
 * Find all supported files in a directory (recursive)
 */
export function findSupportedFiles(dirPath: string): string[] {
  const absolutePath = path.resolve(dirPath);
  const files: string[] = [];

  if (!fs.existsSync(absolutePath)) {
    throw new Error(`Directory not found: ${absolutePath}`);
  }

  const stat = fs.statSync(absolutePath);
  if (!stat.isDirectory()) {
    // Single file
    if (isSupportedFile(absolutePath)) {
      return [absolutePath];
    }
    throw new Error(`Not a supported file: ${absolutePath}`);
  }

  function walkDir(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // Skip hidden directories
        if (!entry.name.startsWith(".")) {
          walkDir(fullPath);
        }
      } else if (entry.isFile() && isSupportedFile(fullPath)) {
        files.push(fullPath);
      }
    }
  }

  walkDir(absolutePath);
  return files.sort();
}
