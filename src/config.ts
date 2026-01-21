import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

export interface Config {
  doclingUrl: string;
  doclingApiKey: string;
  pineconeApiKey: string;
  pineconeDenseIndex: string;
  pineconeSparseIndex: string;
  openaiApiKey: string;
}

function getEnvVar(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function loadConfig(): Config {
  return {
    doclingUrl: getEnvVar("DOCLING_URL"),
    pineconeApiKey: getEnvVar("PINECONE_API_KEY"),
    pineconeDenseIndex: getEnvVar("PINECONE_DENSE_INDEX"),
    pineconeSparseIndex: getEnvVar("PINECONE_SPARSE_INDEX"),
    openaiApiKey: getEnvVar("OPENAI_API_KEY"),
    doclingApiKey: getEnvVar("DOCLING_API_KEY"),
  };
}

// Lazy-loaded config singleton
let _config: Config | null = null;

export function getConfig(): Config {
  if (!_config) {
    _config = loadConfig();
  }
  return _config;
}

export default getConfig;
