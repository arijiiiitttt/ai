// Vibe coded - Multi agent system + RAG
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { OpenRouter } from '@openrouter/sdk';
import { tools } from "../tools";
import { streamText, embed, embedMany, stepCountIs } from "ai";
import { config } from "../config";
import { z } from "zod";
import fs from "fs";
import path from "path";
import type { CreateEmbeddingsResponse, CreateEmbeddingsResponseBody } from "@openrouter/sdk/models/operations";

// Initialize OpenRouter
const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
});

const openRouterNativeSdk = new OpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY || "",
});
  
// ============= IN-MEMORY VECTOR DATABASE =============

interface VectorEntry {
  id: string;
  filePath: string;
  content: string;
  embedding: number[];
  metadata: {
    extension: string;
    lines: number;
    size: number;
  };
}

class InMemoryVectorDB {
  private entries: VectorEntry[] = [];

  add(entry: VectorEntry): void {
    // Update if exists, otherwise add
    const existingIndex = this.entries.findIndex((e) => e.id === entry.id);
    if (existingIndex >= 0) {
      this.entries[existingIndex] = entry;
    } else {
      this.entries.push(entry);
    }
  }

  addMany(entries: VectorEntry[]): void {
    entries.forEach((entry) => this.add(entry));
  }

  // Cosine similarity between two vectors
  private cosineSimilarity(a: number[], b: number[]): number {
    if (!a || !b || a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      const aVal = a[i] ?? 0;
      const bVal = b[i] ?? 0;
      dotProduct += aVal * bVal;
      normA += aVal * aVal;
      normB += bVal * bVal;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  // Search for similar entries
  search(
    queryEmbedding: number[],
    topK: number = 5
  ): Array<{ entry: VectorEntry; score: number }> {
    const results = this.entries
      .map((entry) => ({
        entry,
        score: this.cosineSimilarity(queryEmbedding, entry.embedding),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return results;
  }

  size(): number {
    return this.entries.length;
  }

  clear(): void {
    this.entries = [];
  }
}

// Global vector database instance
const vectorDB = new InMemoryVectorDB();

// ============= CODEBASE EMBEDDING =============

// File extensions to include
const INCLUDED_EXTENSIONS = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".json",
  ".md",
  ".py",
  ".go",
  ".rs",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".css",
  ".scss",
  ".html",
  ".vue",
  ".svelte",
  ".yaml",
  ".yml",
  ".toml",
  ".sql",
  ".graphql",
  ".gql",
]);

// Directories to skip
const EXCLUDED_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".nuxt",
  "coverage",
  "__pycache__",
  ".venv",
  "venv",
  "target",
  ".idea",
  ".vscode",
]);

// Recursively get all files in a directory
function getAllFiles(dirPath: string, fileList: string[] = []): string[] {
  try {
    const files = fs.readdirSync(dirPath);

    for (const file of files) {
      const filePath = path.join(dirPath, file);

      try {
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
          if (!EXCLUDED_DIRS.has(file)) {
            getAllFiles(filePath, fileList);
          }
        } else {
          const ext = path.extname(file).toLowerCase();
          if (INCLUDED_EXTENSIONS.has(ext)) {
            // Skip files larger than 100KB to avoid token limits
            if (stat.size < 100 * 1024) {
              fileList.push(filePath);
            }
          }
        }
      } catch {
        // Skip files we can't access
      }
    }
  } catch {
    // Skip directories we can't access
  }

  return fileList;
}

// Chunk content for embedding (max ~8000 chars to stay within token limits)
function chunkContent(
  content: string,
  maxChars: number = 6000
): string[] {
  if (content.length <= maxChars) {
    return [content];
  }

  const chunks: string[] = [];
  const lines = content.split("\n");
  let currentChunk = "";

  for (const line of lines) {
    if (currentChunk.length + line.length + 1 > maxChars) {
      if (currentChunk) {
        chunks.push(currentChunk);
      }
      currentChunk = line;
    } else {
      currentChunk += (currentChunk ? "\n" : "") + line;
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

// Embed the codebase
async function embedCodebase(codebasePath: string): Promise<void> {
  console.log(`\n📂 Scanning codebase at: ${codebasePath}`);

  const files = getAllFiles(codebasePath);
  console.log(`📄 Found ${files.length} files to embed`);

  // Prepare all content for embedding
  const fileContents: Array<{
    filePath: string;
    content: string;
    chunkIndex: number;
    totalChunks: number;
  }> = [];

  for (const filePath of files) {
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      const chunks = chunkContent(content);

      chunks.forEach((chunk, index) => {
        fileContents.push({
          filePath,
          content: chunk,
          chunkIndex: index,
          totalChunks: chunks.length,
        });
      });
    } catch {
      // Skip files we can't read
    }
  }

  console.log(`📦 Total chunks to embed: ${fileContents.length}`);

  // Embed in batches of 20 (to respect rate limits)
  const batchSize = 20;
  console.log("openrouter");

  for (let i = 0; i < fileContents.length; i += batchSize) {
    const batch = fileContents.slice(i, i + batchSize);
    const progress = Math.min(i + batchSize, fileContents.length);

    console.log(`🔄 Embedding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(fileContents.length / batchSize)} (${progress}/${fileContents.length})`);

    try {
      // Create embedding text with file path context
      const textsToEmbed = batch.map(
        (f) =>
          `File: ${path.relative(codebasePath, f.filePath)}\n\n${f.content}`
      );

    const response = await openRouterNativeSdk.embeddings.generate({
        model: 'openai/text-embedding-3-small',
        input: textsToEmbed,
    });

      const result = (response as CreateEmbeddingsResponseBody).data[0]?.embedding;

      // Store embeddings in vector DB
      batch.forEach((file, index) => {
        const embedding = result;
        if (!embedding) return; // Skip if embedding failed
        
        const lines = file.content.split("\n").length;
        const stat = fs.statSync(file.filePath);

        vectorDB.add({
          id:
            file.totalChunks > 1
              ? `${file.filePath}#chunk${file.chunkIndex}`
              : file.filePath,
          filePath: file.filePath,
          content: file.content,
          embedding: embedding as number[],
          metadata: {
            extension: path.extname(file.filePath),
            lines,
            size: stat.size,
          },
        });
      });
    } catch (error) {
      console.error(`❌ Error embedding batch:`, error);
      // Continue with next batch
    }

    // Small delay to avoid rate limiting
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  console.log(`✅ Embedded ${vectorDB.size()} chunks into vector database\n`);
}

// ============= SEMANTIC SEARCH TOOL =============

const semanticSearchTool = {
  description:
    "Search the codebase semantically to find relevant files. Use this to find files related to a concept, feature, or problem. Returns the most relevant file paths and snippets.",
  inputSchema: z.object({
    query: z
      .string()
      .describe(
        "Natural language query describing what you're looking for (e.g., 'user authentication logic', 'database connection handling', 'error handling for API calls')"
      ),
    topK: z
      .number()
      .optional()
      .default(5)
      .describe("Number of results to return (default: 5)"),
  }),
  execute: async ({
    query,
    topK = 5,
  }: {
    query: string;
    topK?: number;
  }): Promise<string> => {
    console.log(openrouter.textEmbeddingModel);
    console.log(`🔍 Semantic search: "${query}"`);

    const response = await openRouterNativeSdk.embeddings.generate({
        model: 'openai/text-embedding-3-small',
        input: [query],
    });
    const queryResult = (response as CreateEmbeddingsResponseBody).data[0]?.embedding;

    // Search vector database
    const results = vectorDB.search(queryResult as number[], topK);

    if (results.length === 0) {
      return "No relevant files found. The codebase may not be embedded yet.";
    }

    // Format results
    const formattedResults = results.map((r, index) => {
      const relativePath = path.relative(config.codebasePath, r.entry.filePath);
      const preview = r.entry.content.slice(0, 500) + (r.entry.content.length > 500 ? "..." : "");

      return `
### Result ${index + 1} (Score: ${r.score.toFixed(3)})
**File:** ${relativePath}
**Lines:** ${r.entry.metadata.lines} | **Size:** ${r.entry.metadata.size} bytes

\`\`\`
${preview}
\`\`\`
`;
    });

    console.log(`📋 Found ${results.length} relevant files`);
    return formattedResults.join("\n---\n");
  },
};

// ============= COMBINED TOOLS =============

const agent3Tools = {
  // Semantic search for finding relevant files
  semanticSearch: semanticSearchTool,
  // Standard file operations
  getFileContents: tools.getFileContents,
  command: tools.command,
  createFile: tools.createFile,
  deleteFile: tools.deleteFile,
  applyGitPatch: tools.applyGitPatch,
  createDirectory: tools.createDirectory,
  deleteDirectory: tools.deleteDirectory,
};

// ============= MAIN AGENT =============

export const main = async () => {
  console.log("═".repeat(60));
  console.log("🧠 SEMANTIC CODEBASE AGENT (Agent 3)");
  console.log("═".repeat(60));
  console.log(`\n📁 Target: ${config.codebasePath}`);
  console.log(`📝 Task: ${config.description.slice(0, 100)}...\n`);
  console.log("═".repeat(60));

  try {
    // Phase 1: Embed the codebase
    console.log("\n📚 [PHASE 1] Embedding codebase into vector database...\n");
    await embedCodebase(config.codebasePath);

    // Phase 2: Run the agent with semantic search capabilities
    console.log("\n🤖 [PHASE 2] Running agent with semantic search...\n");

    const response = streamText({
      model: openrouter(config.modelName),
      stopWhen: stepCountIs(30),
      maxOutputTokens: 16384,
      tools: agent3Tools,
      prompt: `You are an expert software engineer with semantic search capabilities.

CODEBASE PATH: ${config.codebasePath}

PROBLEM DESCRIPTION:
${config.description}

YOUR CAPABILITIES:
1. **semanticSearch**: Use this FIRST to find relevant files. Search by concept, not exact names.
   Example queries: "test files for membership", "cancel membership logic", "graphql mutation handlers"

2. **getFileContents**: Read specific files after finding them via semantic search

3. **command**: Run shell commands (ls, git, npm test, etc.)

4. **createFile**: Create new files

5. **applyGitPatch**: Apply patches to edit existing files

6. **createDirectory/deleteDirectory/deleteFile**: Manage files and directories

WORKFLOW:
1. Start with semanticSearch to find relevant files for the task
2. Read the most relevant files to understand the code
3. Look for existing test files and patterns
4. Implement the required changes
5. Run tests to verify your changes

IMPORTANT:
- Use semantic search to explore the codebase efficiently
- Don't rely on guessing file paths - search for them
- Follow existing patterns in the codebase
- Test your changes

##How to test the codebase?
You can run the command npm run run_tests to test the full codebase.
Or npm run run_tests file_path to test a specific file.

Begin by using semanticSearch to find the relevant files for this task.`,
    });

    // Stream the response
    for await (const chunk of response.fullStream) {
      if (chunk.type === "text-delta") {
        process.stdout.write(chunk.text);
      } else if (chunk.type === "tool-call") {
        console.log(`\n🔧 Tool call: ${chunk.toolName}`);
      } else if (chunk.type === "tool-result") {
        console.log(`\n✅ Tool result received`);
      }
    }

    const finalText = await response.text;

    console.log("\n\n" + "═".repeat(60));
    console.log("📋 FINAL SUMMARY");
    console.log("═".repeat(60));
    console.log(finalText);

    return finalText;
  } catch (error) {
    console.error("\n❌ Error:", error);
    throw error;
  }
};

