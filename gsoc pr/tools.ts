import fs from 'fs';
import { z } from 'zod';
import { execSync } from "child_process";
import { config } from "./config";

export const tools = {
    getFileContents: {
    description: 'Get the contents of a file',
    inputSchema: z.object({
      filePath: z.string(),
    }),
    execute: async ({ filePath }) => {
      const fileContents = fs.readFileSync(filePath, 'utf8');
      console.log(`Reading file ${filePath}`);
      console.log(fileContents);
      return fileContents;
    },
  },
  command: {
    description: 'Run a command in the codebase',
    inputSchema: z.object({
      command: z.string(),
    }),
    execute: async ({ command }) => {
      console.log(`Running command ${command}`);
      return execSync(command, { cwd: config.codebasePath });
    },
  },
  createFile: {
    description: 'Create a file in the codebase',
    inputSchema: z.object({
      filePath: z.string(),
      content: z.string(),
    }),
    execute: async ({ filePath, content }) => {
      console.log(`Creating file ${filePath}`);
      fs.writeFileSync(filePath, content);
      return `File ${filePath} created`;
    },
  },
  deleteFile: {
    description: 'Delete a file in the codebase',
    inputSchema: z.object({
      filePath: z.string(),
    }),
    execute: async ({ filePath }) => {
      console.log(`Deleting file ${filePath}`);
      fs.unlinkSync(filePath);
      return `File ${filePath} deleted`;
    },
  },
  createDirectory: {
    description: 'Create a directory in the codebase',
    inputSchema: z.object({
      directoryPath: z.string(),
    }),
    execute: async ({ directoryPath }) => {
      console.log(`Creating directory ${directoryPath}`);
      fs.mkdirSync(directoryPath, { recursive: true });
      return `Directory ${directoryPath} created`;
    },
  },
  deleteDirectory: {
    description: 'Delete a directory in the codebase',
    inputSchema: z.object({
      directoryPath: z.string(),
    }),
    execute: async ({ directoryPath }) => {
      console.log(`Deleting directory ${directoryPath}`);
      fs.rmdirSync(directoryPath, { recursive: true });
      return `Directory ${directoryPath} deleted`;
    },
  },
  applyGitPatch: {
    description: 'Apply a git patch to the codebase',
    inputSchema: z.object({
      patch: z.string(),
    }),
    execute: async ({ patch }) => {
      console.log(`Applying git patch ${patch}`);
      return execSync(`git apply ${patch}`, { cwd: config.codebasePath });
    },
  },
}