import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { tools } from "../tools";
import { stepCountIs, streamText } from "ai";
import { config } from "../config";

export const main = async () => {
    const openrouter = createOpenRouter({
        apiKey: process.env.OPENROUTER_API_KEY || '',
        });

        const response = streamText({
        stopWhen: stepCountIs(30), 
        model: openrouter(config.modelName),
        maxOutputTokens: 16384,
        prompt: `
        You are an expert agent that can fix codebases. 
        You are given a description of a problem and a codebase path. 
        You need to fix the codebase to fix the problem. 
        You need to fix this issue using the tool calls provided to you.
        The codebase path is ${config.codebasePath}.
        The description of the problem is ${config.description}.
        
        Here are the tool calls you can use:
        - getFileContents: Get the contents of a file
        - command: Run a command in the codebase (eg:ls, git add, npm install express, etc.)
        - createFile: Create a file in the codebase
        - deleteFile: Delete a file in the codebase
        - applyGitPatch: Apply a git patch to the codebase. Used for editing a file.
        - createDirectory: Create a directory in the codebase
        - deleteDirectory: Delete a directory in the codebase
        Make the code change that you think is necessary to fix the problem. 
        Use the tools like creteFile or applyGitPatch to make the code change.
        Return the final summary of the code change you made.


        ##How to test the codebase?
        You can run the command npm run run_tests to test the full codebase.
        Or
        npm run run_tests file_path to test a specific file.
        `,
        tools: tools,
    });

    await response.consumeStream();
    return response.text;
}