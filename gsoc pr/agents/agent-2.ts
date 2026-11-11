// Vibe coded - Multi agent system
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { tools } from "../tools";
import { streamText, stepCountIs } from "ai";
import { config } from "../config";

// Initialize OpenRouter
const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
});

// Read-only tools for exploration/planning
const explorerTools = {
  getFileContents: tools.getFileContents,
  command: tools.command,
};

// Full toolkit for code modifications
const coderTools = {
  getFileContents: tools.getFileContents,
  command: tools.command,
  createFile: tools.createFile,
  deleteFile: tools.deleteFile,
  applyGitPatch: tools.applyGitPatch,
  createDirectory: tools.createDirectory,
  deleteDirectory: tools.deleteDirectory,
};

// Testing tools
const testerTools = {
  getFileContents: tools.getFileContents,
  command: tools.command,
};

// ============= AGENT DEFINITIONS =============

/**
 * Planner Agent - Analyzes the problem and creates a detailed action plan
 */
const runPlannerAgent = async (): Promise<string> => {
  console.log("\n🧠 [PLANNER AGENT] Analyzing problem and creating plan...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(15),
    tools: explorerTools,
    prompt: `You are a senior software architect and planner agent. Your job is to analyze the problem and create a detailed plan.

CODEBASE PATH: ${config.codebasePath}

PROBLEM DESCRIPTION:
${config.description}

YOUR TASK:
1. Explore the codebase structure using the command tool (ls, find, tree commands)
2. Read the relevant source file mentioned in the problem
3. Identify what tests exist and what's missing
4. Check the current test coverage status

After exploration, create a DETAILED PLAN with:
- Files that need to be modified or created
- Specific code sections that need test coverage
- Edge cases and scenarios that should be tested
- Step-by-step instructions for the coder agent

Output your plan in a structured format with clear sections.`,
  });

  await response.consumeStream();
  console.log("\n📋 [PLANNER] Plan created\n");
  return response.text;
};

/**
 * Explorer Agent - Deep dives into the codebase to gather context
 */
const runExplorerAgent = async (plan: string): Promise<string> => {
  console.log("\n🔍 [EXPLORER AGENT] Gathering detailed codebase context...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(20),
    tools: explorerTools,
    prompt: `You are an expert code explorer agent. Your job is to deeply understand the codebase.

CODEBASE PATH: ${config.codebasePath}
PLAN FROM PLANNER:
${plan}

YOUR TASK:
1. Read ALL relevant files mentioned in the plan
2. Understand the testing patterns used in this codebase
3. Find existing test files for similar modules to use as reference
4. Identify imports, dependencies, and mocking patterns used
5. Look at how other test files in the project are structured

Provide a comprehensive summary including:
- File contents of the source file to be tested
- Existing test file contents (if any)
- Reference test files showing the testing patterns
- Import statements and mocking patterns used
- Any helper utilities or test fixtures available

Be thorough - the coder agent will rely on this information.`,
  });

  await response.consumeStream();
  console.log("\n📚 [EXPLORER] Context gathered\n");
  return response.text;
};

/**
 * Coder Agent - Implements the actual code changes
 */
const runCoderAgent = async (
  plan: string,
  context: string
): Promise<string> => {
  console.log("\n💻 [CODER AGENT] Implementing code changes...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(25),
    tools: coderTools,
    prompt: `You are an expert software developer agent. Your job is to implement code changes.

CODEBASE PATH: ${config.codebasePath}

PLAN FROM PLANNER:
${plan}

CONTEXT FROM EXPLORER:
${context}

YOUR TASK:
1. Implement the test cases as specified in the plan
2. Follow the existing testing patterns in the codebase
3. Use the same mocking strategies as other tests
4. Create comprehensive tests that cover:
   - Happy path scenarios
   - Error handling paths
   - Edge cases
   - All branches and conditions

Use the createFile tool to create new test files or applyGitPatch to modify existing ones.
Use the command tool to verify the files are correctly created.

After implementing, provide a summary of what you created/modified.`,
  });

  await response.consumeStream();
  console.log("\n✅ [CODER] Implementation complete\n");
  return response.text;
};

/**
 * Tester Agent - Runs tests and validates coverage
 */
const runTesterAgent = async (coderSummary: string): Promise<string> => {
  console.log("\n🧪 [TESTER AGENT] Running tests and validating coverage...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(15),
    tools: testerTools,
    prompt: `You are a QA and testing expert agent. Your job is to validate the implemented changes.

CODEBASE PATH: ${config.codebasePath}

CHANGES MADE BY CODER:
${coderSummary}

YOUR TASK:
1. Run the test suite for the modified/created test files
2. Check if tests pass
3. Verify code coverage for the target file
4. Identify any failing tests or coverage gaps

Use commands like:
- npm test or yarn test or bun test (check package.json for the correct command)
- npm run coverage or similar coverage commands
- Check the coverage report output

Report:
- Test results (pass/fail)
- Coverage percentage achieved
- Any remaining uncovered lines
- Suggestions for fixes if tests fail`,
  });

  await response.consumeStream();
  console.log("\n📊 [TESTER] Validation complete\n");
  return response.text;
};

/**
 * Reviewer Agent - Reviews code quality and provides final assessment
 */
const runReviewerAgent = async (
  plan: string,
  coderSummary: string,
  testResults: string
): Promise<string> => {
  console.log("\n👀 [REVIEWER AGENT] Performing code review...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(10),
    tools: explorerTools,
    prompt: `You are a senior code reviewer agent. Your job is to review the changes made.

CODEBASE PATH: ${config.codebasePath}

ORIGINAL PLAN:
${plan}

CODER'S CHANGES:
${coderSummary}

TEST RESULTS:
${testResults}

##How to test the codebase?
You can run the command npm run run_tests to test the full codebase.
Or
npm run run_tests file_path to test a specific file.

YOUR TASK:
1. Review the test files that were created/modified
2. Check if they follow best practices
3. Verify the tests are meaningful and not just coverage padding
4. Ensure proper assertions are used
5. Check for any code smells or issues

Provide:
- Overall assessment (APPROVED / NEEDS CHANGES)
- Specific feedback on the implementation
- Any improvements needed
- Final summary of the work done`,
  });

  await response.consumeStream();
  console.log("\n✨ [REVIEWER] Review complete\n");
  return response.text;
};

/**
 * Fixer Agent - Fixes any issues found by tester or reviewer
 */
const runFixerAgent = async (
  issues: string,
  context: string
): Promise<string> => {
  console.log("\n🔧 [FIXER AGENT] Addressing issues...\n");

  const response = streamText({
    model: openrouter(config.modelName),
    stopWhen: stepCountIs(20),
    tools: coderTools,
    prompt: `You are a debugging and fixing expert agent. Your job is to fix issues.

CODEBASE PATH: ${config.codebasePath}

ISSUES TO FIX:
${issues}

CONTEXT:
${context}

YOUR TASK:
1. Analyze the failing tests or coverage issues
2. Fix the test files to address the problems
3. Ensure tests pass after fixes
4. Maintain or improve code coverage

Use applyGitPatch or createFile to make fixes.
Provide a summary of fixes made.`,
  });

  await response.consumeStream();
  console.log("\n🔧 [FIXER] Fixes applied\n");
  return response.text;
};

// ============= MAIN ORCHESTRATOR =============

export const main = async () => {
  console.log("═".repeat(60));
  console.log("🚀 MULTI-AGENT CODE IMPROVEMENT SYSTEM");
  console.log("═".repeat(60));
  console.log(`\n📁 Target: ${config.codebasePath}`);
  console.log(`📝 Task: ${config.description.slice(0, 100)}...\n`);
  console.log("═".repeat(60));

  try {
    // Phase 1: Planning
    const plan = await runPlannerAgent();

    // Phase 2: Deep Exploration
    const context = await runExplorerAgent(plan);

    // Phase 3: Implementation
    let coderSummary = await runCoderAgent(plan, context);

    // Phase 4: Testing & Validation
    let testResults = await runTesterAgent(coderSummary);

    // Phase 5: Review
    let review = await runReviewerAgent(plan, coderSummary, testResults);

    // Phase 6: Fix loop (if needed) - max 2 iterations
    let iterations = 0;
    const maxIterations = 2;

    while (
      review.toLowerCase().includes("needs changes") &&
      iterations < maxIterations
    ) {
      console.log(`\n🔄 [ORCHESTRATOR] Iteration ${iterations + 1}: Fixing issues...\n`);

      const fixerSummary = await runFixerAgent(review, context);
      testResults = await runTesterAgent(fixerSummary);
      coderSummary = fixerSummary;
      review = await runReviewerAgent(plan, coderSummary, testResults);

      iterations++;
    }

    // Final Summary
    console.log("\n" + "═".repeat(60));
    console.log("📋 FINAL SUMMARY");
    console.log("═".repeat(60));

    const finalSummary = `
MULTI-AGENT EXECUTION COMPLETE
==============================

📋 PLAN:
${plan}

💻 IMPLEMENTATION:
${coderSummary}

🧪 TEST RESULTS:
${testResults}

👀 REVIEW:
${review}

${iterations > 0 ? `🔄 Fix iterations: ${iterations}` : "✅ No fixes needed"}
`;

    console.log(finalSummary);
    return finalSummary;
  } catch (error) {
    console.error("\n❌ [ORCHESTRATOR] Error:", error);
    throw error;
  }
};

