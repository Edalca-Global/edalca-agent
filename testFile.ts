import { performance } from "perf_hooks";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  Annotation,
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  BaseMessage,
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import {
  ListEventsCommand,
  RetrieveMemoryRecordsCommand,
  CreateEventCommand,
  BedrockAgentCoreClient,
  Role,
} from "@aws-sdk/client-bedrock-agentcore";
import { fetchWorkOrderTool } from "./tools/fetchWorkOrderTool";
import { queryKnowledgeBaseTool } from "./tools/queryKnowledgeBaseTool";
import * as crypto from "crypto";
import { SYSTEM_INSTRUCTION } from "./constants/prompts";
import { webSearchGroundingTool } from "./tools/webSearchTool";

// ---------------------------
// Timing Utility
// ---------------------------
const logStep = (label: string, start: number) => {
  const duration = performance.now() - start;
  console.log(`⏱️  ${label} took ${duration.toFixed(2)} ms`);
  return duration;
};

// --- Graph Setup ---
const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  summaries: Annotation<string[]>(),
});

// --- Singletons ---
const tools = [fetchWorkOrderTool, queryKnowledgeBaseTool, webSearchGroundingTool];
const toolNode = new ToolNode(tools);

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GEMINI_API_KEY,
  streaming: true,
  temperature: 0.3,
}).bindTools(tools);

// ---------------------------
// Model Call With Timing
// ---------------------------
const callModel = async (
  state: typeof StateAnnotation.State,
  config?: any
) => {
  const start = performance.now();
  console.log("🚀 Agent model streaming started");

  const stream = await model.stream(
    [new SystemMessage(SYSTEM_INSTRUCTION), ...state.messages],
    config
  );

  let finalContent = "";
  let finalToolCalls: any[] | undefined;

  for await (const chunk of stream) {
    const rawContent = (chunk as any).content;
    console.log('rawContent- ', rawContent);
    
    let textDelta = "";

    // Extract ONLY human-readable text, ignore function/tool call payloads
    if (typeof rawContent === "string") {
      textDelta = rawContent;
    } else if (Array.isArray(rawContent)) {
      for (const part of rawContent as any[]) {
        if (typeof part === "string") {
          textDelta += part;
        } else if (part && typeof part === "object") {
          // Common Gemini shapes: { type: "text", text: "..." } or similar
          if (
            typeof part.text === "string" &&
            // Explicitly skip functionCall/tool call chunks
            !part.functionCall &&
            part.type !== "functionCall" &&
            part.type !== "tool" &&
            part.type !== "toolCall"
          ) {
            textDelta += part.text;
          }
        }
      }
    }

    if (textDelta) {
      finalContent += textDelta;

      // 🔥 TRUE TOKEN STREAMING (text only)
      const onToken = config?.configurable?.onToken;
      if (onToken) {
        onToken(textDelta);
      }
    }

    // Preserve tool calls if present
    if ((chunk as AIMessage).tool_calls?.length) {
      finalToolCalls = (chunk as AIMessage).tool_calls;
    }
  }

  logStep("🧠 Model stream complete", start);

  return {
    messages: [
      new AIMessage({
        content: finalContent,
        tool_calls: finalToolCalls,
      }),
    ],
  };
};

// ---------------------------
// Graph Compile Timing
// ---------------------------
const compileStart = performance.now();

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
    return lastMsg.tool_calls?.length ? "tools" : END;
  })
  .addEdge("tools", "agent");

const app = workflow.compile();

logStep("📦 Graph compilation", compileStart);

// ---------------------------
// MAIN EXECUTION
// ---------------------------
export async function runWorkOrderAgent(
  userQuery: string,
  thread_id: string,
  {
    memoryClient,
    memoryId,
    actor_id,
    session_id,
    organizationId,
    onToken,
  }: {
    memoryClient: BedrockAgentCoreClient;
    memoryId: string;
    actor_id: string;
    session_id: string;
    organizationId: string;
    onToken?: (token: string) => void;
  }
) {
  const totalStart = performance.now();
  console.log("=================================================");
  console.log("🔥 runWorkOrderAgent START");
  console.log("=================================================");

  // ---------------------------
  // 1. Context Fetching
  // ---------------------------
  const memoryStart = performance.now();
  console.log("📥 Memory fetch started");

  const [historyResponse, summariesResponse] = await Promise.all([
    memoryClient.send(
      new ListEventsCommand({
        memoryId,
        actorId: actor_id,
        sessionId: session_id,
        maxResults: 10,
      })
    ),
    memoryClient.send(
      new RetrieveMemoryRecordsCommand({
        memoryId,
        namespace: `/summaries/${actor_id}/${session_id}`,
        searchCriteria: { searchQuery: userQuery, topK: 5 },
      })
    ),
  ]);

  logStep("📥 Memory fetch (history + summaries)", memoryStart);

  // ---------------------------
  // 2. Message Processing
  // ---------------------------
  const processStart = performance.now();

  const historyMessages: BaseMessage[] = (historyResponse.events || []).flatMap(
    (event: any) => {
      return (event.payload || []).map((p: any) => {
        const content = p.conversational?.content?.text || "";
        return p.conversational?.role === "USER"
          ? new HumanMessage(content)
          : new AIMessage(content);
      });
    }
  );

  const summaryContext: string[] = (
    summariesResponse.memoryRecordSummaries || []
  ).map((record: any) => record.content?.text || "");

  logStep("🧱 History + summary processing", processStart);

  // ---------------------------
  // 3. Streaming Execution
  // ---------------------------
  const streamStart = performance.now();
  console.log("🌊 Streaming started");

  const initialState = {
    messages: [...historyMessages, new HumanMessage(userQuery)],
    summaries: summaryContext,
  };

  const stream = await app.stream(initialState, {
    configurable: { thread_id, onToken: onToken },
    metadata: { onToken },
    streamMode: "messages",
  });

  let finalContent = "";
  let lastToolCall: any = null;

  for await (const [message, metadata] of stream) {
    const nodeStart = performance.now();

    if (metadata?.langgraph_node) {
      console.log(`➡️ Node executed: ${metadata.langgraph_node}`);
    }

    if (metadata.langgraph_node === "agent") {
      const content = message.content as string;

      // if (content && onToken) {
      //   onToken(content);
      // }

      finalContent += content;

      if ((message as AIMessage).tool_calls?.length) {
        lastToolCall = (message as AIMessage).tool_calls;
        console.log(
          `🛠️ Tool requested: ${lastToolCall
            .map((t: any) => t.name)
            .join(", ")}`
        );
      }
    }

    logStep(`⏳ Node ${metadata?.langgraph_node}`, nodeStart);
  }

  logStep("🌊 Total streaming", streamStart);

  // ---------------------------
  // 4. Persistence
  // ---------------------------
  const persistStart = performance.now();
  console.log("💾 Persistence started");

  await memoryClient.send(
    new CreateEventCommand({
      memoryId,
      actorId: actor_id,
      sessionId: session_id,
      eventTimestamp: new Date(),
      clientToken: crypto.randomUUID(),
      payload: [
        { conversational: { role: Role.USER, content: { text: userQuery } } },
        ...(lastToolCall
          ? [
              {
                conversational: {
                  role: Role.TOOL,
                  content: {
                    text: `Used Tools: ${lastToolCall
                      .map((t: any) => t.name)
                      .join(", ")}`,
                  },
                },
              },
            ]
          : []),
        {
          conversational: {
            role: Role.ASSISTANT,
            content: { text: finalContent },
          },
        },
      ],
    })
  );

  logStep("💾 Persistence", persistStart);

  // ---------------------------
  // TOTAL TIME
  // ---------------------------
  const totalDuration = logStep("🔥 TOTAL runWorkOrderAgent", totalStart);

  console.log("=================================================");
  console.log("✅ FINAL RESULT");
  console.log("Final Content Length:", finalContent.length);
  console.log("Last Tool Call:", lastToolCall);
  console.log(`🏆 TOTAL TIME: ${totalDuration.toFixed(2)} ms`);
  console.log("=================================================");

  return finalContent;
}