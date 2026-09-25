import { performance } from "perf_hooks";
import { createChatModel, isOllama } from "./llm";
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
  AIMessageChunk,
} from "@langchain/core/messages";
import {
  ListEventsCommand,
  RetrieveMemoryRecordsCommand,
  CreateEventCommand,
  BedrockAgentCoreClient,
  Role,
} from "@aws-sdk/client-bedrock-agentcore";
import { findWorkOrdersTool } from "./tools/findWorkOrdersTool";
import { createWorkOrderTool } from "./tools/createWorkOrderTool";
import { queryKnowledgeBaseTool } from "./tools/queryKnowledgeBaseTool";
import * as crypto from "crypto";
import { buildSystemInstruction } from "./constants/prompts";
import { webSearchGroundingTool } from "./tools/webSearchTool";
import { MAX_MEMORY_TEXT_BYTES, truncateUtf8 } from "./utils";

// ---------------------------
// Timing Utility
// ---------------------------
const logStep = (label: string, start: number) => {
  const duration = performance.now() - start;
  console.log(`⏱️  ${label} took ${duration.toFixed(2)} ms`);
  return duration;
};

/** Per-token chunk logging. Noisy by design — off unless explicitly asked for. */
const DEBUG_TOKENS = process.env.DEBUG_AGENT_TOKENS === "1";

/** Per tool result, in the history kept for later turns. */
const TOOL_RESULT_MEMORY_CHARS = 2000;

/**
 * Gemini and Ollama both stream `content` as a string OR as an array of parts,
 * so anything that reads it for logging has to flatten it rather than cast.
 */
const textOf = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return (content as any[])
      .map((part) =>
        typeof part === "string"
          ? part
          : part && typeof part.text === "string"
            ? part.text
            : ""
      )
      .join("");
  }
  return "";
};

// --- Graph Setup ---
const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  summaries: Annotation<string[]>(),
});

// --- Singletons ---
const tools = [
  findWorkOrdersTool,
  createWorkOrderTool,
  queryKnowledgeBaseTool,
  // webSearchGroundingTool calls the Gemini API directly (grounding), so it is
  // still rate-limited even when the chat model runs on Ollama. Drop it while
  // testing locally.
  ...(isOllama ? [] : [webSearchGroundingTool]),
];
const toolNode = new ToolNode(tools);

// gemini-2.5-flash returns EMPTY CANDIDATES (finishReason STOP, 0 output tokens,
// no text and no function call) for this system prompt + tool set — reproducibly,
// and for ordinary queries, not just edge cases. Trimming the prompt flips the
// behaviour unpredictably, so it is not one offending phrase to remove. 3-flash
// answers every one of the same prompts correctly, and is what index.ts already
// uses. 
const model = createChatModel({
  model: "gemini-3-flash-preview",
  streaming: true,
  temperature: 0.3,
}).bindTools!(tools);

// ---------------------------
// Model Call With Timing
// ---------------------------
const callModel = async (
  state: typeof StateAnnotation.State,
  config?: any
) => {
  const start = performance.now();
  console.log("🚀 Agent model streaming started");

  // Gemini regularly emits a COMPLETE answer alongside a tool call, so the graph
  // loops back through `tools` and this node answers a second time. Every pass
  // reaches the user through `onToken`, with nothing separating them, so one
  // bubble ended up holding two answers glued together mid-token
  // ("…Modules%20(2).pdfA **SIGA-CR** is a…"). That also broke the SOURCES
  // parser in web-front, which read the second answer as part of the first
  // answer's S3 URL and produced a link that 403s.
  //
  // Text from an earlier pass of THIS turn is superseded by what follows, so
  // tell the client to drop what it has rather than appending to it. Scoped
  // after the last human message: earlier turns are history, not this answer.
  const lastHumanIndex = state.messages
    .map((m) => m.getType())
    .lastIndexOf("human");
  const hasSupersededText = state.messages
    .slice(lastHumanIndex + 1)
    .some((m) => m.getType() === "ai" && textOf(m.content).trim().length > 0);

  if (hasSupersededText) {
    console.log("↩️ Resetting stream: an earlier agent pass already sent text");
    config?.configurable?.onReset?.();
  }

  const stream = await model.stream(
    // Built per call, not once at import: the prompt carries today's date and
    // this process stays up for days.
    [new SystemMessage(buildSystemInstruction(new Date())), ...state.messages],
    config
  );

  let finalContent = "";
  let finalToolCalls: any[] | undefined;

  for await (const chunk of stream) {
    const rawContent = (chunk as any).content;
    if (DEBUG_TOKENS) console.log("rawContent- ", rawContent);

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
    actionToken,
    onToken,
    onReset,
  }: {
    memoryClient: BedrockAgentCoreClient;
    memoryId: string;
    actor_id: string;
    session_id: string;
    organizationId: string;
    /** Short-lived credential minted by web-back; the work order tool presents it. */
    actionToken?: string;
    onToken?: (token: string) => void;
    /**
     * Fired when a later agent pass supersedes text already streamed for this
     * turn. The client must DISCARD what it has rather than append — see the
     * comment in `callModel`.
     */
    onReset?: () => void;
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

  // ListEvents returns the NEWEST event first. Fed in that order, the model read
  // the conversation backwards: a "yes" landed after whichever question was
  // oldest, so it confirmed an assignee instead of the discard it had just
  // asked about, and a confirmed discard was asked for again forever.
  const orderedEvents = [...(historyResponse.events || [])].sort(
    (a: any, b: any) =>
      new Date(a.eventTimestamp).getTime() - new Date(b.eventTimestamp).getTime()
  );

  const historyMessages: BaseMessage[] = orderedEvents.flatMap(
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

  let finalContent = "";
  let lastToolCall: any = null;
  /**
   * Every character the agent node emitted that has not been superseded.
   *
   * `finalContent` is wiped whenever a chunk carries a tool call, so that
   * "let me look that up…" does not get persisted as the answer. But the user
   * has ALREADY SEEN that text — `onToken` streams all agent text over SSE as it
   * is produced — and if the run ends on a tool-call chunk the wipe leaves
   * nothing at all. That is how an empty assistant entry reached Bedrock, which
   * rejects the whole event and took the user's own message down with it.
   *
   * A reset is the one case where the user has seen text that must NOT be kept:
   * the client is told to drop it, so persisting it would put an answer in
   * memory that no longer matches the bubble.
   */
  let streamedText = "";

  /**
   * What each tool returned this turn, keyed by tool call so a message the stream
   * emits twice is kept once. Persisted so the next turn can see it: a tool that
   * answers "ask the user, then call again with confirmDiscard: true" is useless
   * if all the model later remembers is that the tool was called.
   */
  const toolResults = new Map<string, { name: string; text: string }>();

  // Keep the persistence view in step with the client's. Both buffers hold text
  // the next pass is about to replace.
  const handleReset = () => {
    finalContent = "";
    streamedText = "";
    onReset?.();
  };

  const stream = await app.stream(initialState, {
    configurable: {
      thread_id,
      onToken: onToken,
      onReset: handleReset,
      actionToken,
    },
    metadata: { onToken },
    streamMode: "messages",
  });

  // `streamMode: "messages"` yields one tuple per TOKEN CHUNK, not per node —
  // so a node name on a tuple says which node produced that token, nothing more.
  // Logging inside the loop therefore printed one "node executed" line per token
  // and timed the loop body (microseconds) instead of the node. Track the
  // transitions instead: the log below emits one line per real node entry, and
  // the elapsed time it reports is the node's actual wall-clock cost.
  let currentNode: string | undefined;
  let nodeStart = performance.now();
  let nodeTokens = 0;
  let step = 0;

  const closeNode = () => {
    if (!currentNode) return;
    const ms = performance.now() - nodeStart;
    console.log(
      `⬅️ Node ${currentNode} finished in ${ms.toFixed(2)} ms (${nodeTokens} chunk${nodeTokens === 1 ? "" : "s"})`
    );
  };

  for await (const [message, metadata] of stream) {
    const node: string | undefined = metadata?.langgraph_node;

    if (node && node !== currentNode) {
      closeNode();
      step += 1;
      console.log(`➡️ [step ${step}] Node entered: ${node}`);
      currentNode = node;
      nodeStart = performance.now();
      nodeTokens = 0;
    }
    nodeTokens += 1;

    if (node === "agent") {
      // Tokens arrive as AIMessageChunks. When the node finishes, LangGraph also
      // emits the AIMessage it returned — whole, because it has no id to dedupe
      // on — so counting that too stored every answer twice, back to back.
      if (AIMessageChunk.isInstance(message)) {
        const delta = textOf(message.content);
        finalContent += delta;
        streamedText += delta;
      }

      if ((message as AIMessage).tool_calls?.length) {
        lastToolCall = (message as AIMessage).tool_calls;
        for (const call of lastToolCall) {
          console.log(
            `🛠️ Tool requested: ${call.name} ${JSON.stringify(call.args)}`
          );
        }
        finalContent = "";
      }
    }

    if (node === "tools") {
      // The tool's own return value. Nothing used to log it, which is why a
      // PREVIEW, a NEEDS CLARIFICATION and an outright failure were
      // indistinguishable in the logs after the fact.
      const result = textOf(message.content);
      console.log(
        `🧰 Tool result: ${(message as any).name} → ${result.slice(0, 600)}${result.length > 600 ? " …[truncated]" : ""}`
      );
      const key = (message as any).tool_call_id ?? `${(message as any).name}:${result}`;
      toolResults.set(key, { name: (message as any).name, text: result });
    }
  }
  closeNode();

  logStep("🌊 Total streaming", streamStart);
  console.log(`🧭 Graph path: ${step} node execution${step === 1 ? "" : "s"}`);

  /**
   * A turn that produced NO text and NO tool call is a dead end, and nothing
   * throws on the way there: the request succeeds, SSE closes, and the user is
   * left looking at an empty bubble. Gemini does exactly this — a candidate with
   * finishReason STOP and zero output tokens — and this emptiness is the only
   * signal it happened. Say something rather than nothing.
   *
   * Streamed to the user but deliberately NOT persisted: a canned apology in the
   * history teaches the model nothing and would be read back as a real answer.
   */
  let fallbackText = "";
  if (streamedText.trim().length === 0 && !lastToolCall) {
    fallbackText =
      "I wasn't able to put together an answer for that. Could you rephrase it and try again?";
    console.warn(
      "⚠️ Empty model turn (no text, no tool call) — streaming a fallback reply"
    );
    onToken?.(fallbackText);
  }

  // ---------------------------
  // 4. Persistence
  // ---------------------------
  const persistStart = performance.now();
  console.log("💾 Persistence started");

  // Bedrock rejects any payload entry whose text is empty — and it rejects the
  // WHOLE event, so one blank assistant reply used to lose the user's message
  // too and fail a request whose answer had already been streamed to them.
  // Build the entries, then drop the blanks.
  // Prefer the post-tool answer; fall back to everything the user was shown.
  const assistantText = finalContent.trim().length > 0 ? finalContent : streamedText;

  // Results are capped so one large search result does not crowd out the
  // conversation in the 10 events read back next turn.
  const toolText =
    toolResults.size > 0
      ? [...toolResults.values()]
          .map(
            ({ name, text }) =>
              `Used Tool: ${name}\nResult:\n${text.length > TOOL_RESULT_MEMORY_CHARS ? `${text.slice(0, TOOL_RESULT_MEMORY_CHARS)} …[truncated]` : text}`
          )
          .join("\n\n")
      : lastToolCall
        ? `Used Tools: ${lastToolCall.map((t: any) => t.name).join(", ")}`
        : "";

  const memoryPayload = [
    { role: Role.USER, text: userQuery },
    ...(toolText ? [{ role: Role.TOOL, text: toolText }] : []),
    { role: Role.ASSISTANT, text: assistantText },
  ]
    .filter((entry) => typeof entry.text === "string" && entry.text.trim().length > 0)
    // Oversize entries are rejected the same way — whole event — so cap them.
    .map((entry) => ({
      conversational: {
        role: entry.role,
        content: { text: truncateUtf8(entry.text, MAX_MEMORY_TEXT_BYTES) },
      },
    }));

  if (finalContent.trim().length === 0 && assistantText.trim().length > 0) {
    console.warn(
      "⚠️ Run ended on a tool call; persisting the text the user was actually shown"
    );
  } else if (assistantText.trim().length === 0) {
    // The turn produced no text at all. Persist it without an assistant entry
    // rather than losing the user's message to a rejected event.
    console.warn(
      "⚠️ No assistant text for this turn; persisting the turn without an assistant entry"
    );
  }

  if (memoryPayload.length > 0) {
    try {
      await memoryClient.send(
        new CreateEventCommand({
          memoryId,
          actorId: actor_id,
          sessionId: session_id,
          eventTimestamp: new Date(),
          clientToken: crypto.randomUUID(),
          payload: memoryPayload,
        })
      );
    } catch (err) {
      // The user already has the full answer over SSE. Losing the memory write
      // costs continuity on the next turn; failing the request here would throw
      // away a reply they have already read.
      console.error("💾 Persistence failed; continuing without it:", err);
    }
  }

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

  return assistantText.trim().length > 0 ? assistantText : fallbackText;
}