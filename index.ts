// agentTemplate.ts

import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
} from "@langchain/core/prompts";
import { addMessages, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { Annotation } from "@langchain/langgraph";
import { fetchWorkOrderTool } from "./tools/fetchWorkOrderTool";
import {
  BedrockAgentCoreClient,
  CreateEventCommand,
  PayloadType,
  RetrieveMemoryRecordsCommand,
} from "@aws-sdk/client-bedrock-agentcore";
import { ListEventsCommand } from "@aws-sdk/client-bedrock-agentcore";
import { cleanPastMessagesAfterReset, extractSourcesFromMessages } from "./utils";
import { queryKnowledgeBaseTool } from "./tools/queryKnowledgeBaseTool";
import { SYSTEM_PROMPT } from "./constants/prompts";
import { createChatModel } from "./llm";

function convertEventToMessages(event: any): BaseMessage[] {
  console.log('event', event);
  
  if (!Array.isArray(event?.payload)) return [];

return event.payload.map((item: any) => {
  console.log('item', item);
  
    const text = item.text;
    const role = item.role; // e.g., "USER" or "ASSISTANT"

    if (role === "USER") {
      return new HumanMessage({ content: text, additional_kwargs: { role: "user" } });
    }
    if (role === "ASSISTANT") {
      return new AIMessage({ content: text, additional_kwargs: { role: "model" } });
    }
    return new HumanMessage(text);
  });
}
// function convertEventToMessages(event: any): BaseMessage[] {
//   const payload = event?.payload;
//   if (!Array.isArray(payload)) return [];

//   return payload.reduce((acc: BaseMessage[], item: any) => {
//     const conv = item?.Conversational ?? item?.conversational;
//     const text = conv?.Content?.Text ?? conv?.content?.text;
//     const role = conv?.Role ?? conv?.role;

//     if (text) {
//       if (role === "ASSISTANT") acc.push(new AIMessage(text));
//       else acc.push(new HumanMessage(text));
//     }
//     return acc;
//   }, []);
// }

async function fetchConversationHistory(
  memoryClient: BedrockAgentCoreClient,
  memoryId: string,
  session_id: string,
  actor_id: string
): Promise<BaseMessage[]> {
  const command = new ListEventsCommand({
    memoryId: memoryId,
    sessionId: session_id,
    actorId: actor_id,
    // maxResults: 7,
  });

  const response = await memoryClient.send(command);
  const events = response?.events || [];
  return events.flatMap((event: any) =>
    convertEventToMessages(event)
  );
}

async function fetchConversationSummary(
  userQuery: string,
  memoryClient: BedrockAgentCoreClient,
  memoryId: string,
  session_id: string,
  actor_id: string
): Promise<BaseMessage[]> {
  const command = new RetrieveMemoryRecordsCommand({
    memoryId: memoryId,
    namespace: `/summaries/${actor_id}/${session_id}`,
    searchCriteria: {
        "searchQuery": userQuery,
        "topK": 5
    },
  });

  const response = await memoryClient.send(command);
  const memoryRecordSummaries = response?.memoryRecordSummaries || [];
  
  return memoryRecordSummaries.map((record: any) =>
    record?.content?.text
  );
}

// Add this helper to your agentTemplate.ts
function sanitizeMessages(messages: BaseMessage[]): BaseMessage[] {
  return messages.map((msg: any) => {
    // Force role mapping for Gemini
    if (msg.getType() === "human") return new HumanMessage(msg.content);
    if (msg.getType() === "ai") return new AIMessage(msg.content);
    
    // Gemini handles System messages better if they are converted to Human messages
    // or passed via specific configuration. LangChain's latest version handles this, 
    // but explicit conversion is safer.
    if (msg.getType() === "system") {
      return new HumanMessage({ content: msg.content });
    }
    
    return msg;
  });
}

// ---------------------------
// Define Agent State
// ---------------------------
const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
  reducer: addMessages, // This is the gold standard for LangGraph
}),
  intents: Annotation<string[]>(),
  loopCount: Annotation<number>({
    reducer: (x, y) => (y ?? 0), // We will set the absolute value
    default: () => 0,
  }),
  summaries: Annotation<BaseMessage[]>(), // Added so summaries aren't lost
onToken: Annotation<((token: string) => void) | undefined>({
  reducer: (x, y) => y ?? x, // Keep the existing callback
}),});

// ---------------------------
// Tools
// ---------------------------
const tools = [fetchWorkOrderTool, queryKnowledgeBaseTool];
const toolNode = new ToolNode<typeof GraphState.State>(tools);

// ---------------------------
// Chat Model
// ---------------------------
// ---------------------------
// Model Function
// ---------------------------
// 1. Initialize the model
// Use the official LangChain class - it handles Zod schemas automatically
const model = createChatModel({
  model: 'gemini-3-flash-preview', // or "gemini-1.5-flash"
  temperature: 0,
}).bindTools!(tools);

// 2. Updated callModel node
async function callModel(state: typeof GraphState.State) {
  state.messages.forEach((m, i) => {
    console.log(`Msg ${i}: Type=${m.getType()}, Role=${(m as any).role}, Content=${m.content.toString().slice(0,20)}`);
  });
  
  const cleanMessages = sanitizeMessages(state.messages);

  // Build the full message list with system message first
  const fullMessages = [
    new SystemMessage({ content: SYSTEM_PROMPT, additional_kwargs: { role: "system" } }),
    new SystemMessage({ content: `Historical Context: ${state.summaries?.join("\n") || "None"}`, additional_kwargs: { role: "system" } }),
    ...cleanMessages
  ];

  // Create prompt with all messages explicitly
  const prompt = ChatPromptTemplate.fromMessages(fullMessages);
  const chain = prompt.pipe(model);

  // Invoke with no additional messages since they're all in the prompt
  const response = await chain.invoke({});

  return {
    messages: [response],
    loopCount: state.loopCount + 1,
  };
}

// async function callModel(state: typeof GraphState.State) {
//   const modelStartTime = Date.now();
//   console.log(`\n🧠 [Model] Invoking LLM with ${state.messages.length} messages...`);

//   const prompt = ChatPromptTemplate.fromMessages([
//     ["system", SYSTEM_PROMPT],
//     "Conversation summary for Context:\n{summary}",
//     new MessagesPlaceholder("messages"),
//   ]);

//  const formattedPrompt = await prompt.formatMessages({
//     summary: state.summaries?.join("\n") || "No previous context.",
//     messages: state.messages,
//   });

//   const stream = await chatModel.stream(formattedPrompt);
//    // We use a "Gather" message to collect all parts of the stream
//   let gatheringMessage: any = null;
//   let emittedSearch = false;
//   // This variable stays empty if ONLY tool calls happen, 
//   // ensuring no "Searching..." text gets saved.
//   let fullText = "";
// for await (const chunk of stream) {
//   console.log('gathering', gatheringMessage, 'chunk', chunk.candidates);
  
//   // 1. SAFE ACCUMULATION
//   if (gatheringMessage === null) {
//     gatheringMessage = chunk.candidates[0].content.parts[0].text;
//   } else {
//     // Check if concat exists, otherwise manually merge or re-assign
//     if (typeof gatheringMessage.concat === "function") {
//       gatheringMessage = gatheringMessage.concat(chunk.candidates[0].content.parts[0].text);
//     } else {
//       // Fallback for non-chunk messages (rare but happens with some adapters)
//       gatheringMessage = chunk.candidates[0].content.parts[0].text; 
//     }
//   }

//     // 2. UI Feedback: Show "Searching" ONLY when the model officially initiates a tool call
//     // We check tool_call_chunks because it's populated during the stream
//     const hasToolChunks = gatheringMessage && gatheringMessage.tool_call_chunks && gatheringMessage.tool_call_chunks.length > 0;
    
//     if (hasToolChunks && !emittedSearch) {
//       state.onToken?.("🔍 *Searching internal database...* \n\n");
//       emittedSearch = true;
//     }

//     // 3. UI Feedback: Stream tokens to user
//     if (chunk.content) {
//       const token = typeof chunk.content === "string" ? chunk.content : "";
      
//       // SAFETY: Only add to fullText and emit if it's NOT a technical 'Calling tools' string.
//       // If we are currently calling a tool, we skip adding this technical text to the state.
//       if (!hasToolChunks) {
//         fullText += token; 
//         state.onToken?.(token);
//       }
//     }
//   }

//   const finalMessage = gatheringMessage || new AIMessage("");
//   // We override the content with our 'fullText' to ensure technical 
//   // 'Calling tools:' strings are never saved in the message history.
//   finalMessage.content = fullText;

//   console.log(`⏱️ [Model] LLM response received in ${Date.now() - modelStartTime}ms`);

//   // --- Intent Logic ---
//   let detectedIntents = state.intents || [];
//   if (finalMessage.tool_calls && finalMessage.tool_calls.length > 0) {
//     const toolIntents = finalMessage.tool_calls.map((tc: any) => {
//       if (tc.name === "query_documents_kb") return "MODULE_1_INTERNAL_DOCUMENTS";
//       if (tc.name === "fetch_workOrder_tool") return "MODULE_2_WORK_ORDER_SEARCH";
//       return "Use Your Knowledge";
//     });
//     detectedIntents = [...new Set([...detectedIntents, ...toolIntents])];
//   } else if (state.loopCount === 0) {
//     detectedIntents = ["UNSUPPORTED_OR_NO_TOOL"];
//   }

//   return {
//     // Return the final complete message (the reducer handles appending)
//     messages: [finalMessage], 
//     intents: detectedIntents,
//     loopCount: state.loopCount + 1
//   };
// }

// ---------------------------
// Routing Logic
// ---------------------------
// const intentModel = new ChatOpenAI({
//   model: "gpt-4o-mini-2024-07-18",
//   temperature: 0,
// });

// const intentSchema = {
//   name: "intent_classifier",
//   schema: {
//     type: "object",
//     properties: {
//       intents: {
//         type: "array",
//         items: {
//           type: "string",
//           enum: [
//             "GREETINGS",
//             "MODULE_1_GENERAL_KNOWLEDGE",
//             "MODULE_2_INTERNAL_DOCUMENTS",
//             "MODULE_3_WORK_ORDER_SEARCH",
//             "MODULE_4_UNSUPPORTED"
//           ]
//         },
//         minItems: 1,
//         uniqueItems: true
//       }
//     },
//     required: ["intents"],
//     additionalProperties: false
//   }
// };

// const intentPrompt = ChatPromptTemplate.fromMessages([
//   [
//     "system",
//     `
// You classify user requests into one of the following intents:
// If the user is greeting, thanking, or saying goodbye, classify as GREETINGS.

// - MODULE_1_GENERAL_KNOWLEDGE
// - MODULE_2_INTERNAL_DOCUMENTS
// - MODULE_3_WORK_ORDER_SEARCH
// - MODULE_4_UNSUPPORTED

// If the request contains:
// - a general definition AND
// - company-specific, internal, or comparative language such as
//   "our", "ours", "company", "internal", "this system"

// Return BOTH MODULE_1_GENERAL_KNOWLEDGE and MODULE_2_INTERNAL_DOCUMENTS.

// Return ONLY valid JSON that matches the provided schema.
// Do not answer the user.
// `
//   ],
//   ["human", "{input}"]
// ]);

// async function intentClassifier(state: typeof GraphState.State) {
//   const lastUserMessage =
//     state.messages[state.messages.length - 1].content.toString();

//   console.log("[IntentClassifier] User message:", lastUserMessage);

//   const response = await intentModel.invoke(
//     await intentPrompt.formatMessages({ input: lastUserMessage }),
//     { response_format: { type: "json_schema", json_schema: intentSchema } }
//   );

//   console.log("[IntentClassifier] Raw model response:", response.content);

//   const parsed = JSON.parse(response.content.toString());

// const intents = parsed.intents; // string[]

//   console.log("[IntentClassifier] Parsed intents:", intents);
// return { intents };

// }

// function routeByIntent(state: typeof GraphState.State) {
//   const intents = state.intents;

//   if (intents.includes("GREETINGS")) return "greetings";

//   if (intents.includes("MODULE_4_UNSUPPORTED")) return "unsupported";

//   return "agent"; // agent will decide which tools to call
// }


// async function greetingsNode() {
//   console.log("[Node] greetingsNode invoked");
//   return {
//     messages: [
//       new AIMessage("Hello! I'm Edalca AI. How can I assist you today?")
//     ]
//   };
// }

// async function unsupportedNode() {
//   console.log("[Node] unsupportedNode invoked");
//   return {
//     messages: [
//       new AIMessage("This capability is not yet available.")
//     ]
//   };
// }

// async function validator(state: typeof GraphState.State) {
//   const messages = state.messages;
  
//   // Check if ANY message in the history is a ToolMessage 
//   // or contains the metadata/sources you require.
//   const hasToolContext = messages.some(m => m.getType() === "tool" || m.additional_kwargs?.sources);
//   const sources = extractSourcesFromMessages(messages);

//   if (!sources && !hasToolContext) {
//       return {
//         messages: [new AIMessage("I'm sorry, I don't have that in my records.")]
//       };
//   }
//   return state;
// }

async function validator(state: typeof GraphState.State) {
  // 1. Check for literal Tool Messages
  const hasToolMessage = state.messages.some(m => m.getType() === "tool");

  // 2. Check for Assistant messages that have "Source:" in the text 
  // (Since your prompt forces this format, it's a reliable indicator of tool use)
  const hasSourceInText = state.messages.some(m => 
    m.getType() === "ai" && m.content.toString().includes("[Source:")
  );

  const isGreeting = state.intents?.includes("GREETINGS");

  // LOG FOR DEBUGGING
  console.log(`⚖️ [Validator] ToolMsg: ${hasToolMessage} | SourceInText: ${hasSourceInText} | Greeting: ${isGreeting}`);

  // 3. Logic: If there is no tool message AND no assistant message with a source,
  // AND the agent is trying to answer a non-greeting question: BLOCK.
  if (!isGreeting && !hasToolMessage && !hasSourceInText && state.loopCount > 0) {
    console.warn("⚠️ Validator: Blocked response - No tool context or sources found in history.");
    return {
      messages: [new AIMessage("I'm sorry, I do not have that information in our records.")]
    };
  }

  return state;
}

function shouldContinue(state: typeof GraphState.State) {
  const last = state.messages[state.messages.length-1] as AIMessage;
  return last.tool_calls?.length ? "tools" : "validator";
}

const workflow = new StateGraph(GraphState)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addNode("validator", validator)

  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent")
  .addEdge("validator", "__end__");

const app = workflow.compile();

// ---------------------------
// Final Agent Entry Point (used by AWS AgentCore)
// ---------------------------
export async function callAgent(
  userQuery: string,
  thread_id: string,
  {
    memoryClient,
    memoryId,     // Bedrock memory for session
    actor_id,
    session_id,    // session ID
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
  const agentStartTime = Date.now();
  console.log(`\n⏱️ [Agent] Process started`);
  
  // 1️⃣ Fetch past messages from this session
  const historyStartTime = Date.now();
console.log(`⏱️ [Agent] Fetching conversation history and summary...`);

// Start both fetches in parallel
const pastMessagesPromise = fetchConversationHistory(
  memoryClient,
  memoryId,
  session_id,
  actor_id
);

const summaryStartTime = Date.now();
const pastSummariesPromise = fetchConversationSummary(
  userQuery,
  memoryClient,
  memoryId,
  session_id,
  actor_id
);

// Wait for both to complete
const [pastMessages, pastSummaries] = await Promise.all([
  pastMessagesPromise,
  pastSummariesPromise
]);

console.log(`⏱️ [Agent] Conversation history fetched in ${Date.now() - historyStartTime}ms`);
console.log(`⏱️ [Agent] Conversation summary fetched in ${Date.now() - summaryStartTime}ms`);

// Instead of: const initialMessage = new HumanMessage(userQuery);
const initialMessage = new HumanMessage({
  content: userQuery,
  additional_kwargs: { role: "user" } // Forces the role Google is looking for
});
console.log('initialmsg', initialMessage);

  // Clean messages if previous session reset happened
  const cleanedMessages = cleanPastMessagesAfterReset(pastMessages);
  const initialState = { 
    messages: [...cleanedMessages, initialMessage], 
    summaries: pastSummaries,
    onToken: onToken, // Pass onToken callback to state
  };

  console.log(`⏱️ [Agent] Starting workflow execution...`);
  const workflowStartTime = Date.now();
  // const finalState = await app.invoke(initialState, {
  //   recursionLimit: 15,
  //   configurable: {
  //     thread_id,
  //     user: { userId: actor_id, organizationId },
  //     state: initialState,
  //   },
  // });

let finalState = initialState;

// Inside your callAgent function
const eventStream = app.streamEvents(initialState, { 
  version: "v2", // Use v2 for better event metadata
  configurable: { thread_id, user: { userId: actor_id, organizationId } }
});

for await (const event of eventStream) {
  // on_chat_model_stream captures tokens from the NEW model automatically
  if (event.event === "on_chat_model_stream") {
    const chunk = event.data.chunk;
    if (chunk.content) {
      onToken?.(chunk.content);
    }
  }
}

  console.log(`⏱️ [Agent] Workflow completed in ${Date.now() - workflowStartTime}ms`);

  const allMessages = finalState.messages;

  // Find the index of the LAST human message (the one the user just sent)
const lastHumanIndex = allMessages.map(m => m.getType()).lastIndexOf("human");

// Grab everything from that Human message to the end of the array
const currentTurnMessages = allMessages.slice(lastHumanIndex);

console.log(`⏱️ [Agent] Preparing ${currentTurnMessages.length} messages for memory storage...`);

  // 2️⃣ Insert events into Bedrock memory for this session
  console.log(`⏱️ [Agent] Preparing messages for memory storage...`);
  const memoryPrepStartTime = Date.now();
  const payload: PayloadType[] = currentTurnMessages.map((msg: any) => {
    let role: "USER" | "ASSISTANT" | "TOOL" | "OTHER" = "OTHER";
    
    if (msg.getType() === "human") role = "USER";
    else if (msg.getType() === "ai") role = "ASSISTANT";
    else if (msg.getType() === "tool") role = "TOOL";

   // Handle content extraction (handles strings or tool call arrays)
  let textContent = "";
  if (typeof msg.content === "string") {
    textContent = msg.content;
  } else if (Array.isArray(msg.content)) {
    textContent = msg.content.map((c: any) => c.text || JSON.stringify(c)).join("\n");
  }

  // If it's an AI message with tool calls but no text yet, 
  // we should store the fact that a tool was called.
  if (!textContent && msg.tool_calls?.length > 0) {
    textContent = `Calling tools: ${msg.tool_calls.map((tc: any) => tc.name).join(", ")}`;
  }
  return { 
    conversational: { 
      content: { text: textContent || "No content" }, 
      role 
    } 
  } as unknown as PayloadType;
});
  console.log(`⏱️ [Agent] Messages prepared in ${Date.now() - memoryPrepStartTime}ms`);

  console.log(`⏱️ [Agent] Saving to Bedrock memory...`);
  const memorySaveStartTime = Date.now();
  console.dir(payload, { depth: null });
  
  // Kick off memory save without awaiting
(async () => {
  try {
    const command = new CreateEventCommand({
      memoryId,
      actorId: actor_id,
      sessionId: session_id,
      eventTimestamp: new Date(),
      payload,
      clientToken: crypto.randomUUID(),
    });

    await memoryClient.send(command);
    console.log(`⏱️ [Agent] Memory saved in ${Date.now() - memorySaveStartTime}ms`);
    console.log("✅ Memory saved");
  } catch (err) {
    console.error("❌ Memory save failed", err);
  }
})();

  const finalMessageContent = allMessages[allMessages.length - 1].content;
  const sources = extractSourcesFromMessages(allMessages);

  console.log(`⏱️ [Agent] Total process time: ${Date.now() - agentStartTime}ms\n`);

  return sources ? { message: finalMessageContent, sources } : { message: finalMessageContent };
}

