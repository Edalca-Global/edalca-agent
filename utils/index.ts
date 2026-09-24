/**
 * Lynk has ONE AgentCore memory for the whole platform (per environment),
 * configured as AGENTCORE_MEMORY_ID. Tenants are separated only by actorId, which
 * also becomes the summary namespace `/summaries/{actorId}/{sessionId}` — so the
 * org prefix gives every tenant its own namespace prefix for audit and deletion.
 *
 * Both parts are restricted to [A-Za-z0-9-]: `_` is the separator and `/` would
 * let a crafted id escape its namespace segment. In practice both are ObjectIds.
 */
const ACTOR_ID_PART = /^[A-Za-z0-9-]+$/;

export function buildActorId(organizationId: unknown, userId: unknown): string {
  if (
    typeof organizationId !== "string" || !ACTOR_ID_PART.test(organizationId) ||
    typeof userId !== "string" || !ACTOR_ID_PART.test(userId)
  ) {
    throw new Error("Invalid organizationId or userId for memory actorId");
  }
  return `${organizationId}_${userId}`;
}

/**
 * CreateEvent rejects any single message over 100 KB, and it rejects the WHOLE
 * event — the same failure mode as an empty entry. Cut long text to a byte budget
 * below that limit instead of losing the turn.
 */
export const MAX_MEMORY_TEXT_BYTES = 90 * 1024;

export function truncateUtf8(text: string, maxBytes: number): string {
  const buf = Buffer.from(text, "utf8");
  if (buf.length <= maxBytes) return text;
  // A cut through a multi-byte character decodes to U+FFFD; drop it.
  return buf.subarray(0, maxBytes).toString("utf8").replace(/\uFFFD+$/, "");
}

import { BaseMessage } from "@langchain/core/messages";
import ChatSession from "../model/chat/ChatSessionModel";
// import { ChatOpenAI } from "@langchain/openai";

/**
 * Cleans up past conversation messages by removing any messages
 * before a successful tool call that contains a `reset: true` flag.
 *
 * This is useful for removing stale "draft" states once a work order
 * has been successfully created, preventing old data from leaking into
 * a new work order creation flow.
 *
 * @param pastMessages - Array of past conversation messages
 * @returns A cleaned array of messages starting *after* the last reset point
 */
export function cleanPastMessagesAfterReset(
  pastMessages: BaseMessage[]
): BaseMessage[] {
  const cleanedMessages: BaseMessage[] = [];

  let resetFound = false;

  // Reverse loop to find the most recent reset point
  for (let i = pastMessages.length - 1; i >= 0; i--) {
    const msg = pastMessages[i];

    let content: any;

    try {
      // Try to parse content if it's a JSON string
      content =
        typeof msg.content === "string" ? JSON.parse(msg.content) : msg.content;

      // If a tool message explicitly marked a reset
      if (content?.reset === true) {
        resetFound = true;
        break;
      }
    } catch (err) {
      // Skip invalid or non-tool message content
    }

    // Add message to the beginning of cleaned list
    cleanedMessages.unshift(msg);
  }

  // If reset was found, return messages after it (cleanedMessages)
  // If not found, return the full history (no cleanup)
  return resetFound ? cleanedMessages : pastMessages;
}

export function extractSourcesFromMessages(messages: any): string[] | null {
  const urls = new Set<string>();

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];

    // Case 1: JSON string content
    if (typeof msg.content === "string") {
      try {
        const parsed = JSON.parse(msg.content);
        parsed?.sources?.forEach((s: any) => urls.add(s?.uri));
      } catch {}
    }

    // Case 2: structured content
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        part?.sources?.forEach((u: string) => urls.add(u));
      }
    }

    if (urls.size > 0) break; // keep "last message wins" behavior
  }

  return urls.size ? [...urls] : null;
}

// export async function GenerateTitleForSession(
//   client: BedrockAgentCoreControlClient,
//   chatId: string,
//   sessionId: string,
//   userQuery: string
// ) {
//   let session = await ChatSession.findOne({
//     chatId: chatId,
//     _id: sessionId,
//     isActive: true,
//   });
//   const model = new ChatOpenAI({
//     model: "gpt-4o-mini-2024-07-18",
//     temperature: 0.2,
//   });
//   if (!session?.title) {
//     // generate a short title using the users first message for the session
//     if (!session?.title) {
//       const titlePrompt = `
// Generate a short, clear title 2-3 words (max 7 words) that summarizes this user request.
// Rules:
// - No quotes
// - No punctuation
// - Title Case
// - Be concise

// User message:
// "${userQuery}"
// `;

//       const titleResponse = await model.invoke(titlePrompt);

//       const title =
//         typeof titleResponse.content === "string"
//           ? titleResponse.content.trim()
//           : "New Chat";

//       session.title = title;
//       await session.save();
//     }
//   }

//   return session;
// }
