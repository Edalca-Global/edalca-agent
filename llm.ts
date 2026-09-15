import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOllama } from "@langchain/ollama";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

/**
 * Chat model factory.
 *
 * Set LLM_PROVIDER=ollama to run against a local Ollama server instead of
 * Gemini (testing only — avoids the Gemini API rate limits). Anything else
 * (or unset) keeps the default Gemini behaviour.
 *
 * Env:
 *   LLM_PROVIDER   "gemini" (default) | "ollama"
 *   OLLAMA_MODEL   defaults to "llama3.1:8b"
 *   OLLAMA_BASE_URL defaults to "http://127.0.0.1:11434"
 */
export const LLM_PROVIDER = (process.env.LLM_PROVIDER || "gemini").toLowerCase();

export const isOllama = LLM_PROVIDER === "ollama";

type ChatModelOptions = {
  model: string;
  temperature?: number;
  streaming?: boolean;
};

export function createChatModel(options: ChatModelOptions): BaseChatModel {
  if (isOllama) {
    const ollamaModel = process.env.OLLAMA_MODEL || "llama3.1:8b";
    console.log(`🦙 Using Ollama model: ${ollamaModel}`);
    return new ChatOllama({
      model: ollamaModel,
      baseUrl: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
      temperature: options.temperature,
      // Local models need a generous context to fit the system prompt + history
      numCtx: Number(process.env.OLLAMA_NUM_CTX || 8192),
    });
  }

  return new ChatGoogleGenerativeAI({
    model: options.model,
    apiKey: process.env.GEMINI_API_KEY,
    streaming: options.streaming,
    temperature: options.temperature,
  });
}
