import express, { Request, Response } from "express";
import { callAgent } from "./index";
import dotenv from "dotenv";
import cors from "cors";
import { BedrockAgentCoreControlClient } from "@aws-sdk/client-bedrock-agentcore-control";
import { BedrockAgentCoreClient } from "@aws-sdk/client-bedrock-agentcore";
import { runWorkOrderAgent } from "./testFile";

dotenv.config();
const app = express();
const port = process.env.PORT || 8080;
// Middleware
app.use(express.json());
app.use(cors());

const client = new BedrockAgentCoreControlClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

const memoryClient = new BedrockAgentCoreClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

// Handle invocation requests from the Bedrock AgentCore Runtime with streaming
app.post("/invocations", async (req: Request, res: Response) => {
  const startTime = Date.now();
  console.log('Agent Invoked ', startTime);
  
  const stepTimings: { step: string; timestamp: number; elapsed?: number }[] = [];
  
  const logStep = (stepName: string) => {
    const now = Date.now();
    const elapsed = stepTimings.length > 0 ? now - stepTimings[stepTimings.length - 1].timestamp : 0;
    stepTimings.push({ step: stepName, timestamp: now, elapsed });
    console.log(`⏱️ [${stepName}] - Elapsed from previous step: ${elapsed}ms | Total elapsed: ${now - startTime}ms`);
  };
  
  try {
    logStep("[1/6] Server: Request received");
    // await initializeMongoConnection();
    // logStep("[2/6] Server: MongoDB connection initialized");
    console.log("-Request Body", req.body);

    const { prompt: userQuery, chatId, memoryId, sessionId, userId, organizationId } = req.body;

    if (!userQuery || typeof userQuery !== "string") {
      return res
        .status(400)
        .json({ error: "Missing or invalid 'prompt' in request payload." });
    }

    logStep("[3/6] Server: Request validated");
    // ----- SESSION INTEGRATION -----
    // const session = await GenerateTitleForSession(client, chatId, sessionId, userQuery);
    // logStep("[4/6] Server: Session generated");

    // const chatDoc = await Chat.findById(chatId);
    // if (!chatDoc) {
    //   return res.status(404).json({ error: "Chat not found." });
    // }
    // const memoryId = chatDoc.memoryId;

    // console.log(`- Using sessionId: ${session._id}, memoryId: ${memoryId}`);
    logStep("[5/6] Server: Calling agent");

    // Set up SSE (Server-Sent Events) for streaming
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Add this to your /invocations headers
    res.setHeader('X-Accel-Buffering', 'no');

    // Token buffer to collect streaming tokens
    let accumulatedTokens = "";
    let sources: any = null;

    const onToken = (token: string) => {
      accumulatedTokens += token;
      // Send each token as an SSE event
      res.write(`data: ${JSON.stringify({
        type: 'token',
        content: token,
        timestamp: new Date().toISOString(),
      })}\n\n`);
    };

    // const agentResponse = await callAgent(userQuery, `thread-${Date.now()}`, {
    //   memoryClient,
    //   memoryId,
    //   actor_id: userId,
    //   session_id: sessionId,
    //   organizationId,
    //   onToken,
    // });

    const agentResponse = await runWorkOrderAgent(userQuery, `thread-${Date.now()}`, {
      memoryClient,
      memoryId,
      actor_id: userId,
      session_id: sessionId,
      organizationId,
      onToken,
    });

    logStep("[6/6] Server: Agent response received");
    // sources = agentResponse?.sources;
    
    const responseTime = Date.now() - startTime;
    console.log(`\n✅ Agent responded successfully in ${responseTime}ms`);
    console.log("\n📊 Step-by-step timing:");
    stepTimings.forEach((t, idx) => {
      console.log(`  ${t.step}: +${t.elapsed}ms (Total: ${t.timestamp - startTime}ms)`);
    });
    console.log(`\n🏁 Total time: ${responseTime}ms\n`);
    console.log('Response completed at ', Date.now());
    
    // Send final metadata and completion event
    res.write(`data: ${JSON.stringify({
      type: 'metadata',
      sources: sources,
      metadata: {
        responseTime,
        sessionId: sessionId,
        timestamp: new Date().toISOString(),
      },
    })}\n\n`);

    // Send completion signal
    res.write(`data: ${JSON.stringify({
      type: 'done',
      timestamp: new Date().toISOString(),
    })}\n\n`);

    res.end();
  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    console.error(`❌ Agent invocation failed after ${responseTime}ms`, error);
    
    // Send error via SSE
    res.write(`data: ${JSON.stringify({
      type: 'error',
      error: "Agent processing failed.",
      details: error?.message,
      timestamp: new Date().toISOString(),
    })}\n\n`);
    
    res.end();
  }
});

// Health check endpoint with more details
app.get("/ping", (req: Request, res: Response) => {
  // console.log("🏓 Health check requested");
  res.status(200).json({
    status: "OK",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});
// Start the server
app.listen(port, () => {
  console.log(`🚀 Agent server listening at http://localhost:${port}`);
});
