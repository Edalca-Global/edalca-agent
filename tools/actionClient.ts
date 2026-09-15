import { RunnableConfig } from "@langchain/core/runnables";
import { randomUUID } from "crypto";

/**
 * The one way an agent tool talks to the backend Action Layer.
 *
 * Both work order tools previously carried a byte-identical copy of this: the
 * same credential lookup, the same fetch, the same error ladder. Beyond the
 * duplication they shared four faults, all fixed here — no timeout (Node's
 * fetch will sit for minutes), no reading of `response.status` at all, no branch
 * for a 401 even though the action token lives ten minutes, and no id tying a
 * failure here to the request that caused it over there.
 *
 * Deliberately returns NO user-facing text. Wording belongs to the tool: a
 * failed lookup and a failed creation need to say very different things to the
 * model, and only the tool knows which it is.
 */

export type ActionFailure =
  | { kind: "unconfigured"; missing: "actionToken" | "WEB_BACK_URL" }
  /** 401 — the credential is missing, malformed or past its ten minutes. */
  | { kind: "unauthenticated" }
  | { kind: "permission_denied" }
  | { kind: "invalid_input"; details: unknown }
  | { kind: "rate_limited"; retryAfterMs?: number }
  /** The request was abandoned by us. The server may still have acted on it. */
  | { kind: "timeout" }
  /** DNS/TCP/TLS. Indistinguishable at this layer from a request that landed. */
  | { kind: "network" }
  | { kind: "server_error"; status: number }
  | { kind: "bad_response"; status: number }
  | { kind: "action_error"; code?: string; message?: string };

export type ActionResult<T> =
  | { ok: true; value: T; correlationId: string }
  | { ok: false; failure: ActionFailure; correlationId: string };

export interface CallActionOptions {
  /** Action name, e.g. "work_order.create". */
  action: string;
  /** Raw tool args; undefined/null are stripped before sending. */
  args: Record<string, unknown>;
  config?: RunnableConfig;
  /** Reads are fine at the default. Give writes more room. */
  timeoutMs?: number;
  /** Log prefix, e.g. "create_work_order". */
  label: string;
}

const DEFAULT_TIMEOUT_MS = 15_000;

/** Strips undefined/null so the backend's `.strict()` schema sees only real fields. */
export function toInput(args: Record<string, unknown>): Record<string, unknown> {
  const input: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args ?? {})) {
    if (value === undefined || value === null) continue;
    // An empty array is noise the backend's `.min(1)` would reject.
    if (Array.isArray(value) && value.length === 0) continue;
    input[key] = value;
  }
  return input;
}

/**
 * Safe to re-attempt automatically — for a READ.
 *
 * Never consult this for a write. `timeout` and `network` mean the outcome is
 * unknown, and on a create that is precisely the case where retrying produces a
 * second work order.
 */
export function isReadRetryable(failure: ActionFailure): boolean {
  return (
    failure.kind === "timeout" ||
    failure.kind === "network" ||
    failure.kind === "server_error" ||
    failure.kind === "rate_limited"
  );
}

export async function callAction<T = any>(
  opts: CallActionOptions
): Promise<ActionResult<T>> {
  const { action, args, config, label } = opts;
  const correlationId = randomUUID();
  const started = Date.now();

  const fail = (failure: ActionFailure, status?: number): ActionResult<T> => {
    // Structured and greppable. `args` is deliberately absent — a work order
    // description carries arbitrary user text.
    console.error(
      `[${label}] action=${action} cid=${correlationId} status=${status ?? "-"} ` +
        `kind=${failure.kind} ms=${Date.now() - started}`
    );
    return { ok: false, failure, correlationId };
  };

  const actionToken = (config?.configurable as any)?.actionToken as string | undefined;
  const baseUrl = process.env.WEB_BACK_URL?.trim();

  if (!actionToken) return fail({ kind: "unconfigured", missing: "actionToken" });
  if (!baseUrl) return fail({ kind: "unconfigured", missing: "WEB_BACK_URL" });

  let response: Response;
  try {
    response = await fetch(
      `${baseUrl.replace(/\/+$/, "")}/api/actions/${action}`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json",
          authorization: `Bearer ${actionToken}`,
          // Logged on both sides, so one id traces a failure across services.
          "x-correlation-id": correlationId,
        },
        body: JSON.stringify({ input: toInput(args) }),
        signal: AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS),
      }
    );
  } catch (error: any) {
    const aborted =
      error instanceof Error &&
      (error.name === "TimeoutError" || error.name === "AbortError");
    return fail({ kind: aborted ? "timeout" : "network" });
  }

  // Status first. The previous implementation never looked at it, so a 401 and a
  // 500 both arrived as the same generic "something went wrong".
  if (response.status === 401) return fail({ kind: "unauthenticated" }, 401);
  if (response.status === 403) return fail({ kind: "permission_denied" }, 403);
  if (response.status === 429) {
    const header = response.headers.get("retry-after");
    const seconds = header ? Number(header) : NaN;
    return fail(
      {
        kind: "rate_limited",
        ...(Number.isFinite(seconds) ? { retryAfterMs: seconds * 1000 } : {}),
      },
      429
    );
  }
  if (response.status >= 500) {
    return fail({ kind: "server_error", status: response.status }, response.status);
  }

  const body: any = await response.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return fail({ kind: "bad_response", status: response.status }, response.status);
  }

  if (body.ok !== true) {
    if (body.code === "PERMISSION_DENIED") {
      return fail({ kind: "permission_denied" }, response.status);
    }
    if (body.code === "INVALID_INPUT") {
      return fail(
        { kind: "invalid_input", details: body.details ?? body.message },
        response.status
      );
    }
    if (body.code === "UNAUTHENTICATED") {
      return fail({ kind: "unauthenticated" }, response.status);
    }
    return fail(
      { kind: "action_error", code: body.code, message: body.message },
      response.status
    );
  }

  return { ok: true, value: (body.value ?? {}) as T, correlationId };
}
