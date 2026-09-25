import { tool } from "@langchain/core/tools";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";
import { callAction } from "./actionClient";

/**
 * create_work_order — routed through the backend Action Layer.
 *
 * There is no database here and no state here. The backend holds a draft scoped
 * to this conversation and merges each call into it, so this tool sends only
 * what the user has just said and the server replies with everything collected
 * so far. That inversion is the point: the old contract demanded all seven
 * fields on the first call, which is exactly when the model has to invent a
 * procedure or a date nobody has mentioned.
 *
 * The draft is keyed by the chat inside the action token, never by an argument.
 * The model has no identifier to carry, so it cannot reuse a stale one and two
 * work orders raised in one conversation cannot share a value.
 *
 * Writing is still a separate, deliberate act: a call without `confirm` returns
 * a preview having written NOTHING, and the backend refuses a confirmation whose
 * values no longer match what was previewed.
 */

const RECURRENCE_TYPES = [
  "daily",
  "weekly",
  "monthly_date",
  "monthly_weekday",
  "yearly",
] as const;

/** A closed menu is far more reliable for the model than free-form field names. */
const CLEARABLE_FIELDS = [
  "name",
  "description",
  "locationName",
  "procedureName",
  "assigneeName",
  "startDate",
  "dueDate",
  "timeInHours",
  "recurrence",
] as const;

const CreateArgs = z.object({
  name: z.string().optional().describe("Short title, e.g. 'Replace AC filter'."),
  description: z
    .string()
    .optional()
    .describe("What needs to be done, in the user's own terms."),
  locationName: z
    .string()
    .optional()
    .describe(
      "Name of the site the work happens at. Use only a name the user gave or that this tool listed."
    ),
  procedureName: z
    .string()
    .optional()
    .describe(
      "Name of the procedure/checklist to attach. Use only a name the user gave or that this tool listed."
    ),
  assigneeName: z
    .string()
    .optional()
    .describe(
      "Name of the person to assign. Use only a name the user gave or that this tool listed."
    ),
  startDate: z.string().optional().describe("When the job starts, YYYY-MM-DD."),
  dueDate: z
    .string()
    .optional()
    .describe("When the job is due, YYYY-MM-DD. Must not be before startDate."),
  timeInHours: z
    .number()
    .optional()
    .describe("Estimated hours. Only pass it if the user actually said so."),
  recurrence: z
    .object({
      type: z.enum(RECURRENCE_TYPES),
      interval: z.number().int().optional().describe("Every N periods. Default 1."),
      daysOfWeek: z
        .array(z.number().int())
        .optional()
        .describe("weekly only. 0=Sunday … 6=Saturday."),
      dayOfMonth: z.number().int().optional().describe("monthly_date only, 1-31."),
      weekOfMonth: z
        .number()
        .int()
        .optional()
        .describe("monthly_weekday only, 1-5 where 5 means last."),
      weekday: z.number().int().optional().describe("monthly_weekday only. 0=Sunday … 6=Saturday."),
      month: z.number().int().optional().describe("yearly only, 1-12."),
      day: z.number().int().optional().describe("yearly only, 1-31."),
      startDate: z.string().describe("When the series starts, YYYY-MM-DD."),
      endDate: z.string().describe("When the series ends, YYYY-MM-DD."),
    })
    .optional()
    .describe(
      "Omit entirely for a one-off. To undo a recurrence already sent, use clear: ['recurrence']."
    ),

  clear: z
    .array(z.enum(CLEARABLE_FIELDS))
    .optional()
    .describe(
      "Erase these details from the work order being built. Use ONLY when the user withdraws a value " +
        "without giving a replacement. To CHANGE a value, just send the field with its new value."
    ),
  discard: z
    .boolean()
    .optional()
    .describe(
      "Set true ONLY when the user abandons the work order being built and wants a DIFFERENT one " +
        "('forget that, raise one for the boiler instead'). Never set it for a correction."
    ),
  confirmDiscard: z
    .boolean()
    .optional()
    .describe(
      "Set true alongside discard ONLY after the user has agreed to throw away the details the tool listed."
    ),
  confirm: z
    .boolean()
    .optional()
    .describe(
      "Set true ONLY after the tool returned PREVIEW and the user agreed to it. Send NO other arguments " +
        "with it — the server already holds every value. Never set it on a first call."
    ),
});

type CreateArgsType = z.infer<typeof CreateArgs>;

const DESCRIPTION = [
  "Create a new work order.",
  "The server remembers the work order being built for this chat, so never restate values you have",
  "already sent — pass ONLY what the user just told you, and the tool replies with everything",
  "collected so far plus whatever is still missing.",
  "Call it as soon as the user asks for a work order, even if all you have is a name: the tool will",
  "say what to ask for and will list the real locations, procedures and people to choose from.",
  "Never invent any of those.",
  "When everything is present it returns PREVIEW without writing anything; show that to the user and",
  "only once they agree, call again with confirm: true and no other arguments.",
  "If the user abandons this work order for a different one, call with discard: true.",
].join(" ");

/** Server field names → what a person calls them. */
const LABELS: Record<string, string> = {
  name: "Name",
  description: "Description",
  locationName: "Location",
  procedureName: "Procedure",
  assigneeName: "Assignee",
  startDate: "Starts",
  dueDate: "Due",
  timeInHours: "Estimated hours",
  recurrence: "Repeats",
  location: "Location",
  procedure: "Procedure",
  assignedTo: "Assignee",
};

const label = (field: string) => LABELS[field] ?? field;

function renderValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") {
    const r = value as Record<string, unknown>;
    if (r.type) {
      const bits = [String(r.type)];
      if (r.startDate && r.endDate) bits.push(`from ${r.startDate} to ${r.endDate}`);
      return bits.join(", ");
    }
    return JSON.stringify(value);
  }
  return String(value);
}

/** A labelled list reads back to a user far more reliably than a JSON blob. */
function renderFields(fields: Record<string, unknown>): string {
  const lines = Object.entries(fields)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `- ${label(k)}: ${renderValue(v)}`);
  return lines.length > 0 ? lines.join("\n") : "(nothing yet)";
}

function renderMenu(
  title: string,
  menu: any,
  emptyHint: string
): string {
  if (!menu) return `- ${title} — ask the user.`;
  if (menu.pending === "location") {
    return `- ${title} — cannot be listed until the location is known. Get the location first.`;
  }
  const items: any[] = Array.isArray(menu.items) ? menu.items : [];
  if (items.length === 0) return `- ${title} — ${emptyHint}`;
  const rows = items
    .map((i) => (i.email ? `    • ${i.name} (${i.email})` : `    • ${i.name}`))
    .join("\n");
  const tail = menu.truncated
    ? "\n    (more exist — if none of these fit, ask the user to name it)"
    : "";
  return `- ${title} — choose one of these, do not invent another:\n${rows}${tail}`;
}

export const createWorkOrderTool = tool(
  async (args: CreateArgsType, config?: RunnableConfig): Promise<string> => {
    const res = await callAction<any>({
      action: "work_order.create",
      args: args as Record<string, unknown>,
      config,
      label: "create_work_order",
      // A create runs a transaction plus assignment emails and notifications;
      // 15s is not enough headroom for the confirming call.
      timeoutMs: args?.confirm ? 30_000 : 20_000,
    });

    if (!res.ok) {
      const f = res.failure;
      switch (f.kind) {
        case "unconfigured":
          return "UNAVAILABLE — nothing was created. Tell the user you cannot create work orders right now. Do not retry.";
        case "unauthenticated":
          return "SESSION EXPIRED — nothing was created. Tell the user their session expired and ask them to send the request again. Do NOT call this tool again in this reply.";
        case "permission_denied":
          return "NOT PERMITTED — nothing was created. Tell the user they do not have access to create work orders.";
        case "invalid_input":
          return `REJECTED — nothing was created. The server rejected: ${JSON.stringify(
            f.details
          )}. Ask the user for a corrected value, then send ONLY that field.`;
        case "rate_limited":
          return "BUSY — nothing was created. Tell the user the system is busy and ask them to try again shortly. Do NOT retry in this reply.";
        case "timeout":
        case "network":
        case "server_error":
          // The ONLY moment "could not reach it" might mean the write landed.
          // Saying "nothing was created" here is how duplicates get made.
          return args?.confirm
            ? [
                "OUTCOME UNKNOWN — the work order service did not answer in time.",
                "It MAY or MAY NOT have been created. Do NOT call create_work_order again.",
                "Tell the user you could not confirm whether it was created, then use find_work_orders",
                "with the work order name to check.",
              ].join(" ")
            : "UNAVAILABLE — nothing was created. Tell the user the work order service could not be reached. Do NOT retry in this reply.";
        default:
          return "UNAVAILABLE — nothing was created. Tell the user the attempt failed, and do NOT claim a work order was created.";
      }
    }

    const value = res.value ?? {};

    switch (value.status) {
      case "needs_input": {
        const missing: string[] = Array.isArray(value.missing) ? value.missing : [];
        const c = value.candidates ?? {};
        const asks: string[] = [];

        for (const field of missing) {
          if (field === "locationName") {
            asks.push(renderMenu("Location", c.locations, "none are available to this user."));
          } else if (field === "procedureName") {
            asks.push(renderMenu("Procedure", c.procedures, "none exist in this organisation."));
          } else if (field === "assigneeName") {
            asks.push(renderMenu("Assignee", c.assignees, "nobody is assignable there."));
          } else if (field === "startDate" || field === "dueDate") {
            asks.push(`- ${label(field)} — YYYY-MM-DD`);
          } else {
            asks.push(`- ${label(field)} — ask the user.`);
          }
        }

        const invalid: any[] = Array.isArray(value.invalid) ? value.invalid : [];
        const problems =
          invalid.length > 0
            ? `\nProblems to fix:\n${invalid
                .map((i) => `- ${label(i.field)}: ${i.message}`)
                .join("\n")}\n`
            : "";

        return [
          "NEEDS INPUT — nothing has been created yet.",
          "",
          "Collected so far (the server has these; do NOT send them again, and do NOT ask for them again):",
          renderFields(value.provided ?? {}),
          problems,
          asks.length > 0 ? "Still needed:\n" + asks.join("\n") : "",
          "",
          "Tell the user what you already have, then ask for what is still needed.",
          "Where options are listed, offer exactly those and nothing else.",
          "When the user answers, call create_work_order again with ONLY the new values.",
        ]
          .filter((s) => s !== "")
          .join("\n");
      }

      case "needs_clarification": {
        const list = Array.isArray(value.candidates)
          ? value.candidates
              .map((c: any) => (c.email ? `- ${c.name} (${c.email})` : `- ${c.name}`))
              .join("\n")
          : "";
        return [
          "NEEDS CLARIFICATION — nothing has been created yet.",
          value.message ?? "",
          list ? `Options:\n${list}` : "",
          `Ask the user which one they mean and list exactly these options. Do not pick for them, and do not retry with a guess.`,
          `Once they answer, call create_work_order with just ${value.field ?? "that field"} set to their choice.`,
        ]
          .filter(Boolean)
          .join("\n");
      }

      case "preview": {
        const r = value.resolved ?? {};
        const recurrence = r.recurrence
          ? renderValue(r.recurrence)
          : "no — this is a one-off";
        return [
          value.requiresReconfirmation
            ? "PREVIEW (details changed since the last preview) — NOTHING HAS BEEN CREATED YET."
            : "PREVIEW — NOTHING HAS BEEN CREATED YET.",
          "",
          renderFields({
            name: r.name,
            description: r.description,
            location: r.location,
            procedure: r.procedure,
            assignedTo: r.assignedTo,
            startDate: r.startDate,
            dueDate: r.dueDate,
            timeInHours: r.timeInHours,
          }),
          // Stated even when absent: an explicit negative gets read aloud, and
          // "did you mean this to repeat?" is the commonest late correction.
          `- Repeats: ${recurrence}`,
          "",
          "Read these details back to the user in plain language and ask them to confirm.",
          "If they want a change, send just the field that changed and you will get a new PREVIEW.",
          "If they agree, call create_work_order with confirm: true and NO other arguments.",
        ].join("\n");
      }

      case "created": {
        const wo = value.workOrder ?? {};
        const warnings: any[] = Array.isArray(value.warnings) ? value.warnings : [];
        const warningBlock =
          warnings.length > 0
            ? [
                "",
                "The work order EXISTS, but some follow-up actions did not happen. Tell the user plainly:",
                ...warnings.map((w) => `- ${w.message}`),
              ].join("\n")
            : "";
        return [
          value.replay
            ? "ALREADY CREATED — this work order was created by an earlier attempt; it was NOT created twice."
            : "CREATED — the work order now exists.",
          "",
          renderFields({
            "Work order id": wo.workOrderId,
            name: wo.name,
            location: wo.location,
            procedure: wo.procedure,
            startDate: wo.startDate,
            dueDate: wo.dueDate,
          }),
          warningBlock,
          "",
          `Tell the user it was created and quote the id ${wo.workOrderId ?? ""}.`.trim(),
          "This work order is finished and its draft has been cleared on the server.",
          "If the user asks for another work order, start completely fresh: carry NOTHING over from",
          "this one — not the location, the procedure, the person, the dates or the description —",
          "and do not offer them as suggestions.",
        ]
          .filter((s) => s !== "")
          .join("\n");
      }

      case "discard_preview":
        return [
          "NOT DISCARDED YET — these details would be thrown away:",
          renderFields(value.wouldLose ?? {}),
          "",
          "Ask the user to confirm they want to abandon this work order.",
          "If they say yes, call create_work_order again with discard: true and confirmDiscard: true.",
          "If they say no, carry on with the work order as it stands.",
        ].join("\n");

      case "discarded":
        return [
          "DISCARDED — nothing was created and nothing was kept.",
          "Confirm to the user that it has been abandoned.",
          "If they now want a different work order, start from nothing and carry no values over.",
        ].join("\n");

      case "resumed_after_expiry":
        return [
          "STARTING FRESH — the work order being built here expired and nothing was created.",
          "",
          "Nothing is collected. Do NOT reuse any values from earlier in this conversation, even if",
          "you can still see them above — they are stale and may have been superseded.",
          "Ask the user to state what they want again, then call create_work_order with only what",
          "they tell you now.",
        ].join("\n");

      case "in_progress":
        return [
          "ALREADY IN PROGRESS — an earlier request to create this work order is still running.",
          "Do NOT call create_work_order again. Tell the user it is being created and check with",
          "find_work_orders in a moment to confirm the result.",
        ].join("\n");

      default:
        console.error("[create_work_order] unrecognised result", value?.status);
        return "UNAVAILABLE — the work order service returned an unexpected result. Do not claim a work order was created.";
    }
  },
  {
    name: "create_work_order",
    description: DESCRIPTION,
    schema: CreateArgs,
  }
);
