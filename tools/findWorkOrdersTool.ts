import { tool } from "@langchain/core/tools";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";
import { callAction } from "./actionClient";

/**
 * find_work_orders — the work order tool, routed through the backend Action Layer.
 *
 * The previous tool (`fetchWorkOrderTool`) opened its own MongoDB connection and
 * ran a filter this model wrote as a JSON string. `organizationId` was never
 * applied, no permission was checked, and because the caller's filter was spread
 * AFTER the soft-delete guard it could override it. Every one of those is fixed by
 * not having a database here at all.
 *
 * What the model can express is now exactly the schema below. There is no filter
 * passthrough, no projection, and no way to name a field the backend has not
 * offered — the tenant comes from the credential, never from an argument.
 */

const STATUSES = ["Open", "On Hold", "In Progress", "Done"] as const;
const SORT_FIELDS = ["startDate", "dueDate", "createdAt", "name"] as const;

const FindArgs = z.object({
  status: z
    .array(z.enum(STATUSES))
    .optional()
    .describe("Restrict to these statuses. Omit for all statuses."),
  text: z
    .string()
    .optional()
    .describe("Free-text search across the work order name and description."),
  workOrderId: z
    .string()
    .optional()
    .describe('Exact human work order id, e.g. "WO-092026-0001".'),
  locationName: z
    .string()
    .optional()
    .describe('Name of the site, e.g. "North Wing". Partial names are matched.'),
  assigneeName: z
    .string()
    .optional()
    .describe("Name of the assigned person or group."),
  assignedToMe: z
    .boolean()
    .optional()
    .describe("True when the user asks about work assigned to them."),
  createdByMe: z
    .boolean()
    .optional()
    .describe("True when the user asks about work orders they created."),
  startDateFrom: z.string().optional().describe("Starts on or after, YYYY-MM-DD."),
  startDateTo: z.string().optional().describe("Starts on or before, YYYY-MM-DD."),
  dueDateFrom: z.string().optional().describe("Due on or after, YYYY-MM-DD."),
  dueDateTo: z.string().optional().describe("Due on or before, YYYY-MM-DD."),
  overdueAsOf: z
    .string()
    .optional()
    .describe("Overdue as of this date (YYYY-MM-DD) — due earlier and not yet Done."),
  createdFrom: z
    .string()
    .optional()
    .describe(
      "Created on or after this date (YYYY-MM-DD). This is when the work order was raised, not when the job starts."
    ),
  createdTo: z
    .string()
    .optional()
    .describe("Created on or before this date (YYYY-MM-DD), inclusive of that whole day."),
  includeArchived: z
    .boolean()
    .optional()
    .describe("Include archived work orders. Defaults to false."),
  view: z
    .enum(["summary", "detail"])
    .optional()
    .describe('"detail" adds description, supervisors and open notes. Default "summary".'),
  sort: z.enum(SORT_FIELDS).optional().describe('Sort field. Default "startDate".'),
  sortDir: z.enum(["asc", "desc"]).optional(),
  limit: z
    .number()
    .int()
    .optional()
    .describe("Maximum results, 1-50. Default 20."),
});

type FindArgsType = z.infer<typeof FindArgs>;

const DESCRIPTION = [
  "Find and count work orders the current user is allowed to see.",
  "Use for any question about jobs, tasks, assignments, schedules, due dates, overdue work,",
  "when a work order was created, or HOW MANY work orders match something.",
  "Every result reports a total count of all matches, not just the rows returned, so this tool",
  "answers 'how many' questions too — call it with no filters for an overall total.",
  "Results are already scoped to the user's organisation and permissions — never ask for an",
  "organisation or user id, and never try to widen the search beyond the arguments listed.",
].join(" ");

export const findWorkOrdersTool = tool(
  async (args: FindArgsType, config?: RunnableConfig): Promise<string> => {
    const onToken = config?.metadata?.onToken as ((t: string) => void) | undefined;
    if (typeof onToken === "function") {
      // This exact string is stripped from the persisted message by web-back
      // (app.ts, the send-message handler) — keep it byte-identical.
      onToken(`🔍 *Searching Work Order in database...*\n\n`);
    }

    const res = await callAction<any>({
      action: "work_order.find",
      args: args as Record<string, unknown>,
      config,
      label: "find_work_orders",
    });

    if (!res.ok) {
      // The backend's messages are deliberately opaque; pass the shape of the
      // failure to the model, never raw internals.
      const f = res.failure;
      switch (f.kind) {
        case "unconfigured":
          return "Result: Work order lookup is unavailable for this conversation. Tell the user you cannot access work orders right now.";
        case "unauthenticated":
          return "Result: The user's session has expired. Ask them to send their question again.";
        case "permission_denied":
          return "Result: This user is not permitted to view work orders. Tell them they lack access.";
        case "invalid_input":
          return `Result: The search arguments were rejected: ${JSON.stringify(
            f.details
          )}. Correct them and try once more.`;
        case "rate_limited":
          return "Result: The work order service is busy. Tell the user to try again in a moment.";
        case "bad_response":
          return "Result: The work order service returned an unexpected response. Tell the user the lookup failed.";
        default:
          return "Result: The work order lookup failed. Tell the user you could not retrieve work orders.";
      }
    }

    const value = res.value ?? {};
    const items = Array.isArray(value.items) ? value.items : [];

    // `total` counts every match within the user's visibility, not just the
    // rows returned — state it unconditionally so a "how many" question can be
    // answered from one call, and so a truncated list is never reported as a
    // complete one.
    const total = typeof value.total === "number" ? value.total : items.length;

    if (items.length === 0) {
      return "TOTAL MATCHING: 0\nNo work orders matched. Tell the user nothing matched, and suggest relaxing a filter.";
    }

    const shown = value.truncated
      ? `Showing the first ${items.length}; ask again with a narrower filter to see the rest.`
      : `All ${items.length} are listed below.`;

    return `TOTAL MATCHING: ${total}\n${shown}\n${JSON.stringify(items)}`;
  },
  {
    name: "find_work_orders",
    description: DESCRIPTION,
    schema: FindArgs,
  }
);
