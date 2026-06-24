import Anthropic from "@anthropic-ai/sdk";
import type { Config } from "./config.js";
import { getActiveProjects, getAllProjects, getGoals } from "./db.js";
import { allocateDay, score, daysSince } from "./scoring.js";
import { getStalledProjects } from "./db.js";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/** Keep token use bounded: only send the most recent turns. */
const MAX_HISTORY = 20;
const MAX_OUTPUT_TOKENS = 1024;

export function isAiConfigured(config: Config): boolean {
  return config.anthropicApiKey.length > 0;
}

function round1(n: number): string {
  return (Math.round(n * 10) / 10).toString();
}

/**
 * Build the system prompt from live data so the agent always reasons over the
 * current goals, projects, scores, and today's allocation.
 */
export function buildSystemPrompt(config: Config): string {
  const goals = getGoals();
  const active = getActiveProjects();
  const all = getAllProjects();
  const allocation = allocateDay();
  const stalled = getStalledProjects(config.stallDays);

  const lines: string[] = [];

  lines.push(
    "You are the AI assistant for manoverboard.ai — a personal assistant and business analyst for a solo developer building a freelance/dev business on the side of a full-time job.",
    "",
    "How you think:",
    "- The user is time-poor: ~1 hour on weeknights, more on weekends. Keep suggestions realistic for that.",
    "- Two project tracks: 'fast' = client/services/paid software = income, ALWAYS the priority; 'passive' = ads/affiliate/own products = long game, only with leftover time.",
    "- Never dump the whole task list. Surface the ONE highest-leverage next move, then a little supporting context.",
    "- Priority score = (revenue_potential * confidence * (6 - time_to_cash)) / max(effort_remaining, 1). Higher = do sooner.",
    "- Be concrete and concise. No motivational fluff. Sharpen vague next actions into specific, shippable steps. Help rank, plan, and unblock.",
    ""
  );

  lines.push("# Goals");
  if (goals.length === 0) {
    lines.push("(none set yet — encourage the user to define one)");
  } else {
    for (const g of goals) {
      lines.push(`- ${g.title}${g.detail ? ` — ${g.detail}` : ""}`);
    }
  }
  lines.push("");

  lines.push("# Active projects");
  if (active.length === 0) {
    lines.push("(no active projects)");
  } else {
    for (const p of active) {
      const bits = [
        `score ${round1(score(p))}`,
        `rev ${p.revenue_potential}/5`,
        `conf ${p.confidence}/5`,
        `time_to_cash ${p.time_to_cash}/5`,
        `~${p.effort_remaining}h left`,
      ];
      if (p.deadline) bits.push(`deadline ${p.deadline}`);
      if (p.last_progress_at) {
        const d = daysSince(p.last_progress_at);
        if (d !== null) bits.push(`${d}d since progress`);
      }
      lines.push(
        `- #${p.id} [${p.type}] ${p.name}${p.client ? ` (client: ${p.client})` : ""} — ${bits.join(", ")}`
      );
      lines.push(`    next: ${p.next_action ?? "(none set)"}`);
      if (p.notes) lines.push(`    notes: ${p.notes}`);
    }
  }
  lines.push("");

  const nonActive = all.filter((p) => p.status !== "active");
  if (nonActive.length > 0) {
    lines.push("# Other projects (not active)");
    for (const p of nonActive) {
      lines.push(`- #${p.id} [${p.type}] ${p.name} — status: ${p.status}`);
    }
    lines.push("");
  }

  lines.push("# Today's allocation (computed by the formula)");
  if (allocation.primary) {
    lines.push(
      `- PRIMARY (income): #${allocation.primary.project.id} ${allocation.primary.project.name} — ${allocation.primary.project.next_action ?? "(no next action)"}`
    );
  } else {
    lines.push("- PRIMARY: none — no active fast/income projects. Push the user to find/close a client.");
  }
  if (allocation.secondary) {
    lines.push(
      `- Secondary (passive, max 30 min): #${allocation.secondary.project.id} ${allocation.secondary.project.name}`
    );
  }
  if (allocation.deadlineWarnings.length > 0) {
    lines.push(
      `- Deadlines within 3 days: ${allocation.deadlineWarnings.map((p) => `${p.name} (${p.deadline})`).join("; ")}`
    );
  }
  if (stalled.length > 0) {
    lines.push(
      `- Stalling (no progress in ${config.stallDays}+ days): ${stalled.map((p) => p.name).join("; ")}`
    );
  }

  return lines.join("\n");
}

/**
 * Send a conversation to the model and return the assistant's text reply.
 * Throws if the AI agent isn't configured.
 */
export async function chat(config: Config, messages: ChatMessage[]): Promise<string> {
  if (!isAiConfigured(config)) {
    throw new Error("AI agent is not configured (set ANTHROPIC_API_KEY).");
  }

  const client = new Anthropic({ apiKey: config.anthropicApiKey });
  const trimmed = messages.slice(-MAX_HISTORY);

  const response = await client.messages.create({
    model: config.anthropicModel,
    max_tokens: MAX_OUTPUT_TOKENS,
    system: buildSystemPrompt(config),
    messages: trimmed.map((m) => ({ role: m.role, content: m.content })),
  });

  const text = response.content
    .filter((block): block is Anthropic.TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  return text || "(the assistant returned no text)";
}
