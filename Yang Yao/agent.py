# agent.py
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError

from llm_client import LLMClient


# -----------------------------
# Tool definitions
# -----------------------------

@dataclass
class ToolSpec:
    """
    A callable tool that the agent can invoke.

    name: tool name used by the LLM
    description: what it does (short, concrete)
    args_schema: pydantic model describing arguments
    func: the actual Python callable (kwargs -> Any)
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    func: Callable[..., Any]


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Tool name to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments as a JSON object")


class AgentDecision(BaseModel):
    """
    LLM must output exactly one of:
    - action = 'tool' with a tool_call
    - action = 'final' with final_answer
    """
    action: str = Field(..., description="Either 'tool' or 'final'")
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None

    # Optional fields for debugging / report writing
    short_rationale: Optional[str] = Field(
        default=None,
        description="A short explanation (1-2 sentences) of why this action was chosen. No hidden chain-of-thought.",
    )


class AgentTraceEvent(BaseModel):
    ts: float
    kind: str  # "llm_decision" | "tool_result" | "error" | "final"
    payload: Dict[str, Any]


def _safe_json(obj: Any) -> Any:
    """Make tool results JSON-serializable (best-effort)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    # fallback to string
    return str(obj)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Agent
# -----------------------------

DEFAULT_SYSTEM_PROMPT = """You are an AI agent that completes tasks by calling tools.

You have access to tools (each tool has a name / description / args schema).

How you should work:
1) First decide whether you need to call a tool. If yes, choose the most appropriate tool.
2) Output a STRICT JSON object that follows the required schema.
3) After you receive tool results, summarize and decide next steps: call another tool or provide the final answer.

Important constraints:
- Your output MUST be JSON (do not output any extra text).
- If you need to call a tool: set action="tool", and include tool_call.tool_name and tool_call.arguments.
- If you can answer directly: set action="final", and include final_answer.
- short_rationale should be 1–2 concise sentences only; do not include detailed chain-of-thought.
"""



class Agent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        tools: Optional[List[ToolSpec]] = None,
        outdir: Union[str, Path] = "outputs",
        max_iters: int = 8,
        strong: bool = False,
    ) -> None:
        self.llm = llm or LLMClient()
        self.tools = tools or []
        self.outdir = Path(outdir)
        self.max_iters = max_iters
        self.strong = strong

        _ensure_dir(self.outdir)

        # quick lookup
        self._tool_map: Dict[str, ToolSpec] = {t.name: t for t in self.tools}

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def _tools_compact_description(self) -> str:
        """A compact tool manifest inserted into the user prompt."""
        lines: List[str] = []
        for t in self.tools:
            schema = t.args_schema.model_json_schema()
            lines.append(
                json.dumps(
                    {
                        "name": t.name,
                        "description": t.description,
                        "args_schema": schema,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)

    def run(
        self,
        task: str,
        context: Optional[str] = None,
        save_trace: bool = True,
        trace_name: str = "agent_trace.json",
    ) -> str:
        """
        Run the agent until it returns a final answer or hits max_iters.
        Returns final answer text.
        """
        trace: List[AgentTraceEvent] = []
        tool_history: List[Dict[str, Any]] = []

        tool_manifest = self._tools_compact_description()
        user_prompt = {
            "task": task,
            "context": context or "",
            "tools": tool_manifest,
            "history": tool_history,
            "output_schema": AgentDecision.model_json_schema(),
        }

        for step in range(1, self.max_iters + 1):
            prompt = (
                "Make a decision based on the input below.\n"
                "You MUST output a JSON object that strictly matches output_schema.\n\n"
                f"{json.dumps(user_prompt, ensure_ascii=False)}"
            )


            decision: AgentDecision
            try:
                decision = self.llm.json(
                    prompt,
                    system=DEFAULT_SYSTEM_PROMPT,
                    schema=AgentDecision,
                    strong=self.strong,
                )
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="llm_decision",
                        payload=_safe_json(decision),
                    )
                )
            except Exception as e:
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="error",
                        payload={
                            "where": "llm_decision",
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                            "step": step,
                        },
                    )
                )
                break

            if decision.action == "final":
                final = decision.final_answer or ""
                trace.append(
                    AgentTraceEvent(ts=time.time(), kind="final", payload={"final_answer": final})
                )
                if save_trace:
                    self._save_trace(trace, trace_name)
                return final

            if decision.action != "tool" or not decision.tool_call:
                # invalid decision
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="error",
                        payload={
                            "where": "validation",
                            "error": "Invalid decision: expected action 'tool' with tool_call, or action 'final'.",
                            "raw": _safe_json(decision),
                            "step": step,
                        },
                    )
                )
                break

            tool_name = decision.tool_call.tool_name
            tool = self._tool_map.get(tool_name)
            if tool is None:
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="error",
                        payload={
                            "where": "tool_lookup",
                            "error": f"Tool not found: {tool_name}",
                            "available_tools": list(self._tool_map.keys()),
                            "step": step,
                        },
                    )
                )
                break

            # Validate tool args
            try:
                args_obj = tool.args_schema(**(decision.tool_call.arguments or {}))
            except ValidationError as ve:
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="error",
                        payload={
                            "where": "tool_args_validation",
                            "tool": tool_name,
                            "error": ve.errors(),
                            "provided": decision.tool_call.arguments,
                            "step": step,
                        },
                    )
                )
                break

            # Call tool
            try:
                result = tool.func(**args_obj.model_dump())
                event_payload = {
                    "tool": tool_name,
                    "args": args_obj.model_dump(),
                    "result": _safe_json(result),
                }
                trace.append(AgentTraceEvent(ts=time.time(), kind="tool_result", payload=event_payload))

                tool_history.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": args_obj.model_dump(),
                        "result": _safe_json(result),
                    }
                )
                user_prompt["history"] = tool_history
            except Exception as e:
                trace.append(
                    AgentTraceEvent(
                        ts=time.time(),
                        kind="error",
                        payload={
                            "where": "tool_call",
                            "tool": tool_name,
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                            "step": step,
                        },
                    )
                )
                break

        # If we end without a final answer:
        fallback = "The agent did not produce a final answer within the iteration limit. Check the trace log to diagnose the issue."

        if save_trace:
            self._save_trace(trace, trace_name)
        return fallback

    def _save_trace(self, trace: List[AgentTraceEvent], trace_name: str) -> None:
        path = self.outdir / trace_name
        payload = [t.model_dump() for t in trace]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
