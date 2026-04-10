"""
Guardrails: Input and Output validation (RSM8430: Guardrails lecture)

Two-layer guardrail system:
  1. Input Guardrails: Block prompt injection + unsafe content BEFORE the LLM
  2. Output Guardrails: Validate LLM response AFTER generation

Course concept: "Guardrails work better when implemented as separate,
focused checks rather than relying on the LLM to self-police."
"""

from __future__ import annotations
import re

# ── INPUT GUARDRAILS ─────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore (all |your |previous )?instructions",
    r"you are now",
    r"forget (everything|your|all)",
    r"pretend you",
    r"act as",
    r"system\s*prompt",
    r"jailbreak",
    r"do anything now",
    r"developer mode",
    r"bypass",
    r"override",
    r"disregard",
    r"new persona",
    r"ignore above",
    r"repeat (the|your) (system|initial) (prompt|instructions)",
]

_UNSAFE_PATTERNS = [
    r"\b(kill|murder|attack|bomb|weapon|hack|exploit|steal)\b",
    r"\b(drugs?|cocaine|heroin|meth)\b",
    r"\b(suicide|self.harm)\b",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def check_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def check_unsafe_content(text: str) -> bool:
    return bool(_UNSAFE_RE.search(text))


def run_guardrails(user_input: str) -> str | None:
    """Input guardrails. Returns rejection message or None if safe."""
    if check_prompt_injection(user_input):
        return (
            "I detected a prompt that looks like it's trying to alter my instructions. "
            "I'm an Uber Eats support assistant and can only help with "
            "Uber Eats-related questions and actions. How can I help you today?"
        )
    if check_unsafe_content(user_input):
        return (
            "I'm not able to help with that type of request. "
            "I'm here to assist with Uber Eats orders, deliveries, "
            "refunds, and account questions. Is there anything else I can help with?"
        )
    return None


# ── OUTPUT GUARDRAILS ────────────────────────────────────────────────

_OUTPUT_BLOCKED_PATTERNS = [
    r"(I am|I'm) (an AI|a language model|ChatGPT|GPT)",
    r"As an AI",
    r"I don't have (real|actual) access",
    r"I('m| am) not (actually|really) (an|a) ",
]

_OUTPUT_BLOCKED_RE = re.compile("|".join(_OUTPUT_BLOCKED_PATTERNS), re.IGNORECASE)


def check_output_guardrails(response: str) -> str | None:
    """Output guardrails. Returns replacement message or None if clean."""
    if _OUTPUT_BLOCKED_RE.search(response):
        return None  # Allow but could flag — for now, pass through
    if len(response) > 3000:
        return response[:3000] + "\n\n_(Response truncated for brevity. Ask me for more details on any specific point.)_"
    return None
