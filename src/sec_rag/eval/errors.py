"""Classify an eval-time exception as *fatal* (unrecoverable) or transient.

Why this exists
---------------
The eval runners retry a failed question on a fresh engine to survive a transient
blip — a dropped Neon socket, a one-off Anthropic timeout, a momentary 429 rate
limit. But an *account-level* failure is NOT transient: the Anthropic credit
balance hitting zero, an OpenAI quota-out, a revoked/expired API key. Every
remaining question will fail the same way. Retrying and continuing turns one root
cause into N per-question "failures" while the runner still emits aggregate recall
(or accuracy) over the lucky prefix that ran before the outage. That partial then
looks like a real benchmark number.

This is not hypothetical — it bit the project twice: the concise-prompt full run
died mid-way on Anthropic credit depletion (110 "errors") and the confound
accuracy arm on OpenAI/Anthropic outage at 66/150. Both still wrote an aggregate.
`CLAUDE.md` rule #2 (never cherry-pick numbers) and the standing debt item
("eval runner swallows infra failures") both point here. This classifier lets a
runner abort the instant it sees an unrecoverable error, so a billing outage can
never masquerade as a result.

Design notes
------------
* **Real signatures, not guesses.** The markers are taken from *observed* failures
  — the committed error strings in ``eval_results/*.json`` ("Your credit balance is
  too low to access the Anthropic API...") — and from what ``ingest/embed.py``
  already fail-fasts on (``insufficient_quota``). See rule #1 (no invented APIs).
* **Version-robust.** It keys off the ``status_code`` attribute (set on every
  ``APIStatusError`` instance by both SDKs) and the exception's *text*, NOT exact
  SDK exception classes — those drift between anthropic/openai releases (the
  installed versions already differ from the lockfile).
* **Pure function of the exception**, so it is trivially unit-testable with
  synthetic errors that mimic the SDK shapes, with no network.

The key subtlety, and why a plain status-code check is not enough:
  * Anthropic credit depletion is a **400** ``BadRequestError`` (not a 401/402) with
    a "credit balance" message — caught by the *message* markers, not the status.
  * OpenAI quota-out is a **429** that is byte-for-byte a transient rate limit
    except for the ``insufficient_quota`` marker — so 429 is deliberately NOT in the
    fatal-status set; only a 429 carrying a billing marker is fatal.
"""

from __future__ import annotations

from collections.abc import Iterator

# Substrings marking an account/billing/auth failure — permanent until a human tops
# up credits, fixes a key, or lifts a quota. Matched case-insensitively against the
# whole exception chain's string form. Lower-cased here so the check is a plain `in`.
_FATAL_MARKERS: tuple[str, ...] = (
    "credit balance",               # Anthropic: "Your credit balance is too low..."
    "insufficient_quota",           # OpenAI: out of credits (a 429, but permanent)
    "exceeded your current quota",  # OpenAI wording variant
    "billing_hard_limit_reached",   # OpenAI hard cap
    "plans & billing",              # Anthropic billing pointer
    "plan and billing",             # OpenAI billing pointer
    "invalid x-api-key",            # Anthropic: bad/rotated key
    "invalid_api_key",              # OpenAI
    "authentication_error",         # either SDK, key rejected
)

# HTTP status codes never worth retrying within a run: the credential or permission
# is wrong, not the network. 402 = payment required. NB **400 is deliberately absent**
# — Anthropic's billing error is *also* a 400, so billing is caught by the message
# markers above, and a generic 400 stays "transient" (surfaced via n_errors, not an
# abort). 429 is absent too: a plain rate limit is transient (the --sleep throttle is
# the remedy); only a 429 with a billing marker above is fatal.
_FATAL_STATUS: frozenset[int] = frozenset({401, 402, 403})


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception it wraps (``__cause__`` / ``__context__``).

    The engine may re-raise an SDK error wrapped in a ``ValueError`` or similar, so
    the billing signal can live one or two levels down. Cycle-guarded by identity."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def fatal_reason(exc: BaseException) -> str | None:
    """Return a short human reason if ``exc`` is unrecoverable (the run should abort),
    else ``None`` (transient — a bounded retry / skip-and-continue is appropriate).

    Fatal = an account/billing/auth failure that every subsequent question would hit
    identically. Inspects the full ``__cause__``/``__context__`` chain."""
    for e in _chain(exc):
        text = str(e).lower()
        for marker in _FATAL_MARKERS:
            if marker in text:
                return f"account/billing/auth failure (matched {marker!r}) — not transient"
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and status in _FATAL_STATUS:
            return (
                f"HTTP {status} ({type(e).__name__}) — credential/permission failure, "
                "not transient"
            )
    return None
