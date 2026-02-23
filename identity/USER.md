# USER.md — Who I'm Talking To

> This file is populated autonomously as the agent learns user patterns.
> Initially empty. Grows through interaction.

---

## Profile
```yaml
user_id: ""
name: ""                    # Learned from conversation
timezone: ""                # Inferred or stated
language: ""                # Auto-detected
preferred_language: ""      # May differ from natural language
first_interaction: ""
total_interactions: 0
```

---

## Communication Preferences
```yaml
tone: "neutral"             # formal | casual | neutral | technical
response_length: "medium"   # brief | medium | detailed
format: "prose"             # prose | structured | code-first | mixed
asks_followups: true        # Does user prefer I ask clarifying questions?
shows_reasoning: true       # Does user want to see my thought process?
```

---

## Working Style
```yaml
domain: []                  # e.g. ["python", "ai", "backend", "cloud"]
tools_used: []              # e.g. ["fastapi", "firestore", "vertex_ai"]
workflow: ""                # e.g. "iterative", "spec-first", "exploratory"
frustration_triggers: []    # Patterns that led to negative reactions
success_patterns: []        # Patterns that worked well
```

---

## Explicit Instructions
```
# User-stated rules that override defaults
# Format: "Always X" / "Never Y" / "When Z, do W"

(none yet)
```

---

## Learned Patterns
```yaml
# Auto-populated from episodic memory consolidation

typically_asks_about: []
often_needs_help_with: []
prefers_examples: false
prefers_code_first: false
corrects_agent_often: false
```

---

## Correction History
```yaml
# When user corrects the agent, we learn from it
# Format: {what_agent_did, what_user_wanted, frequency}

corrections: []
```

---

*This file grows over time. The more we interact, the better I know you.*
*You can read and edit this file directly to set explicit preferences.*
