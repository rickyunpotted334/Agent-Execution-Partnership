# FunctionGemma Integration

FunctionGemma is used only for structured action selection.

It receives task contract, observation summary, tool schemas, constraints, and risk budget.

It can output:

- typed action request
- request more observation
- request approval
- pause
- completion proposal
- escalation

Malformed outputs are rejected by strict schema parsing.
