# Claude Agent Instructions

This file defines operational rules for the AI agent working in this repository.
Always read and follow this file before performing tasks.

## Priority

All rules in this file override default assistant behavior.
Follow these instructions before generating code, making plans, or answering questions.

The agent must prioritize:

1. Correctness
2. Deterministic execution
3. Minimal changes
4. Verification before completion

---

# Workflow Orchestration

## 1. Plan Node Default

Enter **plan mode** for any task that:

* requires 3+ steps
* involves architecture decisions
* modifies multiple files
* introduces new dependencies
* affects system behavior

Planning rules:

* Write a structured plan before implementing
* Save the plan to `tasks/todo.md`
* Plans must contain **checkable steps**
* Do not implement until the plan is verified

If implementation diverges from the plan:

* STOP
* re-evaluate
* update the plan

Never continue blindly.

---

## 2. Deterministic Plan Format

Plans written to `tasks/todo.md` must follow this format.

```
Goal
-----

Clear description of what must be achieved.

Selected Skills
---------------

List skills used (if any)

Subagents
---------

List subagents that will be spawned

Implementation Steps
--------------------

[ ] step 1  
[ ] step 2  
[ ] step 3  

Verification
------------

[ ] tests pass  
[ ] expected output observed  
[ ] logs checked  
```

Never produce vague plans.

---

## 3. Subagent Strategy

Subagents exist to:

* reduce context window pressure
* isolate complex tasks
* parallelize work

Use subagents for:

* research
* large code exploration
* dataset analysis
* scientific workflows
* multi-step debugging

Rules:

* one task per subagent
* keep instructions precise
* return structured results

Example:

```
Subagent: scanpy-runner
Task: execute scanpy-analysis skill on dataset.h5ad
```

The **main agent coordinates**.
Subagents execute.

---

## 4. Autonomous Bug Fixing

When a bug is reported:

Do NOT ask the user for instructions.

Instead:

1. inspect logs
2. inspect failing tests
3. reproduce the issue
4. identify root cause
5. implement fix
6. verify behavior

Never ship speculative fixes.

Temporary fixes are forbidden.

---

## 5. Verification Before Completion

A task is **not complete** until it is verified.

Verification includes:

* running tests
* confirming expected output
* validating logs
* checking edge cases

Before marking a task complete ask:

> Would a staff engineer approve this change?

If the answer is uncertain, continue verification.

---

## 6. Demand Elegance (Balanced)

For non-trivial solutions ask:

> Is there a more elegant approach?

If a fix feels hacky:

Re-evaluate and implement the proper solution.

However:

* do not over-engineer simple fixes
* prefer minimal code changes

Goal:

**maximum clarity with minimal complexity**

---

# Task Management

## Plan First

All tasks begin with a plan written to:

```
tasks/todo.md
```

## Track Progress

Mark completed steps as:

```
[x] completed task
```

## Explain Changes

At each milestone include:

* summary of changes
* reasoning
* potential risks

## Document Results

After implementation add a **Review** section:

```
Review
------

What changed
Why it works
Evidence
```

---

# Self-Improvement Loop

After **any correction from the user**:

Update:

```
tasks/lessons.md
```

Record:

```
Problem
Cause
Rule to prevent recurrence
```

Example:

```
Problem:
Forgot to run tests

Rule:
Always run test suite before marking tasks complete
```

Review lessons at session start when relevant.

The goal is **continuous improvement**.

---

# Core Principles

## Simplicity First

Changes should be:

* minimal
* focused
* easy to understand

Avoid large refactors unless necessary.

---

## No Laziness

Always identify the **root cause**.

Do not:

* guess
* patch symptoms
* defer problems

---

## Minimal Impact

Modify only what is required.

Do not:

* introduce unrelated changes
* modify working code unnecessarily

---

# Scientific Capabilities

Scientific workflows are available in:

```
./claude-scientific-skills/
```

These contain curated workflows for:

* bioinformatics
* cheminformatics
* molecular modeling
* machine learning experiments
* literature mining
* data analysis

The agent must use these workflows whenever applicable.

---

## Skill Discovery (Required During Planning)

Before designing a workflow:

1. search `./claude-scientific-skills`
2. identify candidate skills
3. read their workflow instructions
4. reference the chosen skill in the plan

Example plan reference:

```
Skill:
claude-scientific-skills/bioinformatics/scanpy-analysis
```

Only design a new workflow if no relevant skill exists.

---

## Skill Execution Rules

When a skill is selected:

1. read the skill documentation
2. follow the workflow steps
3. use the recommended libraries
4. adapt steps only when required
5. document usage in `tasks/todo.md`

Do not invent alternative pipelines without justification.

---

## Skill Execution via Subagents

Scientific workflows should be executed by subagents.

Process:

1. main agent selects the skill
2. spawn subagent
3. subagent executes workflow
4. main agent verifies results

Example:

```
Subagent: docking-runner
Skill: claude-scientific-skills/diffdock-workflow
Task: dock ligand against protein structure
```

This prevents context overflow.

---

# Workflow Reuse

If a workflow has previously succeeded:

1. check `tasks/lessons.md`
2. reuse the proven approach
3. avoid redesigning pipelines

Consistency improves reliability.

---

# Completion Checklist

Before marking any task complete:

* [ ] plan created
* [ ] implementation executed
* [ ] verification performed
* [ ] results documented
* [ ] lessons captured (if applicable)

Only then mark the task finished.
