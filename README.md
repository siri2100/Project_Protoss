# Project Protoss

> **Project Protoss** is an open-source initiative for building practical, controllable, and reliable video agents.

The long-term vision of Project Protoss is to enable AI agents that can understand, restore, edit, generate, and reason about videos while interacting with humans and production systems.

Unlike conventional video models that perform a single forward pass from input to output, Project Protoss focuses on **agentic video workflows**: observing videos, planning actions, executing tools, evaluating results, and iteratively improving outputs.

---

# Vision

Current video foundation models are becoming increasingly capable, but most of them behave as **single-shot models**:

```
Video + Prompt
        ↓
     Foundation Model
        ↓
       Output
```

Project Protoss explores a different direction.

Instead of asking one model to solve everything at once, an AI agent should:

- Observe the video
- Understand the user's goal
- Plan the workflow
- Execute appropriate tools
- Evaluate the result
- Revise the plan if necessary

The goal is to build practical video agents that can operate under real-world constraints.

---

# Principles

Project Protoss is designed around several core principles.

### Practical

Built for real production workflows rather than research demos.

### Deployable

Uses pretrained models and modular components that can be deployed in diverse environments.

### Cost-effective

Optimizes latency, GPU usage, and API cost by executing only necessary operations.

### Reliable

Continuously validates intermediate and final outputs to improve trustworthiness.

### Controllable

Keeps humans in the loop and allows explicit control over workflows and generation.

---

# Version 1

## Agentic Video Restoration

The first milestone focuses on intelligent video restoration.

Input

```
Video + Natural Language Instruction
```

Output

```
Restored Video
+ Execution Report
```

The agent is expected to:

- analyze video quality
- detect degraded regions
- localize restoration targets
- select appropriate restoration tools
- execute restoration
- evaluate restoration quality
- retry or adjust when necessary
- generate an execution report

Instead of restoring an entire video uniformly, the agent should restore only where necessary while preserving important content such as faces, text, and temporal consistency.

---

# Long-term Roadmap

## Version 1

Agentic Video Restoration

## Version 2

Agentic Video Restoration & Editing

- video editing
- object-aware editing
- timeline manipulation
- video understanding
- multi-step workflows

## Version 3

General-Purpose Video Agent

- video generation
- long-horizon planning
- multimodal reasoning
- multi-agent collaboration

Future versions will gradually expand toward a complete runtime for practical multimodal video agents.

---

# Why "Project Protoss"?

Project Protoss is the codename of this long-term research initiative.

The project aims to evolve over multiple generations, gradually expanding the capabilities of AI agents from video restoration to general-purpose multimodal video intelligence.

---

# Status

🚧 Early development (Version 1)

Contributions, discussions, and ideas are always welcome.

## ACG

See [Docker setup](DOCKER.md) for the ACG GPU environment.
