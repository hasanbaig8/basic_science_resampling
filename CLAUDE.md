# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores the basic science of Chain-of-Thought (CoT) interventions in LLMs, specifically studying how injecting text at different positions during reasoning affects model outputs. The main focus is on **StrategyQA question answering** with systematic intervention experiments.

### Core Components

The project uses a clean, modular **pipeline architecture** (`pipeline/` directory):

1. **RolloutGenerator** - Generate completions using vLLM (unified for initial generation and continuation)
2. **InterventionInserter** - Clip reasoning traces and inject intervention text
3. **DecisionParser** - Extract boolean decisions from model outputs
4. **analysis_utils** - Statistical analysis and significance testing

All components are designed for interactive use in Jupyter notebooks.

## Development Setup

### Python Environment

The project uses a Python virtual environment with ML dependencies including PyTorch, transformers, vLLM, and nnsight.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### vLLM Server

**Critical**: All experiments require a running vLLM server.

```bash
# Start vLLM server (default configuration)
vllm serve Qwen/Qwen3-8b --port 8000
```

Default configuration:
- **Model**: Qwen/Qwen3-8b (8B parameter model)
- **Server URL**: `http://localhost:8000`
- **Endpoints**: Uses `/v1/completions` endpoint

### Jupyter Notebooks

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook
```

## Quick Start

### Using the Pipeline (Recommended)

The new pipeline architecture is designed for interactive experiments in Jupyter notebooks:

```python
from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
from pipeline.analysis_utils import compute_statistics, test_significance

# Initialize
generator = RolloutGenerator()
inserter = InterventionInserter()
parser = DecisionParser()

# Generate rollouts
rollouts = generator.generate_from_question("Is the sky blue?", n=10)

# Apply intervention
intervened = inserter.apply(
    rollout=rollouts[0],
    intervention_text="Actually, the opposite is true.",
    position_pct=0.5
)

# Continue generation
prompt = generator.format_question_prompt("Is the sky blue?")
continued = generator.continue_generation(prompt, intervened, n=10)

# Analyze
control_decisions = parser.parse_multiple(rollouts)
intervention_decisions = parser.parse_multiple(continued)
result = test_significance(control_decisions, intervention_decisions)
```

**See `example_experiment.ipynb` for a complete walkthrough!**

**Documentation**: `pipeline/README.md` contains comprehensive API reference and usage patterns.

### Dataset Loading

Load StrategyQA dataset from HuggingFace:

```bash
python load_dataset.py
```

Creates `data/strategyqa_data.json` with format: `{"question_id": int, "question": str, "answer": bool}`

## Pipeline Architecture

### Core Classes

**See `pipeline/README.md` for detailed API documentation.** Quick reference:

#### RolloutGenerator (`pipeline/rollout_generator.py`)

Unified class for generating completions - works for both initial generation and continuation.

```python
generator = RolloutGenerator()

# Initial generation from question
rollouts = generator.generate_from_question("Is the sky blue?", n=10)

# Continuation after intervention
prompt = generator.format_question_prompt("Is the sky blue?")
continued = generator.continue_generation(prompt, intervened_text, n=10)
```

**Key insight**: Both operations use the same underlying method - they're just calling the vLLM completions API!

#### InterventionInserter (`pipeline/intervention_inserter.py`)

Clips rollouts and inserts intervention text.

```python
inserter = InterventionInserter()

intervened = inserter.apply(
    rollout="<think>Original reasoning...</think>",
    intervention_text="Wait, let me reconsider.",
    position_pct=0.5  # 0.0-1.0, where 0.5 = halfway
)
```

**Extensible**: Easy to add new intervention strategies by subclassing `InterventionStrategy`.

#### DecisionParser (`pipeline/decision_parser.py`)

Extracts boolean decisions from model outputs.

```python
parser = DecisionParser()

# Parse single decision
decision = parser.parse_decision('{"decision": true}')  # Returns: True

# Parse multiple decisions
decisions = parser.parse_multiple(rollout_texts)  # Returns: [True, False, None, ...]
```

#### Analysis Utilities (`pipeline/analysis_utils.py`)

Statistical analysis functions.

```python
from pipeline.analysis_utils import compute_statistics, test_significance

# Compute % true, % false, % null
stats = compute_statistics(decisions)
print(stats['percent_true'])  # 0.75

# Test if intervention significantly changed outcomes
result = test_significance(control_decisions, intervention_decisions)
print(result['p_value'])        # 0.023
print(result['significant'])    # True
print(result['effect_size'])    # +0.25
```

## Old Scripts (Archived)

The original command-line scripts have been moved to `archive/` and superseded by the pipeline:

- `archive/generate_strategyqa_rollouts.py` - Use `RolloutGenerator` instead
- `archive/run_interventions.py` - Use `InterventionInserter` + `RolloutGenerator` instead
- `archive/parse_rollout_decisions.py` - Use `DecisionParser` instead
- `archive/get_steerable_question_ids.py` - Use `analysis_utils.compute_statistics()` instead
- `archive/interventions.py` - Functionality integrated into `InterventionInserter`

These are kept for reference but the pipeline is the recommended approach.

## Common Workflows

### Complete Experiment (Using Pipeline)

The recommended workflow using the pipeline in a Jupyter notebook:

1. **Start vLLM server**: `vllm serve Qwen/Qwen3-8b --port 8000`
2. **Open `example_experiment.ipynb`** - Contains complete walkthrough
3. **Customize intervention text and parameters**
4. **Run cells to execute experiment**
5. **Results saved to `data/interventions/{timestamp}_{hash}.json`**

See `example_experiment.ipynb` for detailed step-by-step workflow.

### One-Time Setup

Load the StrategyQA dataset (only needed once):

```bash
python load_dataset.py
```

This creates `data/strategyqa_data.json` with 2,780 questions.

### Interactive Exploration (Completion Steerer UI)

For manual exploration of the completion space (optional):

```bash
# Terminal 1: Start vLLM
vllm serve Qwen/Qwen3-8b --port 8000

# Terminal 2: Start UI
cd completion-steerer
npm install  # First time only
npm run dev

# Open browser to http://localhost:5173
```

The UI allows selecting from 10 completion candidates at each generation step, with "time travel" to explore alternative branches.

## Technical Architecture

### Visual Architecture

See `intervention_architecture.svg` for the complete flow diagram. The pipeline implements this architecture:

```
Questions → [RolloutGenerator] → Rollouts
                                     ↓
                   ┌─────────────────┴────────────────┐
                   ↓                                  ↓
            [DecisionParser]              [InterventionInserter]
                   ↓                                  ↓
             Parse decisions              Clip + insert text
                   ↓                                  ↓
            Compute stats                [RolloutGenerator] ← Same class!
                                                     ↓
                                          [DecisionParser]
                                                     ↓
                                          Test significance
```

### Key Design Principles

1. **Unified Generation**: `RolloutGenerator.generate()` works for both initial generation and continuation - it's just the vLLM completions API
2. **Extensible Interventions**: `InterventionInserter` uses strategy pattern for easy extension
3. **Clean Separation**: RolloutGenerator and DecisionParser are stable; InterventionInserter evolves with new strategies
4. **Notebook-First**: All components designed for interactive Jupyter experimentation

### Prompt Formatting

Pipeline uses tokenizer's chat template (rollout_generator.py:66-70):

```python
messages = [{"role": "user", "content": user_message}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # Adds assistant prefix
)
```

For Qwen3, this produces: `<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`

### Output Files

- `data/strategyqa_data.json` - Original questions from HuggingFace
- `data/interventions/YYYY-MM-DD_HH-MM-SS_{hash}.json` - Experiment results (timestamped)
  - Contains control rollouts, intervention rollouts, decisions, and statistical analysis

The pipeline automatically generates timestamped filenames for each experiment run.

## Additional Components

### Completion Steerer UI (`completion-steerer/`)

React + TypeScript + Vite application for interactive steering.

```bash
cd completion-steerer
npm run dev    # Development server
npm run build  # Production build
npm run lint   # Lint TypeScript
```

**Architecture**:
- `src/App.tsx` - Main UI with multi-step completion selection
- `src/lib/vllm-api.ts` - vLLM API integration
- `src/components/ui/` - shadcn/ui components

### Initial Exploration (`initial_rough/`)

Early prototypes and experiments:
- Emotion-based steering vectors (now deprecated in favor of StrategyQA focus)
- Notebooks for concept exploration
- Dataset generation scripts

## Project Context

This research explores how interventions in chain-of-thought reasoning affect LLM decision-making. The key question: **Can injecting specific text at different reasoning positions systematically change model outputs?**

The StrategyQA dataset provides yes/no questions where reasoning matters. By generating diverse rollouts, we identify "steerable" questions where decisions aren't deterministic, then test interventions to see if we can flip answers.
