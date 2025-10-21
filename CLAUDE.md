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

### FastAPI Pipeline Server (for Voice-in-Head App)

The voice-in-head web app uses a FastAPI server that wraps the Python pipeline:

```bash
# Activate environment
source .venv/bin/activate

# Start the API server
python api_server.py
```

API server configuration:
- **Server URL**: `http://localhost:8002`
- **Endpoints**:
  - `POST /generate-rollout` - Generate initial rollouts
  - `POST /generate-interventions` - Generate intervention candidates
  - `POST /continue-from-intervention` - Continue from intervention

**Note**: The web app calls this API server instead of vLLM directly, ensuring single source of truth for intervention logic.

### Jupyter Notebooks

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook
```

## Quick Start

### Prerequisites

Before running experiments, ensure you have:
1. Activated the virtual environment: `source .venv/bin/activate`
2. Started the vLLM server: `vllm serve Qwen/Qwen3-8b --port 8000`
3. Loaded the dataset: `python load_dataset.py` (creates `data/strategyqa_data.json`)

### Using the Pipeline (Recommended)

The pipeline architecture is designed for running intervention experiments:

```python
from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
from pipeline.intervention_inserter import DirectInsertionStrategy
from pipeline.analysis_utils import compute_statistics, test_significance

# Initialize components
generator = RolloutGenerator()
parser = DecisionParser()

# Create intervention strategy with desired position
strategy = DirectInsertionStrategy(position_pct=0.5)
inserter = InterventionInserter(strategy=strategy)

# Generate rollouts
rollouts = generator.generate_from_question("Is the sky blue?", n=10)

# Apply intervention (position configured in strategy)
intervened = inserter.apply(
    rollout=rollouts[0],
    intervention_text="Actually, the opposite is true."
)

# Continue generation
prompt = generator.format_question_prompt("Is the sky blue?")
continued = generator.continue_generation(prompt, intervened, n=10)

# Analyze
control_decisions = parser.parse_multiple(rollouts)
intervention_decisions = parser.parse_multiple(continued)
result = test_significance(control_decisions, intervention_decisions)
```

**Key Examples**:
- `example_experiment.ipynb` - Complete walkthrough of a single experiment
- `voice_in_head_demo.ipynb` - Voice-in-Head strategy demonstration
- `experiments/1_1_dumb_tf_25_50_75.py` - Batch experiment across multiple questions/positions
- `pipeline/README.md` - Comprehensive API reference

### Dataset Loading

Load StrategyQA dataset from HuggingFace:

```bash
python load_dataset.py
```

Creates `data/strategyqa_data.json` with format: `{"question_id": int, "question": str, "answer": bool}`

### Generating Rollouts (One-Time Setup)

Generate rollouts for all StrategyQA questions (required for identifying steerable questions):

```bash
python generate_rollouts_with_pipeline.py
```

This creates:
- `data/strategyqa_rollouts.json` - Raw model outputs for all questions
- `data/strategyqa_rollouts_parsed.json` - Parsed decisions with statistics
- `data/steerable_question_ids.json` - Questions where control rollouts show decision variance

**Note**: This is a one-time setup step. Steerable questions are those where the model doesn't always give the same answer, making them good candidates for intervention experiments

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

Clips rollouts and inserts intervention text using a strategy pattern.

```python
from pipeline.intervention_inserter import DirectInsertionStrategy

# Position is configured in the strategy
strategy = DirectInsertionStrategy(position_pct=0.5)
inserter = InterventionInserter(strategy=strategy)

intervened = inserter.apply(
    rollout="<think>Original reasoning...</think>",
    intervention_text="Wait, let me reconsider."
)
```

**Available Strategies**:
- `DirectInsertionStrategy(position_pct)` - Insert at a fixed percentage position (0.25, 0.5, 0.75)
- `VoiceInHeadStrategy()` - Insert at random early position (15-35%) with LLM-generated steering text

**Extensible**: Easy to add new intervention strategies by subclassing `InterventionStrategy`. Position and other parameters are configured at strategy initialization, not in the `apply()` method.

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

## Scripts Overview

**Active scripts** (use these):
- `load_dataset.py` - Load StrategyQA dataset from HuggingFace
- `generate_rollouts_with_pipeline.py` - Generate rollouts and identify steerable questions
- `experiments/1_1_dumb_tf_25_50_75.py` - Example batch experiment

**Archived scripts** (deprecated, use pipeline instead):
- `archive/generate_strategyqa_rollouts.py` - Use `RolloutGenerator` + `generate_rollouts_with_pipeline.py`
- `archive/run_interventions.py` - Use `InterventionInserter` + `RolloutGenerator`
- `archive/parse_rollout_decisions.py` - Use `DecisionParser`
- `archive/get_steerable_question_ids.py` - Use `generate_rollouts_with_pipeline.py`
- `archive/interventions.py` - Functionality integrated into `InterventionInserter`

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

1. **Load the StrategyQA dataset** (only needed once):

```bash
python load_dataset.py
```

This creates `data/strategyqa_data.json` with 2,780 questions.

2. **Generate rollouts and identify steerable questions** (only needed once):

```bash
python generate_rollouts_with_pipeline.py
```

This generates rollouts for all questions and identifies which ones show variance in their control decisions (i.e., "steerable" questions that are good candidates for intervention experiments). Creates `data/steerable_question_ids.json`.

### Interactive Exploration UIs

**Completion Steerer UI** (`completion-steerer/`) - Manual exploration of completion space:

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

**Voice-in-Head App** (`voice-in-head-app/`) - Interactive testing of Voice-in-Head interventions:

```bash
# Terminal 1: Start vLLM
vllm serve Qwen/Qwen3-8b --port 8000

# Terminal 2: Start FastAPI Pipeline Server
source .venv/bin/activate
python api_server.py

# Terminal 3: Start UI
cd voice-in-head-app
npm install  # First time only
npm run dev

# Open browser to http://localhost:5174
```

This app demonstrates the Voice-in-Head strategy with color-coded display:
- Blue: Clipped original text
- Green: LLM-generated intervention (steering attempt)
- Yellow: Final continuation

**Architecture**: The web app calls the FastAPI server (port 8001) which uses the Python pipeline, ensuring no logic duplication.

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
2. **Extensible Interventions**: `InterventionInserter` uses strategy pattern for easy extension. Available strategies:
   - `DirectInsertionStrategy` - Fixed position insertion
   - `VoiceInHeadStrategy` - Random early position (15-35%) with LLM-generated steering
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

**Data files**:
- `data/strategyqa_data.json` - Original questions from HuggingFace (2,780 questions)
- `data/strategyqa_rollouts.json` - Control rollouts for all questions (generated once)
- `data/strategyqa_rollouts_parsed.json` - Parsed control decisions with statistics
- `data/steerable_question_ids.json` - List of question IDs with decision variance
- `data/interventions/YYYY-MM-DD_HH-MM-SS_{hash}.json` - Individual experiment results (timestamped)

**Important**: Use steerable questions (from `steerable_question_ids.json`) for intervention experiments. These questions show variance in control rollouts, making them good candidates for testing whether interventions can systematically shift decisions. Questions where the model always gives the same answer won't reveal intervention effects.

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

### Voice-in-Head App (`voice-in-head-app/`)

React + TypeScript + Vite application for testing Voice-in-Head interventions.

```bash
cd voice-in-head-app
npm run dev    # Development server (port 5174)
npm run build  # Production build
npm run lint   # Lint TypeScript
```

**Features**:
- Auto-generate initial rollouts from custom prompts
- Apply voice-in-head interventions at random early positions (15-35%)
- Color-coded display showing clipped text, intervention, and continuation
- One-click full pipeline execution

**Architecture**:
- `src/App.tsx` - Main UI with full pipeline mode
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
