# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores the basic science of Chain-of-Thought (CoT) interventions in LLMs through interactive completion steering. The project consists of two main components:

1. **Completion Steerer UI** - A React-based interactive interface for manually steering LLM outputs by choosing from multiple completion candidates at each generation step
2. **Steering Vector Selector** - A Python tool for automatically selecting completions that maximize activation along emotion-based steering vectors

## Development Setup

### Python Environment

The project uses a Python virtual environment with extensive ML dependencies including PyTorch, transformers, vLLM, and nnsight.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### vLLM Server

Most experiments require a running vLLM server. The default configuration expects:
- **Server URL**: `http://localhost:8000`
- **Model**: Qwen/Qwen3-8b for generation (some scripts use Qwen/Qwen3-0.6b or Qwen/Qwen3-4b for embeddings)
- **Port**: 8000

Start the server:
```bash
vllm serve Qwen/Qwen3-8b --port 8000
```

### Completion Steerer UI

The UI is a Vite + React + TypeScript application located in `completion-steerer/`.

```bash
cd completion-steerer

# Install dependencies
npm install

# Start development server (runs on localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Key Components

### Completion Steerer UI (`completion-steerer/`)

**Purpose**: Interactive interface for manually steering LLM generation by selecting from 10 completion candidates at each step.

**Architecture**:
- `src/App.tsx` - Main application component with state management for multi-step completion selection
- `src/lib/vllm-api.ts` - API integration with vLLM server at `localhost:8000`
- `src/lib/utils.ts` - Utility functions
- `src/components/ui/` - shadcn/ui components (Button, Card, Textarea, Badge)

**Key Features**:
- Generates 10 completion options per step using vLLM's `n=10` parameter
- Uses Qwen3 chat template formatting: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- Supports "time travel" - change any previous selection and continue from that branch
- Edit mode - manually edit generated text and continue generation from edited state
- Step history tracking with alternative choice visualization

**API Format** (vllm-api.ts:21-30):
```typescript
{
  model: "Qwen/Qwen3-8b",
  prompt: formatPrompt(userPrompt) + previousText,
  max_tokens: 50,
  temperature: 0.7,
  n: 10
}
```

### Steering Vector Selector (`completion-steerer/steering_selector.py`)

**Purpose**: Automatically select completions that maximize activation along emotion-based steering vectors (angry vs neutral).

**Architecture**:
- Uses layer 15 embeddings from Qwen3-4b for computing steering scores
- Generates completions via vLLM server (Qwen3-8b)
- Computes steering vector as `angry_vector - neutral_vector` from emotion examples
- Scores each completion by projection onto steering direction

**Usage**:
```bash
python completion-steerer/steering_selector.py \
  --prompt "What are the most common emotions?" \
  --max-tokens 100 \
  --temperature 0.7 \
  --n 10 \
  --layer 15 \
  --embedding-model "Qwen/Qwen3-4b" \
  --vllm-model "Qwen/Qwen3-8b"
```

**Key Methods** (steering_selector.py):
- `_get_embedding()` - Extract normalized mean-pooled embeddings from specified layer
- `compute_steering_scores()` - Project embeddings onto steering vector
- `select_best_completion()` - Generate n completions and return highest-scoring one

**Dependencies**:
- Requires `emotion_examples.json` at `/workspace/basic_science_resampling/emotion_examples.json`
- Contains pre-computed angry and neutral emotion examples for steering vector computation

### Initial Exploration (`initial_rough/`)

Contains early experimental notebooks and scripts:
- `steering_demo.ipynb` - Demonstrations of steering concepts
- `mvp.ipynb` - Minimum viable product explorations
- `create_emotion_dataset.py` / `generate_emotion_dataset_from_vllm.py` - Dataset generation scripts
- `emotion_examples.json` / `emotion_score_results.json` - Generated emotion datasets

## Common Workflows

### Running the Interactive Steerer

1. Start vLLM server in one terminal:
   ```bash
   vllm serve Qwen/Qwen3-8b --port 8000
   ```

2. Start the UI in another terminal:
   ```bash
   cd completion-steerer
   npm run dev
   ```

3. Open browser to `http://localhost:5173`
4. Enter a prompt, adjust parameters, and interactively steer the generation

### Using Automatic Steering Selection

1. Ensure vLLM server is running
2. Run the steering selector:
   ```bash
   python completion-steerer/steering_selector.py --prompt "Your prompt here"
   ```

### Working with Jupyter Notebooks

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook

# Navigate to initial_rough/ for exploration notebooks
```

## Technical Architecture

### Multi-Step Completion Selection

The core pattern used throughout this project:

1. **Format Prompt** - Apply chat template to user input
2. **Generate n Completions** - Use vLLM with `n=10` to generate multiple options
3. **Score/Select** - Either manually (UI) or automatically (steering vectors)
4. **Continue** - Append selected text and repeat from step 2

State management in UI (App.tsx:22-24):
```typescript
const [completionSteps, setCompletionSteps] = useState<CompletionStep[]>([]);
const [currentChoices, setCurrentChoices] = useState<CompletionChoice[]>([]);
const [isSelectingFromChoices, setIsSelectingFromChoices] = useState(false);
```

### Embedding-Based Steering

Steering vector computation (steering_selector.py:72-79):
1. Load angry/neutral emotion examples from `emotion_examples.json`
2. Compute mean-pooled layer 15 embeddings for each set
3. Normalize embeddings
4. Compute steering vector as `angry - neutral`
5. Normalize steering vector

Scoring (steering_selector.py:152-179):
1. Extract layer 15 embeddings for completion text
2. Project onto steering vector: `score = embedding · steering_vector`
3. Higher score = more aligned with angry direction

## Project Context

This is research exploring the basic science of how LLMs respond to interventions during chain-of-thought reasoning. The interactive steerer allows manual exploration of the completion space, while the automatic selector demonstrates how embeddings can guide selection toward specific semantic directions.

The project pivoted from earlier emotion dataset work to focus on the fundamental mechanisms of CoT interventions and completion resampling.
