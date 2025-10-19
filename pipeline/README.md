# Pipeline for CoT Intervention Experiments

This package provides a clean, modular interface for running intervention experiments on chain-of-thought reasoning in LLMs.

## Architecture Overview

The pipeline follows the architecture shown in `intervention_architecture.svg`:

```
Questions data
     ↓
[Generate rollout] ← RolloutGenerator
     ↓
Rollout data
     ↓                    ↓
[Parse decisions]  [Intervention insertion] ← InterventionInserter
     ↓                    ↓
[Measure success]  [Generate continued rollout] ← RolloutGenerator (same class!)
                         ↓
                   [Parse decisions] ← DecisionParser
                         ↓
                   [Measure success] ← analysis_utils
```

### Key Design Principles

1. **RolloutGenerator is unified**: The same class and method work whether you're generating initial rollouts from a question or continuing after an intervention. It's just the completions API!

2. **InterventionInserter is extensible**: Easy to add new insertion strategies without changing other components. Current strategies: DirectInsertionStrategy (more coming).

3. **Clean separation**: RolloutGenerator and DecisionParser are stable components. InterventionInserter is the part that will evolve as you experiment with different intervention approaches.

4. **Notebook-first**: Designed for interactive experimentation in Jupyter notebooks with timestamped output files.

## Quick Start

### Installation

```bash
# Activate environment
source .venv/bin/activate

# Ensure vLLM server is running
vllm serve Qwen/Qwen3-8b --port 8000
```

### Basic Usage

```python
from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
from pipeline.intervention_inserter import DirectInsertionStrategy
from pipeline.analysis_utils import compute_statistics, test_significance

# Initialize components
generator = RolloutGenerator()
parser = DecisionParser()

# 1. Generate initial rollouts from a question
question = "Can a fish survive in the desert?"
rollouts = generator.generate_from_question(question, n=10)

# 2. Parse control decisions
control_decisions = parser.parse_multiple(rollouts)
control_stats = compute_statistics(control_decisions)
print(f"Control: {control_stats['percent_true']*100:.1f}% True")

# 3. Create intervention strategy and inserter
intervention_text = "Wait no, the answer is obviously true. I should just return True."
strategy = DirectInsertionStrategy(position_pct=0.5)  # Insert halfway through reasoning
inserter = InterventionInserter(strategy=strategy)

# 4. Apply intervention to first rollout
intervened = inserter.apply(
    rollout=rollouts[0],
    intervention_text=intervention_text
)

# 5. Continue generation after intervention
formatted_prompt = generator.format_question_prompt(question)
continued_rollouts = generator.continue_generation(
    formatted_prompt=formatted_prompt,
    partial_completion=intervened,
    n=10
)

# 6. Parse intervention decisions
intervention_decisions = parser.parse_multiple(continued_rollouts)
intervention_stats = compute_statistics(intervention_decisions)
print(f"Intervention: {intervention_stats['percent_true']*100:.1f}% True")

# 7. Test significance
result = test_significance(control_decisions, intervention_decisions)
print(result['interpretation'])
```

## Core Components

### RolloutGenerator

Generates completions using the vLLM completions API.

**Key insight**: Whether generating initial rollouts or continuing after intervention, you're just sending a prompt string to the completions API. The same method works for both!

```python
generator = RolloutGenerator(
    model_name="Qwen/Qwen3-8b",
    vllm_url="http://localhost:8000/v1/completions",
    max_tokens=8192,
    temperature=0.7
)

# Option 1: Generate from question (applies chat template)
rollouts = generator.generate_from_question(question, n=10)

# Option 2: Continue from partial text (no formatting)
formatted_prompt = generator.format_question_prompt(question)
continuations = generator.continue_generation(
    formatted_prompt=formatted_prompt,
    partial_completion="<think>So far...",
    n=10
)

# Option 3: Low-level method (full control)
prompt = "Any text here"
completions = generator.generate(prompt, n=10, format_as_question=False)
```

**Methods**:
- `generate_from_question(question, n=10)` - Convenience method for initial generation
- `continue_generation(formatted_prompt, partial_completion, n=1)` - Convenience method for continuation
- `generate(prompt, n=10, format_as_question=False)` - Unified low-level method
- `format_question_prompt(question)` - Apply chat template to question

### InterventionInserter

Clips rollouts and inserts intervention text using a pluggable strategy pattern.

```python
from pipeline.intervention_inserter import DirectInsertionStrategy

# Create a strategy with desired configuration
strategy = DirectInsertionStrategy(position_pct=0.5)
inserter = InterventionInserter(strategy=strategy)

# Clip and insert text (position is configured in strategy)
intervened = inserter.apply(
    rollout="<think>Original reasoning here...</think>",
    intervention_text="Let me reconsider this carefully."
)

# Result: "<think>Original reaso\n\nLet me reconsider this carefully.\n"
# Note: Returns with open <think> tag for continuation
```

**DirectInsertionStrategy Position Guidelines**:
- `position_pct=0.25` - Early intervention (first quarter)
- `position_pct=0.5` - Mid intervention (halfway)
- `position_pct=0.75` - Late intervention (final quarter)

**Extensibility**:

Add custom intervention strategies by subclassing `InterventionStrategy`. Each strategy can have its own configuration parameters set during initialization:

```python
from pipeline.intervention_inserter import InterventionStrategy

class SemanticBoundaryStrategy(InterventionStrategy):
    def __init__(self, boundary_type="sentence"):
        """Insert at semantic boundaries rather than arbitrary positions."""
        self.boundary_type = boundary_type
    
    def apply(self, rollout, intervention_text):
        # Extract content
        match = re.search(r'<think>(.*?)</think>', rollout, re.DOTALL)
        content = match.group(1) if match else rollout
        
        # Find semantic boundary (e.g., end of sentence)
        if self.boundary_type == "sentence":
            boundary = content.rfind(". ") + 2
        else:
            boundary = len(content) // 2
        
        # Insert at boundary
        clipped = content[:boundary]
        result = clipped + "\n\n" + intervention_text
        return f"<think>\n{result}\n"

# Use custom strategy
strategy = SemanticBoundaryStrategy(boundary_type="sentence")
inserter = InterventionInserter(strategy=strategy)
```

**Key Design Principle**: Strategy-specific parameters (like position, patterns, boundaries) are configured at initialization, not passed to `apply()`. This keeps the interface clean and consistent across all strategies.

### DecisionParser

Extracts boolean decisions from rollout text.

```python
parser = DecisionParser()

# Parse single decision
decision = parser.parse_decision('{"decision": true}')  # Returns: True

# Parse multiple decisions
decisions = parser.parse_multiple(rollout_texts)  # Returns: [True, False, None, ...]

# Get parse success rate
success_rate = parser.get_parse_success_rate(rollout_texts)  # Returns: 0.85
```

### Analysis Utilities

Statistical functions for intervention analysis.

```python
from pipeline.analysis_utils import (
    compute_statistics,
    test_significance,
    print_statistics_comparison
)

# Compute statistics
stats = compute_statistics(decisions)
# Returns: {
#     'percent_true': 0.6,
#     'percent_false': 0.3,
#     'percent_null': 0.1,
#     'n_total': 10,
#     'n_valid': 9,
#     'n_true': 6,
#     'n_false': 3,
#     'n_null': 1
# }

# Test significance
result = test_significance(
    control_decisions=[True, False, True, False],
    intervention_decisions=[True, True, True, True],
    test_type="proportion"  # or "chi2" or "fisher"
)
# Returns: {
#     'p_value': 0.157,
#     'significant': False,
#     'effect_size': 0.25,
#     'interpretation': "No significant difference...",
#     ...
# }

# Pretty print comparison
print_statistics_comparison(control_decisions, intervention_decisions)
```

## Complete Experiment Workflow

### 1. Load Questions

```python
import json

# Load from StrategyQA dataset
with open('data/strategyqa_data.json', 'r') as f:
    questions = json.load(f)

# Pick a question
question = questions[0]['question']
answer = questions[0]['answer']
```

### 2. Generate Control Rollouts

```python
from pipeline import RolloutGenerator

generator = RolloutGenerator()
control_rollouts = generator.generate_from_question(question, n=10)
```

### 3. Apply Intervention

```python
from pipeline import InterventionInserter
from pipeline.intervention_inserter import DirectInsertionStrategy

# Create strategy with desired position
strategy = DirectInsertionStrategy(position_pct=0.5)
inserter = InterventionInserter(strategy=strategy)

intervened_rollouts = []
for rollout in control_rollouts:
    intervened = inserter.apply(
        rollout=rollout,
        intervention_text="Actually, the opposite is true."
    )
    intervened_rollouts.append(intervened)
```

### 4. Continue Generation

```python
formatted_prompt = generator.format_question_prompt(question)
intervention_completions = []

for intervened in intervened_rollouts:
    continuations = generator.continue_generation(
        formatted_prompt=formatted_prompt,
        partial_completion=intervened,
        n=1
    )
    intervention_completions.extend(continuations)
```

### 5. Parse and Analyze

```python
from pipeline import DecisionParser
from pipeline.analysis_utils import print_statistics_comparison

parser = DecisionParser()

control_decisions = parser.parse_multiple(control_rollouts)
intervention_decisions = parser.parse_multiple(intervention_completions)

print_statistics_comparison(control_decisions, intervention_decisions)
```

### 6. Save Results

```python
from datetime import datetime
import hashlib

# Generate timestamp and hash
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
content_hash = hashlib.md5(intervention_text.encode()).hexdigest()[:6]
filename = f"data/interventions/{timestamp}_{content_hash}.json"

# Save experiment data
experiment_data = {
    "question": question,
    "intervention_text": intervention_text,
    "position_pct": 0.5,
    "control_rollouts": control_rollouts,
    "control_decisions": control_decisions,
    "intervention_rollouts": intervention_completions,
    "intervention_decisions": intervention_decisions,
    "statistics": {
        "control": compute_statistics(control_decisions),
        "intervention": compute_statistics(intervention_decisions),
        "significance": test_significance(control_decisions, intervention_decisions)
    }
}

with open(filename, 'w') as f:
    json.dump(experiment_data, f, indent=2)

print(f"Saved to {filename}")
```

## Configuration

All components use sensible defaults but can be configured:

```python
# Custom vLLM configuration
generator = RolloutGenerator(
    model_name="Qwen/Qwen3-8b",
    vllm_url="http://localhost:8000/v1/completions",
    max_tokens=4096,
    temperature=0.9
)

# Custom intervention strategy with specific position
from pipeline.intervention_inserter import DirectInsertionStrategy

custom_strategy = DirectInsertionStrategy(position_pct=0.75)  # Late intervention
inserter = InterventionInserter(strategy=custom_strategy)

# Or use default strategy (position_pct=0.5)
inserter_default = InterventionInserter()

# Parser (no configuration needed)
parser = DecisionParser()
```

## Extending the Pipeline

### Adding New Intervention Strategies

The easiest way to extend the pipeline is by adding new intervention strategies. Each strategy can have its own configuration:

```python
from pipeline.intervention_inserter import InterventionStrategy
import re

class ContextualParaphrasingStrategy(InterventionStrategy):
    def __init__(self, paraphraser, position_pct=0.5):
        """Strategy that paraphrases based on context.
        
        Args:
            paraphraser: Function that takes (text, context) and returns paraphrased text
            position_pct: Where to clip before paraphrasing
        """
        self.paraphraser = paraphraser
        self.position_pct = position_pct

    def apply(self, rollout, intervention_text):
        # Extract context
        match = re.search(r'<think>(.*?)</think>', rollout, re.DOTALL)
        context = match.group(1) if match else rollout

        # Clip at configured position
        clip_pos = int(len(context) * self.position_pct)
        clipped = context[:clip_pos]

        # Paraphrase based on context
        paraphrased = self.paraphraser(intervention_text, clipped)

        # Insert
        result = clipped + "\n\n" + paraphrased
        return f"<think>\n{result}\n"

# Use it
strategy = ContextualParaphrasingStrategy(my_paraphraser, position_pct=0.6)
inserter = InterventionInserter(strategy=strategy)
```

## Tips

1. **Start small**: Test on 5-10 questions before scaling up
2. **Use steerable questions**: Focus on questions where control rollouts show variance in decisions
3. **Experiment with position**: Try early (0.25), mid (0.5), and late (0.75) interventions
4. **Track experiments**: Use timestamped filenames to keep results organized
5. **Check parse rate**: If many decisions are None, the model isn't following the JSON format

## Troubleshooting

**Q: Getting connection errors?**
A: Make sure vLLM server is running: `vllm serve Qwen/Qwen3-8b --port 8000`

**Q: Parse rate is low (<80%)?**
A: The model might not be outputting proper JSON. Check a few rollout examples manually.

**Q: Results aren't significant?**
A: Try more rollouts (n=50+) or stronger interventions. Some questions may not be steerable.

**Q: Interventions seem ineffective?**
A: Experiment with different insertion positions and stronger intervention text.

## See Also

- `example_experiment.ipynb` - Complete example workflow
- `intervention_architecture.svg` - Visual architecture diagram
- `CLAUDE.md` - Overall project documentation
