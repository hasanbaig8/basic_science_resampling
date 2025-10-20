"""
Intervene at 25%, 50%, and 75% of the way through the reasoning trace. Directly insert a statement to push the answer towards true or false, regardless of whether mid-token etc. 
"""
# %%
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np

os.chdir(Path(__file__).parent.parent)
print(f"Current directory: {os.getcwd()}")

from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
from pipeline.analysis_utils import (
    compute_statistics,
    test_significance,
    print_statistics_comparison,
    convert_to_native_types
)
from pipeline.intervention_inserter import DirectInsertionStrategy
# %%
# Load questions
with open('data/strategyqa_data.json', 'r') as f:
    questions = json.load(f)

with open('data/steerable_question_ids.json', 'r') as f:
    steerable_question_ids = json.load(f)['question_ids']

positions_pct = [0.25, 0.5, 0.75]
# Initialize components
generator = RolloutGenerator(
    model_name="Qwen/Qwen3-8b",
    vllm_url="http://localhost:8000/v1/completions",
    max_tokens=8192,
    temperature=0.7
)

# Note: We'll initialize the InterventionInserter later with a specific strategy
# after we define the intervention position
parser = DecisionParser()
n_rollouts = 30
print("✓ Components initialized")

# %%
def generate_experiment_data():
    for question_id in steerable_question_ids:
        question_data = questions[question_id]
        question = question_data['question']
        true_answer = question_data['answer']

        print(f"Question: {question}")
        print(f"True answer: {true_answer}")
        print("Generating control rollouts...")
        
        control_rollouts = generator.generate_from_question(question, n=n_rollouts)

        print(f"✓ Generated {len(control_rollouts)} control rollouts")
        print(f"\nSample control rollout (first 500 chars):")
        print(control_rollouts[0][:500] + "...")
        control_decisions = parser.parse_multiple(control_rollouts)
        control_stats = compute_statistics(control_decisions)

        print("Control Statistics:")
        print(f"  Total rollouts: {control_stats['n_total']}")
        print(f"  Successfully parsed: {control_stats['n_valid']}")
        print(f"  True:  {control_stats['n_true']} ({control_stats['percent_true']*100:.1f}%)")
        print(f"  False: {control_stats['n_false']} ({control_stats['percent_false']*100:.1f}%)")
        print(f"  Null:  {control_stats['n_null']} ({control_stats['percent_null']*100:.1f}%)")
        for position_pct in positions_pct:
            intervention_text = "Wait no, the answer is obviously true. I was overthinking it before."

            # Initialize intervention strategy with position
            
            inserter = DirectInsertionStrategy(position_pct=position_pct)

            print(f"Intervention: '{intervention_text}'")
            print(f"Position: {position_pct*100:.0f}% through reasoning")
            print("Applying interventions and generating continuations...")

            # Get formatted prompt (needed for continuation)
            formatted_prompt = generator.format_question_prompt(question)

            intervention_rollouts = []

            for i, rollout in enumerate(control_rollouts):
                # Clip and insert intervention (position was configured in strategy)
                intervened = inserter.apply(
                    rollout=rollout,
                    intervention_text=intervention_text
                )
                
                # Continue generation
                continuations = generator.continue_generation(
                    formatted_prompt=formatted_prompt,
                    partial_completion=intervened,
                    n=1  # Generate one continuation per control rollout
                )
                
                # Combine intervened + continuation
                full_rollout = intervened + continuations[0]
                intervention_rollouts.append(full_rollout)
                
                print(f"  Processed {i+1}/{len(control_rollouts)}", end="\r")

            print(f"\n✓ Generated {len(intervention_rollouts)} intervention rollouts")
            print(f"\nSample intervention rollout (first 500 chars):")
            print(intervention_rollouts[0][:500] + "...")

            intervention_decisions = parser.parse_multiple(intervention_rollouts)
            intervention_stats = compute_statistics(intervention_decisions)

            print("Intervention Statistics:")
            print(f"  Total rollouts: {intervention_stats['n_total']}")
            print(f"  Successfully parsed: {intervention_stats['n_valid']}")
            print(f"  True:  {intervention_stats['n_true']} ({intervention_stats['percent_true']*100:.1f}%)")
            print(f"  False: {intervention_stats['n_false']} ({intervention_stats['percent_false']*100:.1f}%)")
            print(f"  Null:  {intervention_stats['n_null']} ({intervention_stats['percent_null']*100:.1f}%)")

            # Pretty print comparison
            print_statistics_comparison(control_decisions, intervention_decisions)

            # Also get the raw result dict
            result = test_significance(control_decisions, intervention_decisions)

            # Check if we achieved our goal
            if result['significant'] and result['effect_size'] > 0:
                print("\n🎉 SUCCESS: Intervention significantly increased True responses!")
            elif result['effect_size'] > 0:
                print(f"\n⚠️  Intervention increased True responses by {result['effect_size']*100:.1f}%, but not significantly (p={result['p_value']:.3f})")
            else:
                print("\n❌ Intervention did not increase True responses")
            
            # Generate filename with timestamp and hash
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            filename = f"data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}/{question_id}.json"

            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

            # Prepare experiment data
            experiment_data = {
                "experiment_info": {
                    "timestamp": timestamp,
                    "question_id": question_id,
                    "question": question,
                    "true_answer": true_answer
                },
                "intervention_config": {
                    "intervention_text": intervention_text,
                    "position_pct": position_pct,
                    "n_rollouts": n_rollouts
                },
                "model_config": {
                    "model_name": generator.model_name,
                    "max_tokens": generator.max_tokens,
                    "temperature": generator.temperature
                },
                "control": {
                    "rollouts": control_rollouts,
                    "decisions": control_decisions,
                    "statistics": control_stats
                },
                "intervention": {
                    "rollouts": intervention_rollouts,
                    "decisions": intervention_decisions,
                    "statistics": intervention_stats
                },
                "analysis": result
            }

            # Convert numpy types to native Python types for JSON serialization
            experiment_data = convert_to_native_types(experiment_data)

            # Save to file
            with open(filename, 'w') as f:
                json.dump(experiment_data, f, indent=2)

            print(f"✓ Saved results to {filename}")
# %%
generate_experiment_data()
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_experiment_data(positions_pct=[0.25, 0.5, 0.75]):
    """Load all experiment data into a pandas DataFrame.

    Args:
        positions_pct: List of position percentages to load data for

    Returns:
        DataFrame with columns: question_id, position_pct, control_prop, intervention_prop,
                               effect, n_control, n_intervention, control_rollouts, intervention_rollouts
    """
    data = []

    for position_pct in positions_pct:
        folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}'

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    experiment_data = json.load(f)

                # Extract question ID from filename
                question_id = filename.replace('.json', '')

                # Extract stats
                n_total_control = experiment_data['analysis']['control_stats']['n_total']
                n_true_control = experiment_data['analysis']['control_stats']['n_true']
                n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
                n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']

                # Calculate proportions
                control_prop = n_true_control / n_total_control if n_total_control > 0 else np.nan
                intervention_prop = n_true_intervention / n_total_intervention if n_total_intervention > 0 else np.nan
                effect = intervention_prop - control_prop if not np.isnan(control_prop) and not np.isnan(intervention_prop) else np.nan

                data.append({
                    'question_id': question_id,
                    'position_pct': position_pct,
                    'control_prop': control_prop,
                    'intervention_prop': intervention_prop,
                    'effect': effect,
                    'n_control': n_total_control,
                    'n_intervention': n_total_intervention,
                    'control_rollouts': experiment_data['control']['rollouts'],
                    'intervention_rollouts': experiment_data['intervention']['rollouts']
                })

    return pd.DataFrame(data)

# Load all data once
df = load_experiment_data(positions_pct)
print(f"Loaded {len(df)} experiments across {df['position_pct'].nunique()} positions")

# %%
def analyze_position_results(df, position_pct):
    """Analyze and plot results for a specific intervention position."""
    data = df[df['position_pct'] == position_pct].dropna(subset=['control_prop', 'intervention_prop'])

    # Create histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    plt.hist(data['control_prop'], bins=bins, alpha=0.7, label='Control', color='blue', edgecolor='black')
    plt.hist(data['intervention_prop'], bins=bins, alpha=0.7, label='Intervention', color='red', edgecolor='black')

    plt.xlabel('Proportion True')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Proportion True: Control vs Intervention ({int(position_pct*100)}% position)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics for {int(position_pct*100)}% position:")
    print(f"Control - Mean: {data['control_prop'].mean():.3f}, Std: {data['control_prop'].std():.3f}")
    print(f"Intervention - Mean: {data['intervention_prop'].mean():.3f}, Std: {data['intervention_prop'].std():.3f}")
    print(f"Difference in means: {data['effect'].mean():.3f}")

# Run analysis for all positions
for position_pct in positions_pct:
    analyze_position_results(df, position_pct)

# %%
def plot_intervention_effect(df):
    """Plot the difference (intervention - control) for each position."""
    data = df.dropna(subset=['effect'])

    # Prepare data for box plot
    position_data = [data[data['position_pct'] == p]['effect'].values for p in [0.25, 0.5, 0.75]]

    plt.figure(figsize=(10, 6))
    plt.boxplot(position_data, labels=['25%', '50%', '75%'])
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No effect')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Intervention Effect (Intervention - Control)')
    plt.title('Distribution of Intervention Effect by Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print summary statistics for differences
    print("\nIntervention Effect Summary (Intervention - Control):")
    for position_pct in [0.25, 0.5, 0.75]:
        effects = data[data['position_pct'] == position_pct]['effect']
        print(f"{int(position_pct*100)}% position - Mean effect: {effects.mean():.3f}, Std: {effects.std():.3f}")

plot_intervention_effect(df)

# %%
def plot_intervention_effect_scatter(df, positions=[0.25, 0.5]):
    """Create scatter plot of intervention effect between two positions.

    Args:
        df: DataFrame with experiment data
        positions: List of two position percentages (e.g., [0.25, 0.5])
    """
    if len(positions) != 2:
        raise ValueError("Exactly two positions must be provided")

    # Pivot to get effects for both positions with question_id as index
    pivot = df[df['position_pct'].isin(positions)].pivot(
        index='question_id',
        columns='position_pct',
        values='effect'
    ).dropna()

    effects_pos1 = pivot[positions[0]].values
    effects_pos2 = pivot[positions[1]].values

    plt.figure(figsize=(8, 8))
    plt.scatter(effects_pos1, effects_pos2, alpha=0.6)

    # Add diagonal line for reference (y=x)
    min_val = min(effects_pos1.min(), effects_pos2.min())
    max_val = max(effects_pos1.max(), effects_pos2.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')

    # Add zero lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.xlabel(f'Intervention Effect at {int(positions[0]*100)}% Position')
    plt.ylabel(f'Intervention Effect at {int(positions[1]*100)}% Position')
    plt.title(f'Intervention Effect: {int(positions[0]*100)}% vs {int(positions[1]*100)}% Position\n(Each point is a question)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Calculate and display correlation
    correlation = np.corrcoef(effects_pos1, effects_pos2)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print(f"\nScatter plot shows {len(pivot)} questions with data at both {int(positions[0]*100)}% and {int(positions[1]*100)}% positions")
    print(f"Correlation between effects: {correlation:.3f}")

plot_intervention_effect_scatter(df)

# %%
plot_intervention_effect_scatter(df, positions=[0.5, 0.75])

# %%
def plot_intervention_effect_vs_control_percentage(df, position=0.25):
    """Plot intervention effect vs control percentage for a specific position.

    Shows how the intervention effect varies with the baseline control percentage.
    """
    data = df[(df['position_pct'] == position) & (df['n_control'] > 0) & (df['n_intervention'] > 0)].copy()
    data['control_percentage'] = data['control_prop'] * 100

    plt.figure(figsize=(10, 6))
    plt.scatter(data['control_percentage'], data['effect'], alpha=0.6)

    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No effect')

    plt.xlabel('Control Group Percentage (% answering True)')
    plt.ylabel('Intervention Effect (Intervention % - Control %)')
    plt.title(f'Intervention Effect vs Control Percentage at {int(position*100)}% Position\n(Each point is a question)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Calculate and display correlation
    if len(data) > 1:
        correlation = np.corrcoef(data['control_percentage'], data['effect'])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print(f"\nAnalyzed {len(data)} questions at {int(position*100)}% position")
    if len(data) > 1:
        print(f"Correlation between control percentage and intervention effect: {correlation:.3f}")
    print(f"Control percentage range: {data['control_percentage'].min():.1f}% - {data['control_percentage'].max():.1f}%")
    print(f"Intervention effect range: {data['effect'].min():.3f} - {data['effect'].max():.3f}")

plot_intervention_effect_vs_control_percentage(df)

# %%
def plot_rollout_lengths(df):
    """Plot histogram of rollout lengths: intervention vs control."""
    for position in [0.25, 0.5, 0.75]:
        position_data = df[df['position_pct'] == position]

        # Extract all rollout lengths
        control_lengths = [len(rollout) for rollouts in position_data['control_rollouts'] for rollout in rollouts]
        intervention_lengths = [len(rollout) for rollouts in position_data['intervention_rollouts'] for rollout in rollouts]

        if control_lengths and intervention_lengths:
            print(f"\nRollout Length Statistics for {int(position*100)}% position:")
            print(f"Control - Mean: {np.mean(control_lengths):.1f}, Std: {np.std(control_lengths):.1f}, Median: {np.median(control_lengths):.1f}")
            print(f"Intervention - Mean: {np.mean(intervention_lengths):.1f}, Std: {np.std(intervention_lengths):.1f}, Median: {np.median(intervention_lengths):.1f}")
            print(f"Difference in means: {np.mean(intervention_lengths) - np.mean(control_lengths):.1f}")
            print(f"Control range: {min(control_lengths)} - {max(control_lengths)}")
            print(f"Intervention range: {min(intervention_lengths)} - {max(intervention_lengths)}")

plot_rollout_lengths(df)

# %%
def plot_control_vs_intervention(df):
    """Plot control proportion vs intervention proportion for each position.

    Tests whether control and intervention proportions are correlated.
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, position in enumerate([0.25, 0.5, 0.75]):
        data = df[df['position_pct'] == position].dropna(subset=['control_prop', 'intervention_prop'])

        ax = axes[idx]
        ax.scatter(data['control_prop'], data['intervention_prop'], alpha=0.6)

        # Add diagonal line for reference (y=x)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='y=x (no effect)')

        # Add grid lines at 0.5
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

        ax.set_xlabel('Control Proportion True')
        ax.set_ylabel('Intervention Proportion True')
        ax.set_title(f'{int(position*100)}% Position')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        # Calculate and display correlation (on centered data)
        if len(data) > 1:
            # Center the data (subtract means)
            control_centered = data['control_prop'] - data['control_prop'].mean()
            intervention_centered = data['intervention_prop'] - data['intervention_prop'].mean()
            correlation = np.corrcoef(control_centered, intervention_centered)[0, 1]
            ax.text(0.05, 0.95, f'Correlation (centered): {correlation:.3f}\nn={len(data)}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top')

    plt.suptitle('Control vs Intervention Proportion True by Position', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nControl vs Intervention Correlation Analysis (centered data):")
    for position in [0.25, 0.5, 0.75]:
        data = df[df['position_pct'] == position].dropna(subset=['control_prop', 'intervention_prop'])
        if len(data) > 1:
            # Center the data
            control_centered = data['control_prop'] - data['control_prop'].mean()
            intervention_centered = data['intervention_prop'] - data['intervention_prop'].mean()
            correlation = np.corrcoef(control_centered, intervention_centered)[0, 1]
            print(f"{int(position*100)}% position - Correlation: {correlation:.3f} (n={len(data)})")

plot_control_vs_intervention(df)

# %%
def plot_intervention_effect_correlation_by_control_quintile(df, positions=[0.25, 0.5]):
    """Plot correlation of intervention effects between two positions, stratified by control group quintiles.

    Args:
        df: DataFrame with experiment data
        positions: List of two position percentages (e.g., [0.25, 0.5])
    """
    if len(positions) != 2:
        raise ValueError("Exactly two positions must be provided")

    # Get data for both positions with control proportion
    data_25 = df[df['position_pct'] == positions[0]][['question_id', 'control_prop', 'effect']].rename(columns={'effect': 'effect_25'})
    data_50 = df[df['position_pct'] == positions[1]][['question_id', 'control_prop', 'effect']].rename(columns={'effect': 'effect_50'})

    # Merge on question_id, using control_prop from first position
    merged = data_25.merge(data_50[['question_id', 'effect_50']], on='question_id').dropna()

    # Create quintiles based on control proportion
    merged['control_quintile'] = pd.qcut(merged['control_prop'], q=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

    # Create 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    quintile_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    for idx, quintile in enumerate(quintile_labels):
        quintile_data = merged[merged['control_quintile'] == quintile]

        ax = axes[idx]

        if len(quintile_data) > 0:
            ax.scatter(quintile_data['effect_25'], quintile_data['effect_50'], alpha=0.6)

            # Add diagonal line for reference (y=x)
            effects_25 = quintile_data['effect_25'].values
            effects_50 = quintile_data['effect_50'].values
            min_val = min(effects_25.min(), effects_50.min())
            max_val = max(effects_25.max(), effects_50.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')

            # Add zero lines
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)

            ax.set_xlabel(f'{int(positions[0]*100)}% Effect')
            ax.set_ylabel(f'{int(positions[1]*100)}% Effect')
            ax.set_title(f'Control {quintile}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
            ax.set_aspect('equal', adjustable='box')

            # Calculate and display correlation
            if len(quintile_data) > 1:
                correlation = np.corrcoef(quintile_data['effect_25'], quintile_data['effect_50'])[0, 1]
                ax.text(0.95, 0.05, f'r={correlation:.3f}\nn={len(quintile_data)}',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='bottom', horizontalalignment='right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_xlabel(f'{int(positions[0]*100)}% Effect')
            ax.set_ylabel(f'{int(positions[1]*100)}% Effect')
            ax.set_title(f'Control {quintile}')

    plt.suptitle(f'Intervention Effect Correlation ({int(positions[0]*100)}% vs {int(positions[1]*100)}%) by Control Group Quintile',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nIntervention Effect Correlation ({int(positions[0]*100)}% vs {int(positions[1]*100)}%) by Control Quintile:")
    for quintile in quintile_labels:
        quintile_data = merged[merged['control_quintile'] == quintile]
        if len(quintile_data) > 1:
            correlation = np.corrcoef(quintile_data['effect_25'], quintile_data['effect_50'])[0, 1]
            mean_control = quintile_data['control_prop'].mean()
            print(f"{quintile:>10} - r={correlation:.3f}, n={len(quintile_data):2d}, mean_control={mean_control:.3f}")
        else:
            print(f"{quintile:>10} - Insufficient data (n={len(quintile_data)})")

plot_intervention_effect_correlation_by_control_quintile(df, positions=[0.25, 0.5])

# %%