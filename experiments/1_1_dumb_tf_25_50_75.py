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
#generate_experiment_data()
# %%
import matplotlib.pyplot as plt
import numpy as np

def analyze_position_results(position_pct):
    """Analyze and plot results for a specific intervention position."""
    # Collect proportions for histogram
    control_proportions = []
    intervention_proportions = []

    # Loop through all JSON files for this position
    folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}'
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                experiment_data = json.load(f)
            
            # Extract stats from this file
            n_total_control = experiment_data['analysis']['control_stats']['n_total']
            n_true_control = experiment_data['analysis']['control_stats']['n_true']
            n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
            n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']
            
            # Calculate proportions
            control_proportion = n_true_control / n_total_control if n_total_control > 0 else 0
            intervention_proportion = n_true_intervention / n_total_intervention if n_total_intervention > 0 else 0
            
            control_proportions.append(control_proportion)
            intervention_proportions.append(intervention_proportion)

    # Create histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    plt.hist(control_proportions, bins=bins, alpha=0.7, label='Control', color='blue', edgecolor='black')
    plt.hist(intervention_proportions, bins=bins, alpha=0.7, label='Intervention', color='red', edgecolor='black')

    plt.xlabel('Proportion True')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Proportion True: Control vs Intervention ({int(position_pct*100)}% position)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics for {int(position_pct*100)}% position:")
    print(f"Control - Mean: {np.mean(control_proportions):.3f}, Std: {np.std(control_proportions):.3f}")
    print(f"Intervention - Mean: {np.mean(intervention_proportions):.3f}, Std: {np.std(intervention_proportions):.3f}")
    print(f"Difference in means: {np.mean(intervention_proportions) - np.mean(control_proportions):.3f}")

# Run analysis for all positions
for position_pct in positions_pct:
    analyze_position_results(position_pct)

# %%
# Plot the difference (intervention - control) for each position
def plot_intervention_effect():
    positions_pct = [0.25, 0.5, 0.75]
    all_differences = []
    position_labels = []
    
    for position_pct in positions_pct:
        control_proportions = []
        intervention_proportions = []
        
        # Loop through all JSON files for this position
        folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    experiment_data = json.load(f)
                
                # Extract stats from this file
                n_total_control = experiment_data['analysis']['control_stats']['n_total']
                n_true_control = experiment_data['analysis']['control_stats']['n_true']
                n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
                n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']
                
                # Calculate proportions
                control_proportion = n_true_control / n_total_control if n_total_control > 0 else 0
                intervention_proportion = n_true_intervention / n_total_intervention if n_total_intervention > 0 else 0
                
                control_proportions.append(control_proportion)
                intervention_proportions.append(intervention_proportion)
        
        # Calculate differences for this position
        differences = [interv - ctrl for interv, ctrl in zip(intervention_proportions, control_proportions)]
        all_differences.extend(differences)
        position_labels.extend([f'{int(position_pct*100)}%'] * len(differences))
    
    # Create box plot of differences by position
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    position_data = []
    positions = [0.25, 0.5, 0.75]
    
    for position_pct in positions:
        control_proportions = []
        intervention_proportions = []
        
        folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    experiment_data = json.load(f)
                
                n_total_control = experiment_data['analysis']['control_stats']['n_total']
                n_true_control = experiment_data['analysis']['control_stats']['n_true']
                n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
                n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']
                
                control_proportion = n_true_control / n_total_control if n_total_control > 0 else 0
                intervention_proportion = n_true_intervention / n_total_intervention if n_total_intervention > 0 else 0
                
                control_proportions.append(control_proportion)
                intervention_proportions.append(intervention_proportion)
        
        differences = [interv - ctrl for interv, ctrl in zip(intervention_proportions, control_proportions)]
        position_data.append(differences)
    
    plt.boxplot(position_data, labels=[f'{int(p*100)}%' for p in positions])
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
    for i, position_pct in enumerate(positions):
        differences = position_data[i]
        print(f"{int(position_pct*100)}% position - Mean effect: {np.mean(differences):.3f}, Std: {np.std(differences):.3f}")

plot_intervention_effect()

# %%
def plot_intervention_effect_scatter(positions=[0.25, 0.5]):
    """Create scatter plot of intervention effect between two positions
    
    Args:
        positions: List of two position percentages (e.g., [0.25, 0.5])
    """
    
    if len(positions) != 2:
        raise ValueError("Exactly two positions must be provided")
    
    # Collect data for the specified positions
    effects_by_position = {}
    
    for position_pct in positions:
        folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position_pct*100)}'
        effects = {}
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    experiment_data = json.load(f)
                
                # Extract question ID from filename (assuming format like 'question_123.json')
                question_id = filename.replace('.json', '')
                
                n_total_control = experiment_data['analysis']['control_stats']['n_total']
                n_true_control = experiment_data['analysis']['control_stats']['n_true']
                n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
                n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']
                
                control_proportion = n_true_control / n_total_control if n_total_control > 0 else 0
                intervention_proportion = n_true_intervention / n_total_intervention if n_total_intervention > 0 else 0
                
                effect = intervention_proportion - control_proportion
                effects[question_id] = effect
        
        effects_by_position[position_pct] = effects
    
    # Find common question IDs
    common_ids = set(effects_by_position[positions[0]].keys()) & set(effects_by_position[positions[1]].keys())
    
    # Prepare data for scatter plot
    effects_pos1 = [effects_by_position[positions[0]][qid] for qid in common_ids]
    effects_pos2 = [effects_by_position[positions[1]][qid] for qid in common_ids]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(effects_pos1, effects_pos2, alpha=0.6)
    
    # Add diagonal line for reference (y=x)
    min_val = min(min(effects_pos1), min(effects_pos2))
    max_val = max(max(effects_pos1), max(effects_pos2))
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
    
    print(f"\nScatter plot shows {len(common_ids)} questions with data at both {int(positions[0]*100)}% and {int(positions[1]*100)}% positions")
    print(f"Correlation between effects: {correlation:.3f}")

plot_intervention_effect_scatter()

# %%
plot_intervention_effect_scatter(positions=[0.5, 0.75])
# %%
def plot_intervention_effect_vs_control_percentage(position=0.25):
    """
    Plot intervention effect vs control percentage for a specific position.
    Shows how the intervention effect varies with the baseline control percentage.
    """
    # Collect data for the specified position
    question_metrics = []
    
    folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position*100)}'
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                experiment_data = json.load(f)
            
            # Extract question ID from filename
            question_id = filename.replace('.json', '')
            
            # Extract stats from this file
            n_total_control = experiment_data['analysis']['control_stats']['n_total']
            n_true_control = experiment_data['analysis']['control_stats']['n_true']
            n_total_intervention = experiment_data['analysis']['intervention_stats']['n_total']
            n_true_intervention = experiment_data['analysis']['intervention_stats']['n_true']
            
            # Skip if no data for either group
            if n_total_control == 0 or n_total_intervention == 0:
                continue
                
            control_proportion = n_true_control / n_total_control
            intervention_proportion = n_true_intervention / n_total_intervention
            
            effect = intervention_proportion - control_proportion
            
            question_metrics.append({
                'question_id': question_id,
                'control_percentage': control_proportion * 100,
                'intervention_effect': effect,
                'n_control': n_total_control,
                'n_intervention': n_total_intervention
            })
    
    # Convert to arrays for plotting
    control_percentages = [m['control_percentage'] for m in question_metrics]
    intervention_effects = [m['intervention_effect'] for m in question_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(control_percentages, intervention_effects, alpha=0.6)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No effect')
    
    plt.xlabel('Control Group Percentage (% answering True)')
    plt.ylabel('Intervention Effect (Intervention % - Control %)')
    plt.title(f'Intervention Effect vs Control Percentage at {int(position*100)}% Position\n(Each point is a question)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate and display correlation
    if len(control_percentages) > 1:
        correlation = np.corrcoef(control_percentages, intervention_effects)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                 transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAnalyzed {len(question_metrics)} questions at {int(position*100)}% position")
    if len(control_percentages) > 1:
        print(f"Correlation between control percentage and intervention effect: {correlation:.3f}")
    
    # Print some summary statistics
    if control_percentages:
        print(f"Control percentage range: {min(control_percentages):.1f}% - {max(control_percentages):.1f}%")
        print(f"Intervention effect range: {min(intervention_effects):.3f} - {max(intervention_effects):.3f}")

plot_intervention_effect_vs_control_percentage()

# %%
# Plot histogram of rollout lengths: intervention vs control
def plot_rollout_lengths():
    positions = [0.25, 0.5, 0.75]
    
    for position in positions:
        control_lengths = []
        intervention_lengths = []
        
        # Loop through all JSON files for this position
        folder_path = f'data/interventions/experiment_1_1_dumb_tf/{int(position*100)}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    experiment_data = json.load(f)
                
                # Extract rollout lengths from control group
                for rollout in experiment_data['control']['rollouts']:
                    control_lengths.append(len(rollout))
                
                # Extract rollout lengths from intervention group
                for rollout in experiment_data['intervention']['rollouts']:
                    intervention_lengths.append(len(rollout))
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        # Determine appropriate bins
        all_lengths = control_lengths + intervention_lengths
        if all_lengths:
            min_len = min(all_lengths)
            max_len = max(all_lengths)
            bins = range(min_len, max_len + 2)  # +2 to include max_len in the last bin
        else:
            bins = 20
        
        
        # Print summary statistics
        if control_lengths and intervention_lengths:
            print(f"\nRollout Length Statistics for {int(position*100)}% position:")
            print(f"Control - Mean: {np.mean(control_lengths):.1f}, Std: {np.std(control_lengths):.1f}, Median: {np.median(control_lengths):.1f}")
            print(f"Intervention - Mean: {np.mean(intervention_lengths):.1f}, Std: {np.std(intervention_lengths):.1f}, Median: {np.median(intervention_lengths):.1f}")
            print(f"Difference in means: {np.mean(intervention_lengths) - np.mean(control_lengths):.1f}")
            print(f"Control range: {min(control_lengths)} - {max(control_lengths)}")
            print(f"Intervention range: {min(intervention_lengths)} - {max(intervention_lengths)}")

plot_rollout_lengths()

# %%
