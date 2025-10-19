import json
import numpy as np
# Load the parsed rollouts data

MIN_PERCENT_TRUE = 0.1
MAX_PERCENT_TRUE = 0.9

with open('/workspace/basic_science_resampling/data/strategyqa_rollouts_parsed.json', 'r') as f:
    rollouts_data = json.load(f)


steerable_question_ids = []
for i in range(len(rollouts_data)):
    decisions = np.array([x for x in rollouts_data[i]['decisions'] if x is not None])
    percent_true = np.sum(decisions) / len(decisions)
    if percent_true > MIN_PERCENT_TRUE and percent_true < MAX_PERCENT_TRUE:
        steerable_question_ids.append(rollouts_data[i]['question_id'])

# Save the steerable question IDs to a file
with open('/workspace/basic_science_resampling/data/steerable_question_ids.json', 'w') as f:
    json.dump({"question_ids": steerable_question_ids, "min_percent_true": MIN_PERCENT_TRUE, "max_percent_true": MAX_PERCENT_TRUE}, f, indent=2)
