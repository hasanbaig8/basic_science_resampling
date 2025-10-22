#!/usr/bin/env python3
"""
InterventionGrader - Grade interventions using LLM on 1-10 scale.

Uses vLLM to ask the model to grade how well an intervention steers
toward the user's intended goal.
"""

import re
import requests
from typing import Optional, List


class InterventionGrader:
    """
    Grade interventions using LLM with structured 1-10 scoring.

    The grader prompts the LLM to evaluate whether an intervention
    successfully steers toward the user's goal, with clear rubric:
    - 1-3: Irrelevant or contradicts the goal
    - 4-5: Somewhat related but doesn't steer effectively
    - 6-7: Decent steering, moves toward the goal
    - 8-9: Good steering, clearly moves toward the goal
    - 10: Excellent steering, directly and effectively moves toward the goal
    """

    def __init__(self, vllm_url: str = "http://localhost:8000/v1/completions", model_name: str = "Qwen/Qwen3-8b"):
        """
        Initialize the intervention grader.

        Args:
            vllm_url: URL of vLLM completions endpoint
            model_name: Model name for vLLM
        """
        self.vllm_url = vllm_url
        self.model_name = model_name

        # Pattern to match {"grade": N} format
        self.grade_pattern = r'\{\s*"grade"\s*:\s*(\d+)\s*\}'

    def grade_intervention(self, original_prompt: str, goal: str, intervention: str) -> Optional[int]:
        """
        Grade an intervention on a 1-10 scale.

        Args:
            original_prompt: The original user prompt/question
            goal: The desired steering goal
            intervention: The candidate intervention text to grade

        Returns:
            Grade 1-10, or None if parsing fails
        """
        # Construct grading prompt
        grading_prompt = f"""You are evaluating whether a text intervention successfully steers a response toward a specific goal.

Original prompt: {original_prompt}

Desired steering goal: {goal}

Candidate intervention text:
{intervention}

Grade this intervention on a 1-10 scale based on how well it steers toward the goal:
- 1-3: Irrelevant or contradicts the goal
- 4-5: Somewhat related but doesn't steer effectively
- 6-7: Decent steering, moves toward the goal
- 8-9: Good steering, clearly moves toward the goal
- 10: Excellent steering, directly and effectively moves toward the goal

Respond ONLY with JSON in this exact format: {{"grade": N}} where N is 1-10."""

        # Call vLLM
        try:
            response = self._call_vllm(grading_prompt)
            grade = self._parse_grade(response)
            return grade
        except Exception as e:
            print(f"[GRADER ERROR] Failed to grade intervention: {e}")
            return None

    def _call_vllm(self, prompt: str) -> str:
        """
        Call vLLM completions endpoint.

        Args:
            prompt: The grading prompt

        Returns:
            Model's response text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 50,  # Just need {"grade": N}
            "temperature": 0.3,  # Low temperature for more consistent grading
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.vllm_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        return result['choices'][0]['text']

    def _parse_grade(self, response_text: str) -> Optional[int]:
        """
        Parse grade from response text.

        Args:
            response_text: Model's response containing {"grade": N}

        Returns:
            Grade integer 1-10, or None if parsing fails
        """
        match = re.search(self.grade_pattern, response_text, re.IGNORECASE)
        if match:
            grade = int(match.group(1))
            # Validate grade is in range
            if 1 <= grade <= 10:
                return grade

        # Try to be more lenient - look for any number after "grade"
        lenient_pattern = r'grade["\s:]+(\d+)'
        match = re.search(lenient_pattern, response_text, re.IGNORECASE)
        if match:
            grade = int(match.group(1))
            if 1 <= grade <= 10:
                return grade

        print(f"[GRADER WARNING] Failed to parse grade from: {response_text}")
        return None

    def batch_grade_interventions(
        self,
        original_prompt: str,
        goal: str,
        interventions: List[str]
    ) -> List[Optional[int]]:
        """
        Grade multiple interventions in a single batch API call.

        This is much more efficient than calling grade_intervention() individually
        for each intervention candidate.

        Args:
            original_prompt: The original user prompt/question
            goal: The desired steering goal
            interventions: List of candidate intervention texts to grade

        Returns:
            List of grades (1-10 or None) corresponding to each intervention
        """
        if not interventions:
            return []

        # Construct batch grading prompt
        batch_prompt = f"""You are evaluating multiple text interventions to determine which ones successfully steer a response toward a specific goal.

Original prompt: {original_prompt}

Desired steering goal: {goal}

Please grade each of the following intervention candidates on a 1-10 scale:
- 1-3: Irrelevant or contradicts the goal
- 4-5: Somewhat related but doesn't steer effectively
- 6-7: Decent steering, moves toward the goal
- 8-9: Good steering, clearly moves toward the goal
- 10: Excellent steering, directly and effectively moves toward the goal

"""

        # Add each intervention with an index
        for idx, intervention in enumerate(interventions, 1):
            batch_prompt += f"\n--- Intervention {idx} ---\n{intervention}\n"

        batch_prompt += """\nRespond with grades in JSON format as a list of integers. Example format:
[7, 5, 9, 3, 8]

Respond ONLY with the JSON list, nothing else."""

        # Call vLLM with larger max_tokens for batch response
        try:
            payload = {
                "model": self.model_name,
                "prompt": batch_prompt,
                "max_tokens": len(interventions) * 10,  # ~10 tokens per grade
                "temperature": 0.3,
                "stream": False
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            response_text = result['choices'][0]['text']

            # Parse the batch response
            grades = self._parse_batch_grades(response_text, len(interventions))
            return grades

        except Exception as e:
            print(f"[GRADER ERROR] Batch grading failed: {e}")
            print(f"[GRADER] Falling back to individual grading...")
            # Fallback to individual grading if batch fails
            return [self.grade_intervention(original_prompt, goal, interv) for interv in interventions]

    def _parse_batch_grades(self, response_text: str, expected_count: int) -> List[Optional[int]]:
        """
        Parse batch grades from response text.

        Args:
            response_text: Model's response containing JSON list of grades
            expected_count: Expected number of grades

        Returns:
            List of grade integers (1-10 or None)
        """
        # Try to extract JSON list
        import json

        # Look for JSON list pattern
        list_pattern = r'\[[\d\s,]+\]'
        match = re.search(list_pattern, response_text)

        if match:
            try:
                grades_raw = json.loads(match.group(0))
                # Validate and convert
                grades = []
                for g in grades_raw:
                    if isinstance(g, int) and 1 <= g <= 10:
                        grades.append(g)
                    else:
                        grades.append(None)

                # Ensure we have the right number of grades
                if len(grades) == expected_count:
                    return grades
                else:
                    print(f"[GRADER WARNING] Expected {expected_count} grades, got {len(grades)}")
                    # Pad or truncate
                    if len(grades) < expected_count:
                        grades.extend([None] * (expected_count - len(grades)))
                    else:
                        grades = grades[:expected_count]
                    return grades

            except json.JSONDecodeError as e:
                print(f"[GRADER WARNING] Failed to parse JSON list: {e}")

        # Fallback: try to extract individual numbers
        numbers = re.findall(r'\b(\d+)\b', response_text)
        grades = []
        for num_str in numbers[:expected_count]:
            num = int(num_str)
            if 1 <= num <= 10:
                grades.append(num)
            else:
                grades.append(None)

        # Pad if needed
        while len(grades) < expected_count:
            grades.append(None)

        return grades[:expected_count]
