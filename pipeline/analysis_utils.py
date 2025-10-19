#!/usr/bin/env python3
"""
Analysis utilities for intervention experiments.

Provides functions for computing statistics and testing significance
of intervention effects on decision outcomes.
"""

from typing import List, Optional, Dict
import numpy as np
from scipy import stats


def compute_statistics(decisions: List[Optional[bool]]) -> Dict:
    """
    Compute statistics for a list of decisions.

    Args:
        decisions: List of boolean decisions (True/False/None)

    Returns:
        Dictionary with statistics:
            - percent_true: Percentage of True decisions (excluding None)
            - percent_false: Percentage of False decisions (excluding None)
            - percent_null: Percentage of None (unparseable) decisions
            - n_total: Total number of decisions
            - n_valid: Number of non-None decisions
            - n_true: Count of True decisions
            - n_false: Count of False decisions
            - n_null: Count of None decisions

    Example:
        >>> decisions = [True, True, False, None, True]
        >>> stats = compute_statistics(decisions)
        >>> stats['percent_true']
        0.75
        >>> stats['n_valid']
        4
    """
    n_total = len(decisions)

    if n_total == 0:
        return {
            'percent_true': 0.0,
            'percent_false': 0.0,
            'percent_null': 0.0,
            'n_total': 0,
            'n_valid': 0,
            'n_true': 0,
            'n_false': 0,
            'n_null': 0
        }

    # Count each type
    n_true = sum(1 for d in decisions if d is True)
    n_false = sum(1 for d in decisions if d is False)
    n_null = sum(1 for d in decisions if d is None)
    n_valid = n_true + n_false

    # Compute percentages (of valid decisions for true/false, of total for null)
    if n_valid > 0:
        percent_true = n_true / n_valid
        percent_false = n_false / n_valid
    else:
        percent_true = 0.0
        percent_false = 0.0

    percent_null = n_null / n_total if n_total > 0 else 0.0

    return {
        'percent_true': percent_true,
        'percent_false': percent_false,
        'percent_null': percent_null,
        'n_total': n_total,
        'n_valid': n_valid,
        'n_true': n_true,
        'n_false': n_false,
        'n_null': n_null
    }


def test_significance(
    control_decisions: List[Optional[bool]],
    intervention_decisions: List[Optional[bool]],
    test_type: str = "proportion"
) -> Dict:
    """
    Test whether intervention significantly changed decision outcomes.

    Args:
        control_decisions: Decisions from control/baseline rollouts
        intervention_decisions: Decisions from intervention rollouts
        test_type: Type of statistical test to perform:
            - "proportion": Two-proportion z-test (default)
            - "chi2": Chi-squared test of independence
            - "fisher": Fisher's exact test (for small samples)

    Returns:
        Dictionary with test results:
            - test_type: Name of test performed
            - p_value: P-value from test
            - significant: Boolean (p < 0.05)
            - control_stats: Statistics for control group
            - intervention_stats: Statistics for intervention group
            - effect_size: Difference in proportion of True decisions
            - interpretation: Human-readable interpretation

    Example:
        >>> control = [True, False, True, False, True]
        >>> intervention = [True, True, True, True, False]
        >>> result = test_significance(control, intervention)
        >>> result['effect_size']
        0.2
    """
    # Filter out None values
    control_valid = [d for d in control_decisions if d is not None]
    intervention_valid = [d for d in intervention_decisions if d is not None]

    # Compute statistics for both groups
    control_stats = compute_statistics(control_decisions)
    intervention_stats = compute_statistics(intervention_decisions)

    # Effect size: difference in proportion of True
    effect_size = intervention_stats['percent_true'] - control_stats['percent_true']

    # Check if we have enough data
    if len(control_valid) == 0 or len(intervention_valid) == 0:
        return {
            'test_type': test_type,
            'p_value': None,
            'significant': False,
            'control_stats': control_stats,
            'intervention_stats': intervention_stats,
            'effect_size': effect_size,
            'interpretation': "Insufficient data for statistical test"
        }

    # Perform statistical test
    if test_type == "proportion":
        p_value = _two_proportion_z_test(control_valid, intervention_valid)
    elif test_type == "chi2":
        p_value = _chi2_test(control_valid, intervention_valid)
    elif test_type == "fisher":
        p_value = _fisher_exact_test(control_valid, intervention_valid)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Determine significance
    significant = p_value < 0.05 if p_value is not None else False

    # Generate interpretation
    if not significant:
        interpretation = "No significant difference between control and intervention"
    else:
        direction = "increased" if effect_size > 0 else "decreased"
        interpretation = f"Intervention significantly {direction} True decisions (p={p_value:.4f})"

    return {
        'test_type': test_type,
        'p_value': p_value,
        'significant': significant,
        'control_stats': control_stats,
        'intervention_stats': intervention_stats,
        'effect_size': effect_size,
        'interpretation': interpretation
    }


def _two_proportion_z_test(control: List[bool], intervention: List[bool]) -> Optional[float]:
    """
    Perform two-proportion z-test.

    Args:
        control: List of boolean decisions (no None values)
        intervention: List of boolean decisions (no None values)

    Returns:
        P-value from two-sided z-test
    """
    n1 = len(control)
    n2 = len(intervention)
    x1 = sum(control)  # Count of True in control
    x2 = sum(intervention)  # Count of True in intervention

    # Pooled proportion
    p_pool = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return 1.0  # No variance, no difference

    # Z-statistic
    p1 = x1 / n1
    p2 = x2 / n2
    z = (p2 - p1) / se

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_value

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    else:
        return obj

def _chi2_test(control: List[bool], intervention: List[bool]) -> Optional[float]:
    """
    Perform chi-squared test of independence.

    Args:
        control: List of boolean decisions
        intervention: List of boolean decisions

    Returns:
        P-value from chi-squared test
    """
    # Create contingency table
    control_true = sum(control)
    control_false = len(control) - control_true
    intervention_true = sum(intervention)
    intervention_false = len(intervention) - intervention_true

    contingency_table = [
        [control_true, control_false],
        [intervention_true, intervention_false]
    ]

    _, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return p_value


def _fisher_exact_test(control: List[bool], intervention: List[bool]) -> Optional[float]:
    """
    Perform Fisher's exact test.

    Args:
        control: List of boolean decisions
        intervention: List of boolean decisions

    Returns:
        P-value from Fisher's exact test
    """
    # Create contingency table
    control_true = sum(control)
    control_false = len(control) - control_true
    intervention_true = sum(intervention)
    intervention_false = len(intervention) - intervention_true

    contingency_table = [
        [control_true, control_false],
        [intervention_true, intervention_false]
    ]

    _, p_value = stats.fisher_exact(contingency_table)
    return p_value


def print_statistics_comparison(
    control_decisions: List[Optional[bool]],
    intervention_decisions: List[Optional[bool]],
    test_type: str = "proportion"
):
    """
    Print a formatted comparison of control vs intervention statistics.

    Args:
        control_decisions: Decisions from control rollouts
        intervention_decisions: Decisions from intervention rollouts
        test_type: Type of statistical test to perform

    Example:
        >>> control = [True, False, True, False, True]
        >>> intervention = [True, True, True, True, False]
        >>> print_statistics_comparison(control, intervention)
        # Prints formatted comparison table
    """
    result = test_significance(control_decisions, intervention_decisions, test_type)

    print("="*60)
    print("INTERVENTION ANALYSIS")
    print("="*60)
    print("\nControl Group:")
    print(f"  Total: {result['control_stats']['n_total']}")
    print(f"  Valid: {result['control_stats']['n_valid']}")
    print(f"  True:  {result['control_stats']['n_true']} ({result['control_stats']['percent_true']*100:.1f}%)")
    print(f"  False: {result['control_stats']['n_false']} ({result['control_stats']['percent_false']*100:.1f}%)")
    print(f"  Null:  {result['control_stats']['n_null']} ({result['control_stats']['percent_null']*100:.1f}%)")

    print("\nIntervention Group:")
    print(f"  Total: {result['intervention_stats']['n_total']}")
    print(f"  Valid: {result['intervention_stats']['n_valid']}")
    print(f"  True:  {result['intervention_stats']['n_true']} ({result['intervention_stats']['percent_true']*100:.1f}%)")
    print(f"  False: {result['intervention_stats']['n_false']} ({result['intervention_stats']['percent_false']*100:.1f}%)")
    print(f"  Null:  {result['intervention_stats']['n_null']} ({result['intervention_stats']['percent_null']*100:.1f}%)")

    print("\nStatistical Test:")
    print(f"  Test: {result['test_type']}")
    print(f"  P-value: {result['p_value']:.4f}" if result['p_value'] is not None else "  P-value: N/A")
    print(f"  Significant: {'Yes (p < 0.05)' if result['significant'] else 'No'}")
    print(f"  Effect size: {result['effect_size']:+.3f} ({result['effect_size']*100:+.1f}%)")

    print(f"\n{result['interpretation']}")
    print("="*60)
