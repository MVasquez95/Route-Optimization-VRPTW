import numpy as np
import pandas as pd

def generate_scenarios(distance_df, n_scenarios=50, sigma=0.15, edge_level=False, seed=42):
    """
    Generate multiple perturbed matrices to simulate scenario variability.

    Args:
        base_matrix (pd.DataFrame): Original distance or time matrix.
        perturbation_levels (list[float]): Levels of perturbation, e.g., [0.9, 1.0, 1.1].
        mode (str): 'multiplicative' or 'additive' perturbation type.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of perturbed matrices.
    """
    rng = np.random.default_rng(seed)
    base = distance_df.values
    scenarios = []
    for _ in range(n_scenarios):
        if edge_level:
            noise = rng.lognormal(mean=0.0, sigma=sigma, size=base.shape)
        else:
            m = rng.lognormal(mean=0.0, sigma=sigma)
            noise = np.full(base.shape, m)
        scenario_time = base * noise
        np.fill_diagonal(scenario_time, 0.0)
        scenarios.append(pd.DataFrame(
            scenario_time,
            index=distance_df.index,
            columns=distance_df.columns
        ))
    return scenarios


def simulate_accident(time_matrix, node_i, node_j, factor=5.0):
    """
    Simulate local disruptions (e.g., accidents or blocked roads).

    Args:
        time_matrix (pd.DataFrame): Base time matrix.
        affected_nodes (list[str]): Subset of nodes affected.
        delay_factor (float): Multiplier for delay impact.
    """
    tm = time_matrix.copy()
    if node_i in tm.index and node_j in tm.columns:
        tm.loc[node_i, node_j] *= factor
        tm.loc[node_j, node_i] *= factor
    return tm


def percentile_matrix(scenarios, q=90):
    """
    Compute element-wise percentile across scenario matrices.

    Args:
        matrices (dict[str, pd.DataFrame]): Scenario matrices.
        percentile (int): Percentile to compute.

    Returns:
        pd.DataFrame: Resulting matrix.
    """
    arr = np.stack([df.values for df in scenarios], axis=2)
    mat = np.percentile(arr, q, axis=2)
    return pd.DataFrame(mat, index=scenarios[0].index, columns=scenarios[0].columns)
