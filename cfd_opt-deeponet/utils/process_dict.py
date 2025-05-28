"""
State Dictionary Processing Utility for DeepONet-CFD Project

This module provides a function to process and clean state dictionaries
by removing specific prefixes from the keys.
"""
def process_state_dict(state_dict):
    """
    Process the state dictionary by removing the '_orig_mod.' prefix from keys.

    Args:
        state_dict (dict): The original state dictionary.

    Returns:
        dict: A new state dictionary with updated keys.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Replace '_orig_mod.' in key names
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value
    return new_state_dict