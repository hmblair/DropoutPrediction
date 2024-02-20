# utils.py

import re

def find_hairpins(structure : str, ignore_pseudoknots : bool = False) -> list[str]:
    """
    Find all hairpin loops in a structure.

    Parameters:
    ----------
    structure (str): 
        The structure to search for hairpin loops.
    ignore_pseudoknots (bool):
        Whether to ignore pseudoknots when searching for hairpin loops.

    Returns:
    -------
    list[str]: 
        The hairpin loops in the structure.
    """
    if ignore_pseudoknots:
        # Remove pseudoknots from the structure
        structure = re.sub(r'\{.*?\}', '', structure)
        structure = re.sub(r'\[.*?\]', '', structure)
    
    # Create a regular expression pattern for hairpin loops
    pattern = r'\(+\.+\)+'

    # Find all matches of the pattern in the structure
    matches = re.findall(pattern, structure)

    # Return the matches
    return matches


def longest_hairpin(structure : str, ignore_pseudoknots : bool = False) -> int:
    """
    Find the longest hairpin loop in a structure.

    Parameters:
    ----------
    structure (str): 
        The structure to search for hairpin loops.
    ignore_pseudoknots (bool):
        Whether to ignore pseudoknots when searching for hairpin loops.

    Returns:
    -------
    int: 
        The length of the longest hairpin loop in the structure.
    """
    # Find all hairpin loops in the structure
    hairpins = find_hairpins(structure, ignore_pseudoknots)

    # Find the longest hairpin loop
    if len(hairpins) == 0:
        return 0
    
    longest_hairpin = max(hairpins, key=len)

    # Return the length of the longest hairpin loop
    return len(longest_hairpin)



def longest_run(sequence : str) -> int:
    """
    Returns the length of the longest run in a string. A run is a sequence of
    consecutive identical characters.

    Parameters:
    ----------
    sequence (str): 
        The string to search for runs.

    Returns:
    -------
    int: 
        The length of the longest run in the string.
    """
    # Initialize the length of the longest run
    longest_run = 0

    # Initialize the length of the current run
    current_run = 0

    # Iterate over the characters in the string
    prev_char = sequence[0]
    for char in sequence[1:]:
        # If the current character is the same as the previous character, 
        # increment the length of the current run
        if char == prev_char:
            current_run += 1
        # else, update the length of the longest run and reset the length of the
        # current run
        else:
            longest_run = max(longest_run, current_run)
            current_run = 1

        # Update the previous character
        prev_char = char

    # Update the length of the longest run
    longest_run = max(longest_run, current_run)

    # Return the length of the longest run
    return longest_run
