def gini_impurity(counts):

    # Calculate the Gini impurity for a list of label counts.
    # Parameter:
    #     counts: The counts for each label.
    # Returns:
    #     The Gini impurity

    total = sum(counts)
    if total == 0:
        return 0.0
    # Compute probabilities for each label
    probabilities = [count / total for count in counts]
    # Compute Gini impurity
    impurity = 1 - sum(p ** 2 for p in probabilities)
    return impurity


# Example usage:
counts = [25, 60, 5, 10]  # corresponding to 0.25, 0.6, 0.05, 0.1 probabilities
print("Gini impurity:", gini_impurity(counts))
