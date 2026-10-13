import pandas as pd


def save_results(results, output_path):
    """Save backbone assessment results to a CSV file."""
    if not results:
        raise ValueError("No results to save.")

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(output_path, index=False)

    return dataframe