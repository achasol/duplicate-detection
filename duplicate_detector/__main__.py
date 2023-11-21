from concurrent.futures import ProcessPoolExecutor
from .utils import process_results
from .bootstrap import bootstrap_run_parallel


def run():
    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Use executor.map to run bootstrap_run in parallel
        results_list = list(executor.map(bootstrap_run_parallel, range(8)))
        for new_results in results_list:
            all_results.extend(new_results)

    process_results(all_results)
