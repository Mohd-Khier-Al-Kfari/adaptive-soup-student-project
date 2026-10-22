"""Same-GPU transfer scheduler. Author: Moh'd Khier Al Kfari."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from .experiments import TRANSFERS
from .train_transfer import run_transfer
def run_all(max_jobs=3,gpu=0):
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        futures={pool.submit(run_transfer,t,gpu=gpu):t for t in TRANSFERS}
        for future in as_completed(futures):
            t=futures[future]
            try: print(f"COMPLETED {t.name}: {future.result()}")
            except Exception as exc: print(f"FAILED {t.name}: {type(exc).__name__}: {exc}")
