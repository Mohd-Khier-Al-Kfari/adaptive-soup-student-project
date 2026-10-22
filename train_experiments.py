"""Student training entry point. Author: Moh'd Khier Al Kfari."""
from src.training.experiments import TRANSFERS
from src.training.train_transfer import run_transfer
from src.training.scheduler import run_all
def main():
    for i,t in enumerate(TRANSFERS,1): print(f"{i}. {t.family}: {t.name}")
    while True:
        answer=input("Select transfer by number/name, or all: ").strip().lower()
        if answer=="all":
            gpu=input("GPU index [0]: ").strip() or "0"
            try: run_all(gpu=int(gpu)); return
            except ValueError: print("Invalid GPU index."); continue
        selected=next((t for i,t in enumerate(TRANSFERS,1) if answer==str(i) or answer==t.name.lower()),None)
        if selected:
            try: print(f"COMPLETED: {run_transfer(selected)}")
            except Exception as exc: print(f"FAILED {selected.name}: {type(exc).__name__}: {exc}")
            return
        print("Invalid transfer. Enter a listed number, name, or all.")
if __name__=="__main__": main()
