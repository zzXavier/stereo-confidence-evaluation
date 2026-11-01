import argparse
from pathlib import Path
from src.benchmark import evaluate, reproduce
from src.fetch import CONFIG

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["demo","reproduce","evaluate"])
    p.add_argument("--cache",type=Path,default=Path("cache"))
    p.add_argument("--out",type=Path)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    out=args.out or Path("scratch/demo" if args.action=="demo" else "results")
    if args.action=="evaluate":
        evaluate(out)
    else:
        names=CONFIG["development"] if args.action=="demo" else CONFIG["test"]
        reproduce(args.cache.resolve(),out,args.device,names)
