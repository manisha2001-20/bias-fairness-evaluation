"""Command-line evaluation runner.

Examples:
  python scripts/run_eval.py --model ollama:qwen2.5:3b --eval coreference
  python scripts/run_eval.py --model ollama:deepseek-r1:1.5b --eval generation --max-per-group 6
  python scripts/run_eval.py --model sim:0.60 --eval all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bias_eval.datasets.loaders import load_bold, load_winogender  # noqa: E402
from bias_eval.evaluations import coreference, generation  # noqa: E402
from bias_eval.models.base import ModelError  # noqa: E402
from bias_eval.models.registry import build  # noqa: E402
from bias_eval.results_store import save_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. ollama:qwen2.5:3b, ollama:deepseek-r1:1.5b, sim:0.60")
    ap.add_argument("--eval", choices=["coreference", "generation", "all"], default="all")
    ap.add_argument("--winogender", help="Winogender-format TSV (default: bundled synthetic templates)")
    ap.add_argument("--bold", help="BOLD-format JSON (default: bundled synthetic prompts)")
    ap.add_argument("--bold-domain", help="domain label when --bold is an official per-domain BOLD file")
    ap.add_argument("--max-per-group", type=int, default=None)
    args = ap.parse_args()

    model = build(args.model)
    ok, msg = model.is_available()
    if not ok:
        print(f"Model unavailable: {msg}")
        return 2
    try:
        if args.eval in ("coreference", "all"):
            items = load_winogender(args.winogender)
            df = coreference.run(model, items, progress=lambda p: print(f"\rcoreference {p:5.0%}", end=""))
            s = coreference.summarise(df)
            path = save_run(model.info, "coreference", s, df, args.winogender or "winogender-style (synthetic sample)")
            print(f"\nbias score {s['bias_score']} | flip rate {s['flip_rate']:.3f} | accuracy {s['accuracy_by_gender']} -> {path.name}")
        if args.eval in ("generation", "all"):
            prompts = load_bold(args.bold, domain=args.bold_domain, max_per_group=args.max_per_group)
            df = generation.run(model, prompts, progress=lambda p: print(f"\rgeneration {p:5.0%}", end=""))
            s = generation.summarise(df)
            path = save_run(model.info, "generation", s, df, args.bold or "bold-style (synthetic sample)")
            print(f"\nbias score {s['bias_score']} | " + ", ".join(f"{d}: gap {v['max_gap']:.3f} (p={v['p_value_max_gap']:.3f})" for d, v in s["domains"].items()) + f" -> {path.name}")
    except ModelError as exc:
        print(f"Model error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
