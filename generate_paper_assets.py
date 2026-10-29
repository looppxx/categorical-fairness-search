"""Generate trace-backed PNG/SVG plots and a booktabs LaTeX results table.

python generate_paper_assets.py --results outputs/search_comparison_native/results.csv
python generate_paper_assets.py --results outputs/hybrid_expanded --out outputs/paper_hybrid
"""
import argparse
from pathlib import Path

from fairness_lab.publication import ExperimentVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True, help="CSV paths or folders with per-algorithm results.csv")
    parser.add_argument("--algorithms", nargs="+", default=["DPSO", "GA", "ACO"], help="Algorithm families, or ALL; exact method labels remain separate")
    parser.add_argument("--x-axis", choices=["iterations", "queries"], default="iterations")
    parser.add_argument("--block-size", type=int, help="Common requests per evaluation block, otherwise read from config")
    parser.add_argument("--context", help="Accurate figure context, e.g. MovieLens/NeuMF, 5 training epochs")
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_assets"))
    args = parser.parse_args()
    visualizer = ExperimentVisualizer(args.results, args.algorithms, args.x_axis, args.block_size, args.context)
    manifest = visualizer.generate(args.out)
    print((args.out / "results_table.tex").read_text(encoding="utf-8"))
    for note in manifest["notes"]:
        print(f"Note: {note}")
    print(f"Assets saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
