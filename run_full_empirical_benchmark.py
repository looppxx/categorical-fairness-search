"""Multi-seed MovieLens/NeuMF fairness benchmark; see README for protocol details.

python run_full_empirical_benchmark.py --dataset movielens-1m --data data/ml-1m \
    --seeds 42 100 2026 --epochs 30 --generate-assets
"""
from fairness_lab.empirical import main


if __name__ == "__main__":
    raise SystemExit(main())
