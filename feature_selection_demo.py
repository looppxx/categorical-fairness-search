"""Run the same categorical selection pipeline at 10, 20 and 200 dimensions."""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from fairness_lab.categorical_selection import CategoricalUnionBackwardSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", nargs="+", type=int, default=[10, 20, 200])
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--top-n", type=int, default=4)
    args = parser.parse_args()
    rows = []
    for p in args.tiers:
        if p < 3:
            raise ValueError("Demo needs at least 3 features")
        rng = np.random.default_rng(42)
        values = rng.integers(0, 4, size=(args.samples, p))
        y = ((values[:, 0] == 0) | ((values[:, 1] > 1) & (values[:, 2] == 2))).astype(int)
        y ^= (rng.random(args.samples) < 0.05).astype(int)
        X = pd.DataFrame(values.astype(str), columns=[f"category_{j}" for j in range(p)])
        train, test = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=42)
        estimator = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                  DecisionTreeClassifier(max_depth=5, random_state=42))
        selector = CategoricalUnionBackwardSelector(top_n=args.top_n, max_features=3,
                    estimator=estimator, cv=2)
        start = perf_counter()
        selected_train = selector.fit_transform(X.iloc[train], y[train])
        selected_test = selector.transform(X.iloc[test])
        estimator.fit(selected_train, y[train])
        rows.append(dict(input_features=p, filter_union=len(selector.filter_union_),
                         wrapper_features=len(selector.selected_),
                         heldout_balanced_accuracy=balanced_accuracy_score(y[test], estimator.predict(selected_test)),
                         seconds=perf_counter() - start,
                         selected=",".join(selector.get_feature_names_out())))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
