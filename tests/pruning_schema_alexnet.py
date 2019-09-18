{
    ("features.0", "weight"): partial(prune.random_unstructured, amount=0.5),
    ("features.0", "bias"): partial(prune.l1_unstructured, amount=0.5),
}