{
    ("conv1", "weight"): partial(prune.random_unstructured, amount=0.5),
    ("conv1", "bias"): partial(prune.l1_unstructured, amount=0.5),
}