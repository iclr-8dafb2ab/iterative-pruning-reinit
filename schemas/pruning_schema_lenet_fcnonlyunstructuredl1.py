{
    ("fc1", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("fc2", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("fc3", "weight"): partial(prune.l1_unstructured, amount=0.2),
}
