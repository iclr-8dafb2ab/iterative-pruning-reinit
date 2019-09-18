{
    ("conv1", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("conv2", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("fc1", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("fc2", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("fc3", "weight"): partial(prune.l1_unstructured, amount=0.2),
}
