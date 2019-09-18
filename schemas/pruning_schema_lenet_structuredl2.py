{
    ("conv1", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=2),
    ("conv2", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=2),
    ("fc1", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=2),
    ("fc2", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=2),
    ("fc3", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=2),
}
