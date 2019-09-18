{
    ("conv1", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=float('-inf')),
    ("conv2", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=float('-inf')),
    ("fc1", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=float('-inf')),
    ("fc2", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=float('-inf')),
    ("fc3", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=float('-inf')),
}
