{
    ("features.0", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.3", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.6", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.8", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.11", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.13", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.16", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("features.18", "weight"): partial(prune.ln_structured, amount=0.2, n=1, axis=1),
    ("classifier.0", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("classifier.3", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("classifier.6", "weight"): partial(prune.l1_unstructured, amount=0.2),
}
