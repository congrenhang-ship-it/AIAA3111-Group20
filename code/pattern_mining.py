"""Priority 2: discretization and Apriori; implement after classification works."""


def mine_rules(train_data, feature_columns, bins=3, max_length=3):
    """TODO: fit bins on training data and mine short fraud-associated rules.

    Compare equal-width and equal-frequency bins. Restrict the feature count
    and itemset length, and require meaningful fraud coverage. Report support,
    confidence, lift, Kulczynski and supporting sample counts. Validate rules
    on held-out data. Class may be a rule consequent, never a predictive input.
    Return the binning information and mined rules for later validation.
    """
    raise NotImplementedError("Pattern mining is a later project stage.")

