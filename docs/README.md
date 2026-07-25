# `docs/`

## `architettura_neural_bt.svg`

Architecture diagram of the neural Bradley-Terry model, annotated with tensor shapes.

The upper part is a single tower `f_theta`, applied with shared weights to both teams:
embeddings for the five categorical features, mean-pooling over the four moves (order
invariance), a per-Pokemon MLP, self-attention across the six Pokemon (where synergy arises),
invariant mean-pooling to a team vector, and a scalar head. The lower part is the siamese
combination: the same tower scores both teams, their difference plus a side bias gives the
logit, a sigmoid gives the probability — enforcing antisymmetry by construction.

The two pooling operations and the absence of positional encoding are highlighted because they
are why this is not a plain feed-forward network: a team is an unordered set, not a sequence.