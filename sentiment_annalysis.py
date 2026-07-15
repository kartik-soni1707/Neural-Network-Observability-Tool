"""
STANDALONE: Sentiment-analysis closed-loop validation of the observability tool.
  induce a pathology -> diagnose -> apply the recommended fix -> verify recovery.

Architecture: Embedding -> mean-pool over tokens -> Linear -> ReLU -> Linear (logit)
The mean-pooling step is why the whole model can't be one nn.Sequential; the
classifier head IS a Sequential so the diagnostics tool can iterate it.

Pathology 1: bad initialization  -> dead hidden layer -> fix: re-initialize
Pathology 2: learning rate too high -> exploding loss -> fix: reduce LR

Run: python sentiment_pathologies_standalone.py
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------- diagnostics
class Diagnostics:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # A neuron is dead if it outputs <= 0 for ALL samples in the batch.
        layers = model if isinstance(model, nn.Sequential) else model.model
        res = []
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
                if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                    n = x.shape[1]
                    dead = (x <= 0).all(dim=0).sum().item()
                    res.append(round(dead / n, 4))
        return res

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Priority-ordered rules over the WHOLE list -- compute evidence
        # first, decide after (no early returns from inside the scan).
        if not dead_fractions:
            return 'healthy'
        increasing = all(a < b for a, b in zip(dead_fractions, dead_fractions[1:]))
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        if increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        return 'healthy'


# ---------------------------------------------------------------------- model
class SentimentClassifier(nn.Module):
    def __init__(self, vocabulary_size: int,
                 bad_init: bool = False, use_leaky: bool = False):
        super().__init__()
        act = nn.LeakyReLU(0.01) if use_leaky else nn.ReLU()
        self.embed = nn.Embedding(vocabulary_size, 16)
        # Raw logit out: binary task -> pair with BCEWithLogitsLoss
        self.head = nn.Sequential(
            nn.Linear(16, 64),
            act,
            nn.Linear(64, 1),
        )
        if bad_init:
            with torch.no_grad():
                self.head[0].bias.fill_(-50.0)
        else:
            nn.init.kaiming_normal_(self.head[0].weight, nonlinearity='relu')

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T) int token ids -> (B, 16). Average the EMBEDDINGS, never the
        # ids: ids are categorical labels, vectors are the space where
        # averaging means "blend the meanings".
        return self.embed(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))     # (B, T) -> (B, 1) logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Inference-only probabilities; rounding lives here, NOT in forward
        # (torch.round has zero gradient -- it would silently kill training)
        with torch.no_grad():
            return torch.round(torch.sigmoid(self.forward(x)), decimals=4)


# ------------------------------------------------------------------- training
def train(model, x, y, lr, epochs=40):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))
    return losses


# ----------------------------------------------------------------- experiment
if __name__ == "__main__":
    torch.manual_seed(0)
    VOCAB, B, T = 500, 128, 12
    x = torch.randint(0, VOCAB, (B, T))          # stand-in token ids
    y = torch.randint(0, 2, (B, 1)).float()      # binary sentiment labels
    diag = Diagnostics()

    print("=" * 60)
    print("PATHOLOGY 1: bad init -> dead hidden layer")
    print("=" * 60)
    sick = SentimentClassifier(VOCAB, bad_init=True)
    fracs = diag.detect_dead_neurons(sick.head, sick.pool(x))
    print(f"dead fractions : {fracs}")
    print(f"diagnosis      : {diag.suggest_fix(fracs)}")
    print(f"loss (stuck)   : {train(sick, x, y, lr=0.5)[:5]} ...")

    print("\n-- applying fix: re-initialize --")
    fixed = SentimentClassifier(VOCAB, bad_init=False)
    fracs = diag.detect_dead_neurons(fixed.head, fixed.pool(x))
    print(f"dead fractions : {fracs}")
    print(f"diagnosis      : {diag.suggest_fix(fracs)}")
    print(f"loss (recovers): {train(fixed, x, y, lr=0.5)[:5]} ...")

    print("\n-- alternative fix: LeakyReLU (dead neurons keep gradient) --")
    leaky = SentimentClassifier(VOCAB, bad_init=True, use_leaky=True)
    print(f"loss (escapes) : {train(leaky, x, y, lr=0.5)[:8]} ...")

    print()
    print("=" * 60)
    print("PATHOLOGY 2: learning rate too high -> exploding loss")
    print("=" * 60)
    hot = SentimentClassifier(VOCAB)
    print(f"lr=200 losses  : {train(hot, x, y, lr=200.0)[:5]}  <- blows up")
    print("\n-- applying fix: reduce learning rate --")
    cool = SentimentClassifier(VOCAB)
    print(f"lr=0.5 losses  : {train(cool, x, y, lr=0.5)[:5]}  <- steady descent")

    print("\nsample predictions:", cool.predict(x[:3]).squeeze(1).tolist())

# To run on a real dataset, replace x, y with tokenized reviews (e.g. IMDB):
# build a vocab, map tokens to ids, pad to equal length T, stack to (B, T).