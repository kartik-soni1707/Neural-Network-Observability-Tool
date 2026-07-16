# NetDoc — Neural Network Training Observability Tool

One-liner: Layer-wise diagnostics for PyTorch training runs — detects
vanishing/exploding gradients and dead neurons, prescribes fixes, and
verifies they work.

## The problem
Training runs fail silently: a bad init or hot learning rate kills a model
in the first steps, but you only notice hours of compute later as a flat
loss curve. This tool names the disease before the budget burns.

## Demo  ← the most important section
[before/after loss curves image, or the terminal output block]

    dead fractions : [1.0]
    diagnosis      : use_leaky_relu
    loss (stuck)   : [2.3026, 2.3026, ...]      <- ln(10): random guessing
    -- fix applied: reinitialize --
    dead fractions : [0.0]
    loss (recovers): [3.1815, 2.6702, 2.1493, ...]

## How it works
Forward hooks collect per-layer activation stats (mean, std, dead-neuron
fraction) and gradient norms. Failure signatures map to diagnoses:
- first-layer death        -> bad init        -> reinitialize
- death increasing w/depth -> LR too high     -> reduce learning rate
- any layer >50% dead      -> activation trap -> LeakyReLU

## Quickstart
    pip install torch
    python mnist_pathologies.py       # full closed loop in ~10s, CPU

## Repo structure
    dead_relu.py               the diagnostics tool
    mnist_pathologies.py       induce -> diagnose -> fix -> verify (MNIST)
    sentiment_pathologies.py   same loop, embedding-based sentiment model
    numpy_reference/           from-scratch backprop, verified vs finite differences

## What I learned / design notes  (optional but great for portfolio repos)
Why (x <= 0) not (x == 0), why LeakyReLU recovers slowly from init-death
while re-init is instant, BCE-with-logits vs sigmoid+BCE, etc.