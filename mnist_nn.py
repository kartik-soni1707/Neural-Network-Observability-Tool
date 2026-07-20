import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(self, bad_init: bool = False, use_leaky: bool = False):
        super().__init__()
        act = nn.LeakyReLU(0.01) if use_leaky else nn.ReLU()
        # Sigmoid dropped: 10-class task -> raw logits + CrossEntropyLoss
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            act,
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        if bad_init:   # pathology: bias yanked negative -> layer-wide neuron death
            with torch.no_grad():
                self.model[0].bias.fill_(-50.0)
        else:          # healthy: Kaiming init, correct for ReLU-family activations
            nn.init.kaiming_normal_(self.model[0].weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.model[3].weight, nonlinearity='relu')

    def forward(self, images):
        return self.model(images)


# ---------- diagnostics (your tool) ----------
def dead_fractions(model: nn.Module, x: torch.Tensor):
    fracs = []
    for layer in model.model:
        x = layer(x)
        if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
            fracs.append(round((x <= 0).all(dim=0).float().mean().item(), 4))
    return fracs


def train(model, X, y, lr, epochs=30):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))
    return losses


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(256, 784)                # stand-in for MNIST batch (swap in real data)
    y = torch.randint(0, 10, (256,))

    # --- Pathology 1: bad init -> dead layer ---
    sick = MNISTClassifier(bad_init=True)
    print("dead fracs (bad init):", dead_fractions(sick, X))        # expect ~1.0
    print("loss stuck:", train(sick, X, y, lr=0.1)[:5], "...")

    # fix: re-initialize (the tool's recommendation)
    fixed = MNISTClassifier(bad_init=False)
    print("dead fracs (re-init):", dead_fractions(fixed, X))        # expect ~0.0
    print("loss recovering:", train(fixed, X, y, lr=0.1)[:5], "...")

    # --- Pathology 2: exploding loss from high LR ---
    hot = MNISTClassifier()
    print("high-LR losses:", train(hot, X, y, lr=25.0)[:5])         # expect blow-up/nan
    cool = MNISTClassifier()
    print("reduced-LR losses:", train(cool, X, y, lr=0.1)[:5])      # expect steady descent
    # ---------- ONNX export for Netron ----------
    model = MNISTClassifier()
    model.eval()                              # important: freezes Dropout out of the graph
    dummy = torch.randn(1, 784)               # example input; shapes get traced from it

    torch.onnx.export(
        model,
        dummy,
        "mnist_classifier.onnx",
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"},   # batch dim marked variable
                      "logits": {0: "batch"}},
    )
    print("saved mnist_classifier.onnx")