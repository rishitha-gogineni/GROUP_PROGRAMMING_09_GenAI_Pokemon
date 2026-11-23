
"""Train Kavya's two MLPs using heuristic-teacher demonstrations."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from team_a.src.models.kavya.features import encode_state, encode_options
from team_a.src.models.kavya.model_1 import Model as Model1
from team_a.src.models.kavya.model_2 import Model as Model2

DATA_DIR = Path("team_a/experiments/day4/training")
OUTPUT_DIR = Path("team_a/experiments/kavya")


def load_data(split):
    examples = []

    with (DATA_DIR / f"{split}.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            obs = record["observation"]
            state = encode_state(obs)
            options = encode_options(obs)
            scores = record["teacher_scores"]

            if len(options) < 2 or len(options) != len(scores):
                continue

            maximum = max(scores)
            winners = [i for i, s in enumerate(scores) if s == maximum]

            if not winners:
                continue

            target = torch.zeros(len(scores), dtype=torch.float32)
            target[winners] = 1.0 / len(winners)

            examples.append((
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(options, dtype=torch.float32),
                target,
            ))

    return examples


def evaluate(model, examples, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for state, options, target in examples:
            logits = model(state.to(device), options.to(device))
            loss = -(target.to(device) * F.log_softmax(logits, dim=0)).sum()

            total_loss += loss.item()
            correct += int(target[logits.argmax().item()] > 0)

    return total_loss / len(examples), correct / len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, choices=[1, 2], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_data = load_data("train")
    val_data = load_data("validation")
    test_data = load_data("held_out")

    model = (Model1() if args.model == 1 else Model2()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / f"model_{args.model}.pt"

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_data)

        for state, options, target in train_data:
            state = state.to(device)
            options = options.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(state, options)

            loss = -(target * F.log_softmax(logits, dim=0)).sum()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_data, device)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, test_acc = evaluate(model, test_data, device)

    print(f"\nHeld-out loss: {test_loss:.4f}")
    print(f"Held-out accuracy: {test_acc:.2%}")
    print(f"Saved: {checkpoint}")


if __name__ == "__main__":
    main()
