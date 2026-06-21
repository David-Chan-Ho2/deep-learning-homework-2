# https://colab.research.google.com/drive/1TzdeU7dIRETh0frsCh5WOZS8-1wkMAYm?usp=sharing 

import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)

        self.data = [self.char_to_idx[ch] for ch in text]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class CharModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_type="LSTM",
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        fc_size=None
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        if model_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be LSTM or GRU")

        if fc_size is None:
            self.fc = nn.Linear(hidden_size, vocab_size)
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, fc_size),
                nn.ReLU(),
                nn.Linear(fc_size, vocab_size)
            )

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    return count_parameters(model) * 4 / (1024 ** 2)


def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

    training_time = time.time() - start_time

    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    return avg_train_loss, avg_val_loss, val_accuracy, training_time


def generate_text(model, dataset, seed_text, device, length=300):
    model.eval()

    result = seed_text

    for _ in range(length):
        input_seq = result[-dataset.seq_len:]

        while len(input_seq) < dataset.seq_len:
            input_seq = " " + input_seq

        x = torch.tensor(
            [dataset.char_to_idx.get(ch, 0) for ch in input_seq],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

        next_char = dataset.idx_to_char[next_idx]
        result += next_char

    return result


def measure_inference_time(model, dataset, device):
    seed = "To be or not to be"
    start = time.time()
    _ = generate_text(model, dataset, seed, device, length=200)
    return time.time() - start


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    results = []
    sample_outputs = []

    experiments = []

    for model_type in ["LSTM", "GRU"]:
        for seq_len in [20, 30, 50]:
            for hidden_size in [128, 256]:
                for num_layers in [1, 2]:
                    for fc_size in [None, 128]:
                        experiments.append({
                            "model_type": model_type,
                            "seq_len": seq_len,
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "fc_size": fc_size
                        })

    for exp in experiments:
        print("\n====================================")
        print(exp)
        print("====================================")

        dataset = CharDataset(text, exp["seq_len"])

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = CharModel(
            vocab_size=dataset.vocab_size,
            model_type=exp["model_type"],
            embed_size=64,
            hidden_size=exp["hidden_size"],
            num_layers=exp["num_layers"],
            fc_size=exp["fc_size"]
        ).to(device)

        train_loss, val_loss, val_acc, train_time = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=5,
            lr=0.001
        )

        inference_time = measure_inference_time(model, dataset, device)

        sample_text = generate_text(
            model,
            dataset,
            seed_text="To be or not to be",
            device=device,
            length=300
        )

        params = count_parameters(model)
        size_mb = model_size_mb(model)

        result = {
            "model": exp["model_type"],
            "sequence_length": exp["seq_len"],
            "hidden_size": exp["hidden_size"],
            "num_layers": exp["num_layers"],
            "fc_size": exp["fc_size"],
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "training_time_sec": train_time,
            "inference_time_sec": inference_time,
            "parameters": params,
            "model_size_MB": size_mb
        }

        results.append(result)

        sample_outputs.append(
            f"\n\nMODEL: {exp}\n"
            f"TRAIN LOSS: {train_loss:.4f}\n"
            f"VAL ACC: {val_acc:.4f}\n"
            f"SAMPLE OUTPUT:\n{sample_text}\n"
        )

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)

    df.to_csv("problem2_results.csv", index=False)

    with open("problem2_sample_outputs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sample_outputs))

    print("\nSaved results to problem2_results.csv")
    print("Saved generated text to problem2_sample_outputs.txt")


if __name__ == "__main__":
    main()