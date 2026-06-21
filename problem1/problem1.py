# https://colab.research.google.com/drive/167yjdHiisPYVUKzmOEnulzkMcNmnoCpN?usp=sharing

import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


TEXT = """
Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology.
"""


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
    def __init__(self, vocab_size, model_type, embed_size=64, hidden_size=128, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        if model_type == "RNN":
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid model type")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    params = count_parameters(model)
    return params * 4 / (1024 ** 2)


def train_model(model, train_loader, val_loader, device, epochs=30, lr=0.001):
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

    train_time = time.time() - start_time

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

    return avg_train_loss, avg_val_loss, val_accuracy, train_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    results = []

    for model_type in ["RNN", "LSTM", "GRU"]:
        for seq_len in [10, 20, 30]:
            print("\n====================================")
            print(f"Model: {model_type}, Sequence Length: {seq_len}")
            print("====================================")

            dataset = CharDataset(TEXT, seq_len)

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            model = CharModel(
                vocab_size=dataset.vocab_size,
                model_type=model_type,
                embed_size=64,
                hidden_size=128,
                num_layers=1
            ).to(device)

            train_loss, val_loss, val_acc, train_time = train_model(
                model,
                train_loader,
                val_loader,
                device,
                epochs=30,
                lr=0.001
            )

            params = count_parameters(model)
            size_mb = model_size_mb(model)

            results.append({
                "model": model_type,
                "sequence_length": seq_len,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "training_time_sec": train_time,
                "parameters": params,
                "model_size_MB": size_mb
            })

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)

    df.to_csv("problem1_results.csv", index=False)
    print("\nSaved results to problem1_results.csv")


if __name__ == "__main__":
    main()