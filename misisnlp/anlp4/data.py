from datasets import load_dataset


def load_wikipedia():
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru")["train"]
    ds = ds.train_test_split(
        train_size=100_000,
        test_size=5_000,
        shuffle=True,
        seed=42
    )
    return ds