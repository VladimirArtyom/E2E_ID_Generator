
import pandas as pd
import datasets
import argparse
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from dataset import QAGEvaluatorDataset
from trainer import Trainer

#spacy.prefer_gpu()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pad_mask_id", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--save_dir", type=str, default="./saved_model_QAG_evaluator/")
    parser.add_argument("--pin_memory", dest="pin_memory",
                         action="store_true", default=False)
    parser.add_argument("--metrics_evaluation", dest="metrics_evaluation",
                         action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = datasets.load_dataset("Voslannack/squad_id_train")
    train_dataset = QAGEvaluatorDataset(dataset["train"], args.max_length, tokenizer)
    val_dataset = QAGEvaluatorDataset(dataset["validation"], args.max_length, tokenizer)
    test_dataset = QAGEvaluatorDataset(dataset["test"], args.max_length, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    user_dir = "./dir/"
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        workers=args.workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        pin_memory=args.pin_memory,
        save_dir=user_dir,
        train_set=train_dataset,
        validation_set=val_dataset,
        test_set=test_dataset,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.valid_batch_size,
        test_batch_size=args.test_batch_size,
        metrics_evaluation=True
    )    

    trainer.train()

