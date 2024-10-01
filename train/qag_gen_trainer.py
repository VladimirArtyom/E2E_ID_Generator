import argparse
import datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW
from dataset import QAGDataset
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Thesis/saved_model_QG_generator")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin_memory", dest="pin_memory",
                        action="store_true", default=False)
    return parser.parse_args()

def prep_model(model_name: str,
               device: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

def prep_tokenizer(model_name: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<answer>", "<context>"]
    })
    return tokenizer


if __name__ == "__main__":
    args = parse_args()
    tokenizer = prep_tokenizer(args.model_name)
    model = prep_model(args.model_name, args.device, tokenizer)
    dataset = datasets.load_dataset("Voslannack/squad_id_train")
    train_set = QAGDataset(dataset["train"], args.max_length, args.pad_mask_id, tokenizer)
    validation_set = QAGDataset(dataset["validation"], args.max_length, args.pad_mask_id, tokenizer)
    test_set = QAGDataset(dataset["test"], args.max_length, args.pad_mask_id, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(
        epochs=args.epochs,
        workers=args.workers,
        device=args.device,
        learning_rate=args.learning_rate,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        test_batch_size=args.test_batch_size,
        train_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
        save_dir=args.save_dir,
        pin_memory=args.pin_memory,
        metrics_evaluation=False
    )

    trainer.train()
