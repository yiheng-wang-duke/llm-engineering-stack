import argparse
import logging
import os
from data_utils import LeetCodeDataset, OpenR1MathDataset, build_mixed_dataloader
from model import load_model
from trainer import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    # data
    parser.add_argument("--dataset", type=str, default="leetcode", choices=["leetcode", "openr1math"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--openr1_subset", type=str, default="all", choices=["all", "default", "extended"])
    parser.add_argument("--total_size", type=int, default=50000)
    # training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. load model & tokenizer
    model, tokenizer = load_model(args.model_path)
    model.gradient_checkpointing_enable()
    print("model and tokenizer built")

    # 2. load dataset & dataloader
    if args.dataset == "leetcode":
        assert args.data_path is not None, "--data_path is required for leetcode dataset"
        dataset = LeetCodeDataset(args.data_path, split="train")
    elif args.dataset == "openr1math":
        data_path = args.data_path if args.data_path else "open-r1/OpenR1-Math-220k"
        dataset = OpenR1MathDataset(data_path, subset=args.openr1_subset)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataloader = build_mixed_dataloader(
        datasets=[dataset],
        proportions=[1.0],
        total_size=args.total_size,
        batch_size=args.batch_size,
    )
    print("dataset and dataloader built")

    os.makedirs(args.save_dir, exist_ok=True)

    # 3. train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_dir=args.save_dir,
    )
    print("trainer built, start training")
    trainer.train()


if __name__ == "__main__":
    main()