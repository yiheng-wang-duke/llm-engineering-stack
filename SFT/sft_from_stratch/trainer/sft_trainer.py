import os
import logging
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from tqdm import tqdm

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(self, model, tokenizer, dataloader, **config):
        # training config
        default_train_cfg = {
            "optimizer": "AdamW",
            "lr": 1e-5,
            "epochs": 1,
            "max_seq_len": 2048,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
            "log_interval": 10,
            "save_dir": "checkpoints",
        }
        self.train_cfg = {**default_train_cfg, **config}

        self.tokenizer = tokenizer
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_cfg["gradient_accumulation_steps"],
            mixed_precision=self.train_cfg["mixed_precision"],
        )

        self._build_optimizer(model)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_cfg["epochs"] * len(dataloader),
        )

        # accelerate prepare
        self.model, self.optimizer, self.dataloader, self.scheduler = (
            self.accelerator.prepare(model, self.optimizer, dataloader, self.scheduler)
        )

    def _build_optimizer(self, model):
        opt_name = self.train_cfg["optimizer"]
        lr = self.train_cfg["lr"]
        if opt_name == "AdamW":
            self.optimizer = AdamW(model.parameters(), lr=lr)
        elif opt_name == "Adam":
            self.optimizer = Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer '{opt_name}' not supported")

    @staticmethod
    def _to_messages(sample):
        """将一条数据转成 chat messages 格式。

        支持两种输入:
        1. 已经是 list[dict] (messages 格式) → 直接返回
        2. dict 带 query/response 字段 → 转成 user/assistant 对话
        """
        if isinstance(sample, list):
            return sample
        return [
            {"role": "user", "content": sample["query"]},
            {"role": "assistant", "content": sample["response"]},
        ]

    def _find_assistant_start(self, input_ids):
        """找到 assistant 回复开始的位置（<|im_start|>assistant\\n 之后）。

        Returns the index of the first token AFTER the assistant header.
        If not found, returns 0 (fall back to computing loss on everything).
        """
        # <|im_start|> = 151644, assistant = 77091, \n = 198
        header = [151644, 77091, 198]
        ids = input_ids.tolist()
        for i in range(len(ids) - len(header) + 1):
            if ids[i:i + len(header)] == header:
                return i + len(header)
        return 0

    def _tokenize(self, batch):
        """将一个 batch 转成模型输入，只对 assistant 部分计算 loss。"""
        texts = [
            self.tokenizer.apply_chat_template(
                self._to_messages(sample), tokenize=False
            )
            for sample in batch
        ]
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.train_cfg["max_seq_len"],
        )

        # Mask labels: only compute loss on assistant response tokens
        input_ids = encodings["input_ids"]
        labels = input_ids.clone()
        for i in range(labels.size(0)):
            assistant_start = self._find_assistant_start(input_ids[i])
            # Mask everything before assistant response
            labels[i, :assistant_start] = IGNORE_INDEX
            # Also mask padding tokens
            labels[i, input_ids[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        encodings["labels"] = labels
        return encodings

    def train(self):
        self.model.train()
        global_step = 0

        for epoch in range(self.train_cfg["epochs"]):
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1}/{self.train_cfg['epochs']}")

            progress = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress:
                with self.accelerator.accumulate(self.model):
                    inputs = self._tokenize(batch)
                    inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs)
                    loss = outputs.loss

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
                if global_step % self.train_cfg["log_interval"] == 0:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )

            # 每个 epoch 结束保存
            self.save_model(
                os.path.join(self.train_cfg["save_dir"], f"epoch_{epoch + 1}")
            )

    def save_model(self, save_dir):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"Model saved to {save_dir}")