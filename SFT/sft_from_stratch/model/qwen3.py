import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name_or_path: str, dtype=torch.bfloat16, use_device_map: bool = False):
    """Load Qwen3 model and tokenizer via Auto classes.

    Args:
        use_device_map: 仅推理时设为 True。训练时用 accelerate 管理设备，不要开。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if use_device_map else None,
        trust_remote_code=True,
    )
    return model, tokenizer


if __name__ == "__main__":
    model_path = "Qwen/Qwen3-4B"
    model, tokenizer = load_model(model_path, use_device_map=True)

    messages = [{"role": "user", "content": "Hello, who are you?"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(response)
