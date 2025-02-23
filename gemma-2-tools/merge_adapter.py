import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import argparse

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.adapter_path,
        attn_implementation="eager",
        add_eos_token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.adapter_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = model.merge_and_unload()

    model.save_pretrained(
        args.output_path,
        safe_serialization=True
    )

    tokenizer.save_pretrained(args.output_path)

    print("Finished model merge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge adapter to base model.", add_help=True)

    parser.add_argument("--adapter_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)

    args = parser.parse_args()
    main(args)