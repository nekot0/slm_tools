import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

torch.set_float32_matmul_precision("high")
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        add_eos_token=True
    )

    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        model_kwargs={"torch_dtype": torch.float16},
        device="cuda"
    )

    messages = [
        {"role": "user", "content": args.prompt}
    ]

    print(f"User: {args.prompt}")

    start = time.time()
    outputs = pipe(messages, max_new_tokens=args.max_new_tokens, return_full_text=False)
    end = time.time()

    assistant_response = outputs[0]["generated_text"]

    num_tokens = len(tokenizer.encode(assistant_response, return_tensors="pt")[0])

    print(f"AI: {assistant_response}")
    print(f"-----time: {end-start:.2f} sec")
    print(f"-----generated tokens: {num_tokens} tokens")
    print(f"-----generation speed: {num_tokens/(end-start):.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference.", add_help=True)

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--max_new_tokens", required=False, type=int, default=512)
    
    parser.add_argument("--prompt", required=True, type=str)

    args = parser.parse_args()
    main(args)