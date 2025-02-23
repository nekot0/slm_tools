import time
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop="<|eot_id>"
    )

    system_prompt = args.system_prompt
    user_prompt = args.prompt

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"User: {user_prompt}")

    start = time.time()
    output = llm.generate(prompt, sampling_params)
    end = time.time()

    num_tokens = len(tokenizer(output[0].outputs[0].text)["input_ids"])

    print(output[0].outputs[0].text)
    print(f"-----time: {end-start:.2f} sec")
    print(f"-----generated tokens: {num_tokens} tokens")
    print(f"-----generation speed: {num_tokens/(end-start):.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference.", add_help=True)

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--temperature", required=False, type=float, default=0.6)
    parser.add_argument("--top_p", required=False, type=float, default=0.9)
    parser.add_argument("--max_tokens", required=False, type=int, default=512)
    
    parser.add_argument("--system_prompt", required=False, type=str, default="あなたは誠実で優秀な日本人のアシスタントです。")
    parser.add_argument("--prompt", required=True, type=str)

    args = parser.parse_args()
    main(args)
