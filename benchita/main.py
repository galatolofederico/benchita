import argparse
import transformers
from datasets import Dataset
from tqdm import tqdm
import torch

from benchita.task import get_tasks, get_task

def main():
    parser = argparse.ArgumentParser(description='Benchita')
    parser.add_argument('--task', type=str, required=True, help='The task to run', choices=get_tasks())
    parser.add_argument('--model', type=str, required=True, help='The model to use')
    parser.add_argument('--tokenizer', type=str, default="", help='The tokenizer to use')

    parser.add_argument('--model-class', type=str, default="AutoModel", help='The model class to use')
    parser.add_argument('--tokenizer-class', type=str, default="AutoTokenizer", help='The tokenizer class to use')

    parser.add_argument('--batch-size', type=int, default=16, help='The batch size')
    parser.add_argument('--num-shots', type=int, default=3, help='The number of shots')
    parser.add_argument('--system-style', type=str, default="system", help='The style of the system prompt', choices=["system", "inject"])
    parser.add_argument('--device', type=str, default="cpu", help='The device to use')
    parser.add_argument('--max-length', type=int, default=1024, help='The device to use')
    
    parser.add_argument("--top-p", type=float, default=0.5, help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    
    args = parser.parse_args()

    if args.tokenizer == "":
        args.tokenizer = args.model

    model_cls = getattr(transformers, args.model_class)
    tokenizer_cls = getattr(transformers, args.tokenizer_class)
    task_cls = get_task(args.task)
    
    task = task_cls()
    model = model_cls.from_pretrained(args.model).to(args.device)
    tokenizer = tokenizer_cls.from_pretrained(args.tokenizer)

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        raise ValueError("Only tokenizer chat templates are supported for now")

    prompts = []
    expecteds = []
    for elem in tqdm(task.build(num_shots=args.num_shots, system_style=args.system_style), total=len(task)):
        messages = elem["messages"]
        expected = elem["expected"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt)
        expecteds.append(expected)

    ds = Dataset.from_dict({"prompt": prompts, "expected": expecteds})
    ds = ds.map(lambda x: tokenizer(x["prompt"], padding="max_length", truncation=False, max_length=args.max_length), batched=True)

    results = []
    for i in tqdm(range(0, len(ds), args.batch_size), total=len(ds) // args.batch_size):
        batch = ds[i:i+args.batch_size]
        input_ids = torch.tensor(batch["input_ids"], device=args.device)
        attention_mask = torch.tensor(batch["attention_mask"], device=args.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=task.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=True
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, o in zip(batch["prompt"], outputs):
                print(i, o)
                results.append(o[len(i):].strip())
        break
    
    print(expecteds)
    print(results)

if __name__ == "__main__":
    main()