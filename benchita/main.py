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

    inference_inputs = []
    for elem in tqdm(task.build(num_shots=args.num_shots, system_style=args.system_style), total=len(task)):
        elem["prompt"] = tokenizer.apply_chat_template(
            elem["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        inference_inputs.append(elem)

    inference_ds = Dataset.from_list(inference_inputs)
    inference_ds = inference_ds.map(lambda x: tokenizer(x["prompt"], padding="max_length", truncation=False, max_length=args.max_length), batched=True)

    inference_outputs = []
    for i in tqdm(range(0, len(inference_ds), args.batch_size), total=len(inference_ds) // args.batch_size):
        batch = inference_ds[i:i+args.batch_size]
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
            
            outputs = tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
            inference_outputs.extend(outputs)

        break

    print(task.evaluate(inference_inputs, inference_outputs))


if __name__ == "__main__":
    main()