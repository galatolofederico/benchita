import argparse
import transformers
from datasets import Dataset
from tqdm import tqdm
import torch

from benchita.task import get_tasks, get_task
from benchita.template import get_template, get_templates
from benchita.utils import parse_str_args
from benchita.logging import log_info, log_warn, log_error

def main():
    parser = argparse.ArgumentParser(description='Benchita')
    parser.add_argument('--task', type=str, required=True, help='The task to run', choices=get_tasks())
    parser.add_argument('--template', type=str, default="", help='The template to use', choices=get_templates() + [""])
    parser.add_argument('--model', type=str, required=True, help='The model to use')
    parser.add_argument('--tokenizer', type=str, default="", help='The tokenizer to use')

    parser.add_argument('--model-class', type=str, default="AutoModelForCausalLM", help='The model class to use')
    parser.add_argument('--tokenizer-class', type=str, default="AutoTokenizer", help='The tokenizer class to use')

    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--tokenizer-args", type=str, default="")
    parser.add_argument("--apply_chat_template-args", type=str, default="tokenize=False,add_generation_prompt=True")

    parser.add_argument('--patch-tokenizer-pad', action="store_true", help='Patch the tokenizer to use eos_token as padding token')
    parser.add_argument('--force-template', action="store_true", help='Force the use of a template even if the tokenizer has a chat template')

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

    model_args = parse_str_args(args.model_args)
    tokenizer_args = parse_str_args(args.tokenizer_args)
    apply_chat_template_args = parse_str_args(args.apply_chat_template_args)

    task = task_cls()
    model = model_cls.from_pretrained(args.model, **model_args).to(args.device)
    tokenizer = tokenizer_cls.from_pretrained(args.tokenizer, **tokenizer_args)

    if ((hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None) and args.template != "") and not args.force_template:
        log_error("Tokenizer has a chat template, but a template was provided use --force-template to ignore this error")

    if (not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None) and args.template == "":
        log_warn("Tokenizer does not have a chat template and no template was provided, using 'default' template")
        args.template = "default"

    if args.template != "":
        template = get_template(args.template)()
        chat_template = template.apply_chat_template
    else:
        chat_template = tokenizer.chat_template

    if args.patch_tokenizer_pad:
        tokenizer.pad_token = tokenizer.eos_token

    inference_inputs = []
    for elem in tqdm(task.build(num_shots=args.num_shots, system_style=args.system_style), total=len(task)):
        elem["prompt"] = chat_template(
            elem["messages"],
            **apply_chat_template_args
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