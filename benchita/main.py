import argparse

from benchita.task import get_tasks
from benchita.template import get_templates

from benchita.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Benchita")
    parser.add_argument("cmd", type=str, nargs="?", help="Command to execute", choices=["evaluate", "collect"])
    
    parser.add_argument("--task", type=str, required=True, help="The task to run", choices=get_tasks())
    parser.add_argument("--template", type=str, default="", help="The template to use", choices=get_templates() + [""])
    parser.add_argument("--model", type=str, required=True, help="The model to use")
    parser.add_argument("--tokenizer", type=str, default="", help="The tokenizer to use")

    parser.add_argument("--model-class", type=str, default="AutoModelForCausalLM", help="The model class to use")
    parser.add_argument("--tokenizer-class", type=str, default="AutoTokenizer", help="The tokenizer class to use")

    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--tokenizer-args", type=str, default="")
    parser.add_argument("--apply_chat_template-args", type=str, default="tokenize=False,add_generation_prompt=True")
    parser.add_argument("--generate-args", type=str, default="top_p=0.5,temperature=0.5,do_sample=True")
    
    parser.add_argument("--patch-tokenizer-pad", action="store_true", help="Patch the tokenizer to use eos_token as padding token")
    parser.add_argument("--force-template", action="store_true", help="Force the use of a template even if the tokenizer has a chat template")

    parser.add_argument("--batch-size", type=int, default=16, help="The batch size")
    parser.add_argument("--num-shots", type=int, default=3, help="The number of shots")
    parser.add_argument("--system-style", type=str, default="system", help="The style of the system prompt", choices=["system", "inject"])
    parser.add_argument("--device", type=str, default="cpu", help="The device to use")
    parser.add_argument("--max-length", type=int, default=1024, help="The device to use")
    parser.add_argument("--dtype", type=str, default="float32", help="The dtype to use")

    parser.add_argument("--dry-run", action="store_true", help="Dry run the task")
    parser.add_argument("--save-dir", type=str, default="./results", help="The directory to save the results")
    
    args = parser.parse_args()

    if args.tokenizer == "":
        args.tokenizer = args.model

    if args.cmd is None:
        args.cmd = "evaluate"

    if args.cmd == "evaluate":
        evaluate(args)

if __name__ == "__main__":
    main()