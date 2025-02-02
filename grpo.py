import uuid
import re
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from grpo_trainer import GRPOTrainer



@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "reflection", "length"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'length', 'reflection'"},
    )


def parse_gsm8k_gt(solution, **kwargs):
    gt_answer = solution.split("####")[-1]
    gt_answer = gt_answer.replace(",", "")
    gt_answer = float(gt_answer.rstrip())
    return gt_answer


def parse_gsm8k_pred(completion, **kwargs):
    try:
        pred_answer = float(re.findall(r'\d+(?:\.\d+)?', completion)[-1])
    except Exception as e:
        pred_answer = None
    return pred_answer



def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):

        gold_parsed = parse_gsm8k_gt(sol)
        answer_parsed = parse_gsm8k_pred(content)

        if gold_parsed == answer_parsed:
            reward = 1.0
        else:
            reward = 0.0

        rewards.append(reward)
    print(rewards)

    return rewards


def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    format_rewards = [1.0 if match else 0.0 for match in matches]
    print(format_rewards)
    return format_rewards

def length_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    # length_rewards = [len(content)/5000.0 for content in completion_contents]
    length_rewards = [0.5 if len(content) > 500 else 0.0 for content in completion_contents]
    print(length_rewards)
    return length_rewards


def reflection_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"].lower() for completion in completions]

    reflection_words = {
        "wait": 0.1,
        "possible": 0.1,
        "perhaps": 0.1,
        "check": 0.1,
        "perhaps": 0.1,
        "maybe": 0.1,
        "let me": 0.1,
        "would be": 0.1,
        "but the": 0.1,
        "wait but": 0.1,
        "check if": 0.1,
        "but how": 0.1,
        "but the": 0.1,
        "wait no": 0.1,
        "but wait": 0.1,
        "let me check": 0.1,
        "let me think": 0.1,
        "but let me": 0.1,
    }

    completion_contents_org = [completion[0]["content"] for completion in completions]
    self_rewards = [0.1 if " I " in content else 0 for content in completion_contents_org]
    
    scores = [sum(reflection_words.get(marker, 0.0) for marker in content.split()) for content in completion_contents]
    reflection_rewards = [score if score > 0.0 else 0.0 for score in scores]
    reflection_rewards = [reflection_reward + self_reward for reflection_reward, self_reward in zip(reflection_rewards, self_rewards)]

    return reflection_rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
    "reflection": reflection_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


if __name__ == "__main__":

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    unique_id = str(uuid.uuid4())[:8]
    print(unique_id)

    # set the output_dir to the unique id
    training_args.output_dir = f"{training_args.output_dir}/{unique_id}"
    training_args.run_name = f"{training_args.run_name}_{unique_id}"

    print(script_args)
    print(training_args)
    print(model_args)


    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset(script_args.dataset_name, name="main")


    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    dataset = dataset.rename_column("answer", "solution")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

