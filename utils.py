import re
import json
from openai import OpenAI
import torch
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()
# Access the API key
gpt_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")


def remove_parentheses_content(text):
    # Remove content inside parentheses along with the parentheses
    return re.sub(r'\([^)]*\)', '', text).strip()


def remove_number_prefix(text):
    return re.sub(r'^\d+[.:]\s*', '', text)


def load_domain_map(jsonl_path):
    domain_map = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            domain_map[obj['domain']] = obj['retrieved_domains']
    return domain_map


def query_domains(domain_key, domain_map):
    return domain_map.get(domain_key, [])


def load_all_uids(jsonl_path):
    uids = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'task_uid' in data:
                    uids.append(data['task_uid'])
            except json.JSONDecodeError:
                continue
    return uids


def save_to_jsonl(data, filename):
    """Appends a list of dictionaries to a JSONL file."""
    with open(filename, 'a', encoding='utf-8') as file:  # Open in append mode
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")  # Ensure newline separation


def generate_output_from_api(model, prompt, script_args=None):
    """
    Generates output using the OpenAI API with the specified model and prompt.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        prompt (str): The text prompt to send to the model.
        script_args (dict, optional): Additional generation arguments (e.g., temperature, max_tokens).

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model type is not recognized.
    """
    if script_args is None:
        script_args = {}

    if 'qwen' in model.lower():
        client = OpenAI(api_key=qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    elif 'gpt' in model.lower():
        client = OpenAI(api_key=gpt_api_key)
    else:
        raise ValueError(f"Invalid model type: {model}. Model must contain either 'qwen' or 'gpt' in its name.")
        
    temperature = script_args.temperature
    max_tokens = script_args.max_length

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def generate_output_from_local(model, tokenizer, prompt, script_args=None):
    """
    Generates output using a local language model with the specified model, tokenizer and prompt.

    Args:
        model: The local language model to use for generation
        tokenizer: The tokenizer associated with the model
        prompt (str): The text prompt to send to the model
        script_args (dict, optional): Additional generation arguments (e.g., temperature, max_tokens)

    Returns:
        str: The generated response from the model
    """
    # Format the messages as chat-style prompt
    chat_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    # Tokenize the prompt
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=script_args.max_length,
            temperature=script_args.temperature,
            top_p=script_args.top_p,
            do_sample=script_args.do_sample,
            eos_token_id=tokenizer.eos_token_id)

    # Slice off the input tokens to get only the generated part
    generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]

    # Decode and print
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()



def extract_action(outputs):
            """
            Extract and clean action from model outputs using various parsing rules.
            
            Args:
                outputs (str): Raw output from the model
                
            Returns:
                str: Cleaned and normalized action string
            """
            # Initial cleaning
            ac = remove_number_prefix(outputs).lower()
            ac = remove_parentheses_content(ac)
            ac = ac.replace('*', '')

            # Extract action using different patterns
            if 'action:' in ac.lower():
                action = ac.lower().split('action:')[1].split('\n')[0].split(',')[0]
            elif 'the recommended next action is to' in ac:
                ac = ac.replace(':', '').strip()
                action = ac.split('the recommended next action is to')[-1].split('\n')[0].split('.')[0].split(',')[0].strip()
            elif 'the recommended next action is' in ac:
                ac = ac.replace(':', '').strip()
                action = ac.split('the recommended next action is')[-1].split('\n')[0].split('.')[0].split(',')[0].strip()
            elif 'the next action is' in ac:
                ac = ac.replace(':', '').strip()
                action = ac.split('the next action is')[-1].split('\n')[0].split('.')[0].split(',')[0].strip()
            elif 'next best action' in ac:
                ac = ac.replace(':', '').strip()
                action = ac.split('next best action')[-1].split('\n')[0].split('.')[0].split(',')[0].strip()
            else:
                ac = ac.replace(':', '').strip()
                action = ac.split('\n')[0].split(',')[0]

            # Final cleaning
            action = action.split('--')[0].strip()
            action = action.replace('.', '').strip()
            return action