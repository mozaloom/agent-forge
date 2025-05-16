from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

peft_model_id = f"{"mozaloom"}/{"gemma-2-2B-it-thinking-function_calling-V0"}" # replace with your newly trained adapter
device = "auto"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map="auto",
                                             )
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(torch.bfloat16)
model.eval()

#this prompt is a sub-sample of one of the test set examples. In this example we start the generation after the model generation starts.
prompt="""<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
<start_of_turn>model
<think>"""

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.to("cuda") for k,v in inputs.items()}
outputs = model.generate(**inputs,
                         max_new_tokens=300,# Adapt as necessary
                         do_sample=True,
                         top_p=0.95,
                         temperature=0.01,
                         repetition_penalty=1.0,
                         eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))