import sys
import os
import json
import uuid
import asyncio
import logging
import traceback
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import time
from typing import List
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
import torch

app = FastAPI()

request_queue = asyncio.Queue()
result_holder = {}
processing_events = {}

with open('settings.json') as settings_file:
    settings = json.load(settings_file)
    
model_name_or_path = settings['model_name_or_path']
trunc_length = settings.get('default_truncation_length', "4096")
formatted_model_name = model_name_or_path.replace('/', '_')
model_directory = f"./models/{formatted_model_name}"
config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)

class Item(BaseModel):
    
    system_prompt: str = "You are a helpful, respectful, and honest assistant."
    prompt: str
    Username: str = "USER"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.45
    top_k: int = 40
    truncation_length: int = int(trunc_length)
    repetition_penalty: float = 1.15
    LLM_Template: str = "Llama_2_Chat"



with open('settings.json', 'r') as file:
    settings = json.load(file)
max_vram_gb = settings.get("Max_VRAM_GB", None)
Auto_Allocate_Resources = settings.get("Auto_Allocate_Resources", "True")
if Auto_Allocate_Resources == "False":
    if max_vram_gb is not None:
        max_vram_mb = max_vram_gb * 1024
        num_splits = torch.cuda.device_count()
        vram_per_split = max_vram_mb // num_splits
        gpu_split = [vram_per_split] * num_splits
    else:
        gpu_split = None 
    model.load_autosplit(cache=cache, reserve_vram=gpu_split)
else:
    model.load_autosplit(cache=cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.warmup()

def create_generator_settings(item):
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = item.temperature
    settings.top_k = item.top_k
    settings.top_p = item.top_p
    settings.token_repetition_penalty = item.repetition_penalty
    return settings

async def process_requests():
    timeout = 5000  # timeout for each request
    while True:
        logging.info("Waiting for the next request...")
        request_id, item = await request_queue.get()
        logging.info(f"Processing request: {request_id}")
        with open('settings.json') as settings_file:
            settings = json.load(settings_file)
            
        Use_Injection = settings['Use_Injection_Prompt']
        try:
            time_begin = time.time()
            truncation_length = getattr(item, 'truncation_length', None)
            max_new_tokens = getattr(item, 'max_new_tokens', None)
        #    truncation_length = truncation_length - max_new_tokens
            llm_template = getattr(item, 'LLM_Template', None)
            username = getattr(item, 'Username', None)
            bot_name = getattr(item, 'Bot_Name', None)
            if llm_template:
                delattr(item, 'LLM_Template')
            if username:
                delattr(item, 'Username')
            if bot_name:
                delattr(item, 'Bot_Name')
            prompt_template = " "
            prompt_overhang = False
            if Use_Injection == "True":
                with open('Injection_Prompt.txt', 'r') as file:
                    Injection_Prompt = file.read()
                item.prompt = f"{Injection_Prompt}\n{item.prompt}"
                
    # Llama-2-Chat Format    
            if llm_template == "Llama_2_Chat":
                end_token = "[/INST]"
                prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt} [/INST]"
                system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: [/INST]"
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt} [/INST]"
                    system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: [/INST]"
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
                    
            if llm_template == "Llama_2_Chat_No_End_Token":
                end_token = "[/INST]"
                prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt}"
                system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: "
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt}"
                    system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: "
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
        
        
    # Alpaca Format
            if llm_template == "Alpaca":
                end_token = "\n\n### Response:"
                prompt_template = f"{item.system_prompt}\n\n### Instruction:\n{username}: {item.prompt}\n\n### Response:"
                system_prompt_prep = f"{item.system_prompt}\n\n### Instruction:\n{username}: \n\n### Response:"
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"{item.system_prompt}\n\n### Instruction:\n{username}: {item.prompt}\n\n### Response:"
                    system_prompt_prep = f"{item.system_prompt}\n\n### Instruction:\n{username}: \n\n### Response:"
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
        
            if llm_template == "Alpaca_No_End_Token":
                end_token = "\n\n### Response:"
                prompt_template = f"{item.system_prompt}\n\n### Instruction:\n{username}: {item.prompt}"
                system_prompt_prep = f"{item.system_prompt}\n\n### Instruction:\n{username}: "
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"{item.system_prompt}\n\n### Instruction:\n{username}: {item.prompt}"
                    system_prompt_prep = f"{item.system_prompt}\n\n### Instruction:\n{username}: "
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
        
        
    # Vicuna Format
            if llm_template == "Vicuna":
                end_token = "ASSISTANT:"
                prompt_template = f"{item.system_prompt} USER: {username}: {item.prompt} ASSISTANT:"
                system_prompt_prep = f"{item.system_prompt} USER: {username}: ASSISTANT:"
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"{item.system_prompt} USER: {username}: {item.prompt} ASSISTANT:"
                    system_prompt_prep = f"{item.system_prompt} USER: {username}: {item.prompt} ASSISTANT:"
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
        
            if llm_template == "Vicuna_No_End_Token":
                end_token = "ASSISTANT:"
                prompt_template = f"{item.system_prompt} USER: {username}: {item.prompt} "
                system_prompt_prep = f"{item.system_prompt} USER: {username}: "
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"{item.system_prompt} USER: {username}: {item.prompt} "
                    system_prompt_prep = f"{item.system_prompt} USER: {username}: {item.prompt} "
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
                    

                    
                    
    # ChatML Format    
            if llm_template == "ChatML":
                end_token = "<|im_end|>\n<|im_start|>assistant"
                prompt_template = f"<|im_start|>system\n{item.system_prompt}<|im_end|>\n<|im_start|>user\n{username}: {item.prompt}<|im_end|>\n<|im_start|>assistant"
                system_prompt_prep = f"<|im_start|>system\n{item.system_prompt}<|im_end|>\n<|im_start|>user\n{username}: <|im_end|>\n<|im_start|>assistant"
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt} [/INST]"
                    system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: [/INST]"
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
                    
            if llm_template == "Llama_2_Chat_No_End_Token":
                end_token = "<|im_end|>\n<|im_start|>assistant"
                prompt_template = f"<|im_start|>system\n{item.system_prompt}<|im_end|>\n<|im_start|>user\n{username}: {item.prompt}"
                system_prompt_prep = f"<|im_start|>system\n{item.system_prompt}<|im_end|>\n<|im_start|>user\n{username}: "
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = input_ids.shape[-1]
                
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    truncation_length = item.truncation_length - len(system_prompt_prep)
                    truncated_input_ids = input_ids[:, -truncation_length:]
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt}"
                    system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: "
                    input_ids = tokenizer.encode(prompt_template)
                    prompt_template_length = input_ids.shape[-1]
        
        
        
            settings = create_generator_settings(item)
            try:
                response = await asyncio.wait_for(asyncio.to_thread(
                    generator.generate_simple, prompt_template, settings, item.max_new_tokens), timeout) 
                    
                response = response.replace(prompt_template, '')

                input_ids = tokenizer.encode(response)
                response_length = input_ids.shape[-1]
            #    response_total_length = response_length - prompt_template_length
                response_total_length = response_length
             #   response = combined_response[prompt_template_length:]
                time_end = time.time()
                time_total = time_end - time_begin
                if prompt_overhang:
                    print(f"Prompt Truncated due to Length. {overhang} Tokens removed from beginning of the prompt.")
                print(f"Prompt token length: {prompt_template_length}. Response Token Length: {response_total_length}.\nResponse generated in {time_total:.2f} seconds, {response_total_length / time_total:.2f} tokens/second.")
            except asyncio.TimeoutError:
                response = "Request timed out"
                logging.error(f"Request {request_id} timed out.")
            result_holder[request_id] = (end_token, response)
        except Exception as e:
            error_trace = traceback.format_exc()
            logging.error(f"Error processing request {request_id}: {error_trace}")
            result_holder[request_id] = str(e)
        finally:
            request_queue.task_done()
            processing_events[request_id].set()

# API endpoints
@app.post("/generate-text/")
async def generate_text(item: Item):
    request_id = str(uuid.uuid4())
    await request_queue.put((request_id, item))
    processing_events[request_id] = asyncio.Event()
    return {"request_id": request_id}

@app.get("/retrieve-text/{request_id}")
async def retrieve_text(request_id: str):
    if request_id in processing_events:
        await processing_events[request_id].wait()
        result_tuple = result_holder.get(request_id)

        # Check if result_tuple is valid
        if result_tuple and isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            end_token, result = result_tuple
            # Use rsplit to split from the end of the string
            if end_token in result:
                result_parts = result.rsplit(end_token, 1)
                if len(result_parts) > 1:
                    result = result_parts[1]  # Take the part after the last occurrence of end_token
                else:
                    result = result_parts[0]  # If end_token is not found, return the entire result
        else:
            result = "No valid result found"

        result_holder.pop(request_id, None)
        del processing_events[request_id]
        return {"generated_text": result}
    else:
        return {"message": "Invalid request ID or processing has not started."}



        
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests())



if __name__ == "__main__":
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)
    host = settings.get('host', "127.0.0.1")
    port = settings.get('port', 8000)

    # Run Uvicorn with asyncio event loop
    import uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "INFO"
    uvicorn.run(app, host=host, port=port, log_config=log_config)

