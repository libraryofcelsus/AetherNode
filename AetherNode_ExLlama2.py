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

app = FastAPI()

class Item(BaseModel):
    system_prompt: str = "You are a helpful, respectful, and honest assistant."
    prompt: str
    Username: str = "USER"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.45
    top_k: int = 40
    truncation_length: int = 4096
    repetition_penalty: float = 1.15
    disallowed_words: List[str] = ['[INST]', '[/INST]', '[Inst]', '[/Inst]']
    LLM_Template: str = "Llama_2"

request_queue = asyncio.Queue()
result_holder = {}
processing_events = {}

model_directory = "./models/TheBloke_Llama-2-13B-chat-GPTQ"
config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.warmup()

def create_generator_settings(item):
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = item.temperature
    settings.top_k = item.top_k
    settings.top_p = item.top_p
    settings.token_repetition_penalty = item.repetition_penalty
    disallowed_words = item.disallowed_words
    disallowed_token_ids = [tokenizer.encode(word)[0] for word in disallowed_words]
    for token_id in disallowed_token_ids:
        settings.disallow_tokens(tokenizer, token_id)
    return settings

async def process_requests():
    timeout = 5000  # timeout for each request
    while True:
        logging.info("Waiting for the next request...")
        request_id, item = await request_queue.get()
        logging.info(f"Processing request: {request_id}")
        try:
            time_begin = time.time()
            
            llm_template = getattr(item, 'LLM_Template', None)
            username = getattr(item, 'Username', None)
            bot_name = getattr(item, 'Bot_Name', None)
            end_token = "[/INST]"
            if llm_template:
                delattr(item, 'LLM_Template')
            if username:
                delattr(item, 'Username')
            if bot_name:
                delattr(item, 'Bot_Name')
            prompt_template = " "
            prompt_overhang = False
            if llm_template == "Llama_2_Chat":
                prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt} [/INST]"
                system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: [/INST]"
                prompt_template_length = len(prompt_template)
                if prompt_template_length > item.truncation_length:
                    overhang = prompt_template_length - item.truncation_length
                    item.prompt = item.prompt[overhang:]  
                    prompt_overhang = True
        
            settings = create_generator_settings(item)
            try:
                response = await asyncio.wait_for(asyncio.to_thread(
                    generator.generate_simple, prompt_template, settings, item.max_new_tokens), timeout)
                prompt_length = len(item.prompt)
             #   response = combined_response[prompt_template_length:]
                time_end = time.time()
                time_total = time_end - time_begin
                max_new_tokens = item.max_new_tokens
                if prompt_overhang:
                    print("Prompt Truncated due to Length")
                print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens / time_total:.2f} tokens/second. (Calculation may be wrong, work in progress.)")
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

