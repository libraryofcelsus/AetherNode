import os
import traceback
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
import asyncio
import uuid
import logging
import uvicorn
import json
import time

app = FastAPI()

class Item(BaseModel):
    system_prompt: str = "You are a helpful, respectful, and honest assistant."
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    Truncation_Length: int = 4096
    repetition_penalty: float = 1.1
    LLM_Template: str = "Llama_2"

# Initialize the queue
request_queue = asyncio.Queue()
result_holder = {}
processing_events = {}

# Define base path and model directory
base_path = os.getcwd()
model_dir = os.path.join(base_path, "models")

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

with open('settings.json') as settings_file:
    settings = json.load(settings_file)
    
# Load the model and tokenizer
model_name_or_path = settings['model_name_or_path']
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main",
    cache_dir=model_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
    cache_dir=model_dir,
    model_max_length=4096,
    truncation_side='left'
)

def create_pipeline(item):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=item.max_new_tokens,
        do_sample=True,
        temperature=item.temperature,
        top_p=item.top_p,
        top_k=item.top_k,
        repetition_penalty=item.repetition_penalty
    )

async def process_requests():
    while True:
        logging.info("Waiting for the next request...")
        request_id, item = await request_queue.get() 
        logging.info(f"Processing request: {request_id}")
        try:
            time_begin = time.time()
            llm_template = getattr(item, 'LLM_Template', None)
            username = getattr(item, 'Username', None)
            bot_name = getattr(item, 'Bot_Name', None)
            truncation_length = getattr(item, 'Truncation_Length', None)
            with open('settings.json') as settings_file:
                settings = json.load(settings_file)
                
            Use_Injection = settings['Use_Injection_Prompt']
            if llm_template:
                delattr(item, 'LLM_Template')
            if username:
                delattr(item, 'Username')
            if bot_name:
                delattr(item, 'Bot_Name')
            if truncation_length:
                delattr(item, 'Truncation_Length')
            pipe = create_pipeline(item)
            prompt_template = " "   
            prompt_overhang = False   
            if Use_Injection == "True":
                with open('Injection_Prompt.txt', 'r') as file:
                    Injection_Prompt = file.read()
                item.prompt = f"{Injection_Prompt}\n{item.prompt}"
                
            if llm_template == "Llama_2_Chat":
                prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt} [/INST]"
                system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: [/INST]"
                
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = len(input_ids)
                
                if prompt_template_length > truncation_length:
                    overhang = prompt_template_length - truncation_length
                    truncated_input_ids = input_ids[overhang:]  
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                    
            if llm_template == "Llama_2_Chat_No_End_Token":
                prompt_template = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: {item.prompt}"
                system_prompt_prep = f"[INST] <<SYS>>\n{item.system_prompt}\n<</SYS>>\n{username}: "
                
                input_ids = tokenizer.encode(prompt_template)
                prompt_template_length = len(input_ids)
                
                if prompt_template_length > truncation_length:
                    overhang = prompt_template_length - truncation_length
                    truncated_input_ids = input_ids[overhang:]  
                    item.prompt = tokenizer.decode(truncated_input_ids)
                    prompt_overhang = True
                
                
                
                
                
            logging.info("Generating text...")
            response = pipe(prompt_template)[0]['generated_text']
            response_ids = tokenizer.encode(response)
            response_length = len(response_ids)
            response_total_length = response_length - prompt_template_length
         #   response = combined_response[prompt_template_length:]
            time_end = time.time()
            time_total = time_end - time_begin
                
            model_response = response.split(prompt_template)[-1].strip()
            if prompt_overhang:
                print(f"Prompt Truncated due to Length. {overhang} Tokens removed from beginning of the prompt.")
            print(f"Prompt token length: {prompt_template_length}. Response Token Length: {response_total_length}.\nResponse generated in {time_total:.2f} seconds, {response_total_length / time_total:.2f} tokens/second.")
            result_holder[request_id] = model_response
        except Exception as e:
            error_trace = traceback.format_exc()
            logging.error(f"Error processing request {request_id}: {error_trace}")
            result_holder[request_id] = str(e)
        finally:
            request_queue.task_done()
            processing_events[request_id].set()

@app.post("/generate-text/")
async def generate_text(item: Item):
    if len(item.prompt) > 4096:
        item.prompt = item.prompt[-4096:]
    request_id = str(uuid.uuid4())
    # Put the request into the queue
    await request_queue.put((request_id, item))
    # Create an event that will be used to signal when the processing is done
    processing_events[request_id] = asyncio.Event()
    # Return the request ID to the client
    return {"request_id": request_id}

@app.get("/retrieve-text/{request_id}")
async def retrieve_text(request_id: str):
    # Wait for the event to be set, indicating the processing is complete
    if request_id in processing_events:
        await processing_events[request_id].wait()
        # Once processing is done, return the result
        result = result_holder.pop(request_id)
        # Clean up the event
        del processing_events[request_id]
        return {"generated_text": result}
    else:
        # If the request ID is not found, it may be an invalid ID or the processing has not started
        return {"message": "Invalid request ID or processing has not started."}

# Run the process_requests coroutine in the background
asyncio.create_task(process_requests())

if __name__ == "__main__":
    asyncio.create_task(process_requests())
    
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)
    host = settings['host', "127.0.0.1"]
    port = settings['port', "8000"]
    uvicorn.run(app, host=host, port=port)
