import os
import traceback
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
import asyncio
import uuid
import logging
import uvicorn

app = FastAPI()

class Item(BaseModel):
    system_prompt: str = "You are a helpful, respectful, and honest assistant."
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
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

# Load the model and tokenizer
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
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
    cache_dir=model_dir
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
            llm_template = getattr(item, 'LLM_Template', None)
            if llm_template:
                delattr(item, 'LLM_Template')
            pipe = create_pipeline(item)
            prompt_template = " "
            if llm_template == "Llama_2":
                prompt_template = f"[INST] <<SYS>> {item.system_prompt} <</SYS>> {item.prompt} [/INST]"
            logging.info("Generating text...")
            response = pipe(prompt_template)[0]['generated_text']
            # Extract only the model's response
            model_response = response.split(prompt_template)[-1].strip()
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
