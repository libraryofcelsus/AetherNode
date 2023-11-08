import aiohttp
import asyncio
import html

async def generate_and_retrieve_text(system_prompt, user_input, host="http://127.0.0.1:8000"):

    data = {
        "LLM_Template": "Llama_2",
        "system_prompt": system_prompt,
        "prompt": user_input,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1
    }


    async with aiohttp.ClientSession() as session:
        # POST request to generate text
        async with session.post(f"{host}/generate-text/", json=data) as post_response:
            if post_response.status == 200:
                post_data = await post_response.json()
                request_id = post_data['request_id']
                print(f"Request ID: {request_id}")
                max_attempts = 10
                attempts = 0
                delay_seconds = 5  # Delay between each polling attempt
                # Poll the GET endpoint to retrieve the result
                while attempts < max_attempts:
                    async with session.get(f"{host}/retrieve-text/{request_id}") as get_response:
                        if get_response.status == 200:
                            get_data = await get_response.json()
                            if "generated_text" in get_data:
                                # If the generated text is ready, return it
                                return html.unescape(get_data["generated_text"])
                            # If the result is not ready, wait before the next attempt
                            await asyncio.sleep(delay_seconds)
                        else:
                            # If there's an error retrieving the result, return the error
                            return "Failed to retrieve the result."
                    attempts += 1
                return "Max polling attempts reached. Please try again later."
            else:
                return f"Failed to submit the prompt: {post_response.status}"

if __name__ == "__main__":
    while True:
        system_prompt = "You are a helpful, respectful, and honest assistant."
        user_input = input("ENTER MESSAGE: ")
        result = asyncio.run(generate_and_retrieve_text(system_prompt, user_input))
        print(result)


