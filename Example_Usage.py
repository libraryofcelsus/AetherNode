import aiohttp
import asyncio
import html

async def generate_and_retrieve_text(system_prompt, user_input, host="http://127.0.0.1:8000"):

    data = {
        "LLM_Template": "Llama_2_Chat", 
        # Available Templates are Llama_2_Chat and Llama_2_Chat_No_End_Token
        "Username": "USER", 
        # Set User's Name
        "Bot_Name": "ASSISTANT", 
        # Set Chatbot's Name
        "system_prompt": system_prompt, 
        # System Prompt/Instruction
        "prompt": user_input, 
        # User Input
        "max_new_tokens": 512, 
        # Max New Tokens for Response
        "temperature": 0.7,
        # Temperature is the main factor in controlling the randomness of outputs. It directly effects the probability distribution of the outputs. A lower temperature will be more deterministic, while as a higher temperature will be more creative.
        "top_p": 0.95,
        # Top_p is also known as nucleus sampling. It is an alternative to just using temperature alone in controlling the randomness of the model’s output. This setting will choose from the smallest set of tokens whose cumulative probability exceeds a threshold p. This set of tokens is referred to as the “nucleus”. It is more dynamic than “top_k” and can lead to more diverse and richer outputs, especially in cases where the model is uncertain.
        "top_k": 40,
        # Top_k is another sampling strategy where the model first calculates probabilities for each token in its vocabulary, instead of considering the entire vocabulary as a whole. It restricts the model to only select an output from the k most likely tokens. Top_k is alot more predictable and more simple to use than top_p, but it can make the output too narrow and repetitive.
        "repetition_penalty": 1.10,
        # This setting will help us improve the output by reducing redundant or repetitive content. When the model is generating an output, the repetition penalty will either discourage, or encourage, repeated selection of the same tokens.
        "truncation_length": 4096
        # This setting will set the length of our prompt before it is cut off.
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{host}/generate-text/", json=data) as post_response: # POST request to generate text
            if post_response.status == 200:
                post_data = await post_response.json()
                request_id = post_data['request_id']
                print(f"Request ID: {request_id}")
                max_attempts = 10
                attempts = 0
                delay_seconds = 5  # Delay between each polling attempt
                while attempts < max_attempts:
                    async with session.get(f"{host}/retrieve-text/{request_id}") as get_response:
                        if get_response.status == 200:
                            get_data = await get_response.json()
                            if "generated_text" in get_data:
                                return html.unescape(get_data["generated_text"])
                            await asyncio.sleep(delay_seconds) # If the result is not ready, wait before the next attempt
                        else:
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


