from openai import OpenAI
 
 
def get_response(user_prompt, model_name = "gpt-oss-120b/", port = 8000, logout: bool = True, reasoning_effort: str = "low"):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY"
    )

    stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.5,
            stream=True,
            extra_body={"reasoning_effort": reasoning_effort}
        )

    output = ""

    for chunk in stream:
        content = None
        
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
        elif hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
            content = chunk.choices[0].delta.reasoning
            
        if content is not None:
            if logout:
                print(content, end='', flush=True)
            output += content

    print()
    return output

if __name__ == "__main__":
    prompt = "1+2 = ?"
    response = get_response(prompt, port = 1145)