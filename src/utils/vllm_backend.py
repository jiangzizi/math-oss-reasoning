from openai import OpenAI


def get_response(
    user_prompt: str,
    model_name: str = "gpt-oss-120b",
    port: int = 8000,
    logout: bool = True,
    reasoning_effort: str = "medium",
):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )

    response = client.responses.create(
        model=model_name,
        instructions="You are a helfpul assistant.",
        input=user_prompt,
        reasoning={"effort": reasoning_effort},
    )

    # print(response)
    final_answer = response.output_text
    for item in response.output:
        if item.type == 'reasoning':
            for content_block in item.content:
                if content_block.type == 'reasoning_text':
                    reasoning_text = content_block.text
                    # print(reasoning_text)
                    break
    full_answer = f"<think>{reasoning_text}</think>{final_answer}"
    return full_answer


if __name__ == "__main__":
    prompt = (
        "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A')."
        "Which of the following is a primary benefit of using no-till farming practices in agriculture?\n"
        "A: Increased soil erosion\n"
        "B: Reduced fuel consumption\n"
        "C: Enhanced weed growth\n"
        "D: Increased need for irrigation\n"
        "E: Improved soil structure\n"
        "F: Decreased soil organic matter\n"
        "G: Reduced crop yield\n"
        "H: Increased use of herbicides\n"
        "I: Enhanced water infiltration\n"
        "J: Decreased biodiversity"
    )
    # prompt = "1+1 = ?"

    response = get_response(prompt, port=1145, reasoning_effort="high")
    print(response)