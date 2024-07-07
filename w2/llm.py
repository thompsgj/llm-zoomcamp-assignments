from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)


def llm(prompt):
    response = client.chat.completions.create(
        model="gemma:2b",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content, response


# result, response = llm("10 * 10")
# print(result)

result, response = llm("What's the formula for energy?")
print(result)
print(response)
