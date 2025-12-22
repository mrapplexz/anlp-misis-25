import click
import openai


_GRAMMAR = r'''
json: %json {"type": "object", "properties": {"result": {"type": "string", "enum": ["Toxic", "Untoxic"]}}}

PREFIX: "Here is your answer: ```json\n"
POSTFIX: "\n```"

start: PREFIX json POSTFIX
'''


@click.command()
def main():
    client = openai.Client(base_url=f"http://127.0.0.1:30000/v1", api_key="None")
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[
            {
                "role": "system",
                "content": "You are text toxicity classifier. Please respond with \'Toxic\' or \'Untoxic\' label only (with JSON format {\"result\": ...})."
            },
            {
                "role": "user",
                "content": "You are a bastard",
            },
        ],
        temperature=0,
        max_tokens=32,
        extra_body={"ebnf": _GRAMMAR},
    )
    print(response.choices[0].message)

if __name__ =='__main__':
    main()
