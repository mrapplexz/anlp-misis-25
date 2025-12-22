import click
import openai

_GRAMMAR = r'''
start: TEXT | fun_call
TEXT: /[^{](.|\n)*/
fun_call: <tool_call> "\n" json_body "\n" </tool_call>
json_body: %json {
  "type": "object",
  "properties": {
    "name": { "type": "string", "enum": ["get_weather", "send_money"] },
    "arguments": {
      "type": "string"
    }
  },
  "required": ["name", "arguments"]
}
'''


@click.command()
def main():
    client = openai.Client(base_url=f"http://127.0.0.1:30000/v1", api_key="None")
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[
            {
                "role": "system",
                "content": "Answer user inquiries and optionally call tools."
            },
            {
                "role": "user",
                "content": "What's the weather in London?",
            },
        ],
        temperature=0,
        max_tokens=32,
        extra_body={"ebnf": _GRAMMAR},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia"
                            }
                        },
                        "required": [
                            "location"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        ]
    )
    print(response.choices[0].message)


if __name__ == '__main__':
    main()
