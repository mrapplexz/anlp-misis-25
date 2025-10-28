import json

import click
from openai import OpenAI
from pydantic import BaseModel

import datetime as dt

_TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_capital_city",
            "description": "Get capital city",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country to get capital city of"
                    }
                },
                "required": [
                    "country"
                ],
                "additionalProperties": False
            }
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    print(f'== Calling tool {name} with arguments {arguments}')
    if name == 'get_weather':
        return '{"temperature": 15.3, "degrees": "celsius"}'
    elif name == 'get_capital_city':
        return 'Moscow'
    raise ValueError(f'Unknown tool {name}')


@click.command()
@click.option('--base-url', type=str, default='http://localhost:8000/v1')
def main(base_url: str):
    client = OpenAI(
        base_url=base_url,
        api_key="123",
    )

    conversation = [
        {
            "role": "system",
            "content": 'You are a smart agent, you should solve a task that user sets for you.'
        },
        {
            "role": "user",
            "content": "How are you?"
        }
    ]

    while True:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=conversation,
            tools=_TOOLS
        )

        choice = completion.choices[0].message

        if choice.tool_calls:
            conversation.append(
                {
                    "role": "assistant",
                    "tool_calls": choice.tool_calls
                }
            )
            for call in choice.tool_calls:
                func = call.function
                call_result = execute_tool(name=func.name, arguments=json.loads(func.arguments))
                conversation.append(
                    {
                        "role": "tool",
                        "content": call_result,
                        "tool_call_id": call.id
                    }
                )
        else:
            print('== Final answer')
            print(choice.content)
            break


if __name__ == '__main__':
    main()
