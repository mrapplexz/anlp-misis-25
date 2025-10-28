import click
from openai import OpenAI
from pydantic import BaseModel


import datetime as dt


class MeetingInfo(BaseModel):
    name: str
    date_time: dt.datetime
    peers: list[str]


@click.command()
@click.option('--base-url', type=str, default='http://localhost:8000/v1')
def main(base_url: str):
    client = OpenAI(
        base_url=base_url,
        api_key="123",
    )

    completion = client.chat.completions.parse(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[
            {
                "role": "system",
                "content": 'Extract relevant information from user query in JSON format like {"name": "meeting name", "date_time": "iso formatted meeting date and time", "peers": ["list", "of", "people", "invited"]}'
            },
            {
                "role": "user",
                "content": "Create a meeting named with a short generated hokku about birthdays at 10 p m for 25th september 2025. Invite Max and any of his friends (Pete, Max, Oleg)"
            }
        ],
        response_format=MeetingInfo
    )

    parsed_object = completion.choices[0].message.parsed

    print(parsed_object)



if __name__ == '__main__':
    main()
