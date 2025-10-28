import click
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


@click.command()
@click.option('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
@click.option('--emit-tool-call/--no-emit-tool-call', type=bool, default=False)
def main(model: str, emit_tool_call: bool):
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        dtype=torch.bfloat16,
        device_map='cuda'
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    conversation = [
        {
            "role": "system",
            "content": "You are smart assistant helping customers. Answer in Russian."
        },
        {
            "role": "user",
            "content": "Какая погода в Москве, Россия?"
        }
    ]
    if emit_tool_call:
        conversation.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"location": "Москва, Россия"}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "content": '{"temperature": 10, "degree": "celsius"}'
                }
            ]
        )

    template_tokens = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tools=[
            {
                "type": "function",
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
        ]
    )

    print('Template tokens:', template_tokens)
    print('=== Template Text Start ===')
    print(tokenizer.decode(template_tokens, skip_special_tokens=False))
    print('=== Template Text End ===')

    result = model_obj.generate(
        input_ids=torch.tensor(template_tokens, dtype=torch.long, device='cuda')[None, :],
        temperature=0.1,
        top_k=10,
        max_length=500
    )
    result = tokenizer.decode(result[0][len(template_tokens):], skip_special_tokens=False)
    print('=== Model Answer ===')
    print(result)
    print('=== Model Answer End ===')


if __name__ == '__main__':
    main()
