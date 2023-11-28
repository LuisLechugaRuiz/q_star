from pathlib import Path
import os

from jinja2 import Environment, FileSystemLoader
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import completion


def load_prompt(template: str, **kwargs) -> str:
    """
    Load and populate the specified template.

    Args:
        template (str): The name of the template to load.
        **kwargs: The arguments to populate the template with.

    Returns:
        str: The populated template.
    """
    try:
        prompts_dir = os.path.abspath(
            os.path.join(Path(__file__).parent, "prompts")
        )
        prompts_env = Environment(loader=FileSystemLoader(prompts_dir))
        template = prompts_env.get_template(f"{template}.j2")
        return template.render(**kwargs)
    except Exception as e:
        print(f"Error loading or rendering template: {e}")
        raise


def fill_messages(system: str, user: str):
    return [
        {"role": "user", "content": user},
        {"role": "system", "content": system}
    ]


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_response(model: str, system: str, user: str, temperature: float = 0.7, n: int = 1) -> str:
    messages = fill_messages(system, user)
    args = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": n,
    }
    try:
        resp = await completion(**args)
        return resp
    except Exception as e:
        # TODO: LOG
        print("Exception: ", e)
        raise
