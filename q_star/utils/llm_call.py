import pkg_resources
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
        package_path = pkg_resources.resource_filename("q_star", "")
        prompts_dir = os.path.join(os.path.abspath(package_path), "prompts")
        prompts_env = Environment(loader=FileSystemLoader(prompts_dir))
        template = prompts_env.get_template(f"{template}.j2")
        return template.render(**kwargs)
    except Exception as e:
        print(f"Error loading or rendering template: {e}")
        raise


def fill_messages(system: str, user: str):
    return [{"role": "user", "content": user}, {"role": "system", "content": system}]


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_response(
    system: str, user: str, temperature: float = 0.7, n: int = 1
) -> str:
    messages = fill_messages(system, user)
    # print("Generating response...")
    args = {
        "model": os.environ.get("GPT_MODEL", "gpt-4-1106-preview"),
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    try:
        resp = completion(**args)
        # print("DEBUG - RESPONSE:", resp)
        return resp
    except Exception as e:
        # TODO: LOG
        print("Exception: ", e)
        raise
