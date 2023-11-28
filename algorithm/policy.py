from utils.llm_call import load_prompt, get_response, fill_messages


def fill_policy_prompt(task, history):
    """
    Fill the policy prompt with the task and history.

    :param task: The initial task or problem statement.
    :param history: The history of reasoning traces up to this point.
    :return: The filled policy prompt.
    """
    policy_inputs = {
        "task": task,
        "history": history,
    }
    system = load_prompt("policy", **policy_inputs)
    user = "Remember to return only the next atomic step without any additional information."
    return system, user


def generate_policy_steps(task, history, n=1):
    """
    Use the LLM to generate the next step in the problem-solving process.

    :param task: The initial task or problem statement.
    :param history: The history of reasoning traces up to this point.
    :return: A new reasoning trace as the next step.
    """
    system, user = fill_policy_prompt(task, history)

    responses = get_response("gpt-4-turbo", system, user, n=n)
    steps = [response["message"]["content"] for response in responses["choices"]]

    return steps


def fill_policy_training_data(task, history):
    system, user = fill_policy_prompt(task, history)
    return fill_messages(system, user)
