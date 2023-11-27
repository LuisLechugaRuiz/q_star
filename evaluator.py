from utils.llm_call import load_prompt, get_response


def generate_evaluation(task, history):
    """
    Use the LLM to evaluate a reasoning trace.

    :param task: The initial task or problem statement.
    :param history: The history of reasoning traces up to this point.
    :return: A value between [..range..] that represents the evaluation of the reasoning trace.
    """
    evaluation_inputs = {
        "task": task,
        "history": history,
    }
    system = load_prompt("evaluation", **evaluation_inputs)
    user = "Remember to return only the numeric evaluation of the reasoning trace."

    response = get_response("gpt-4-turbo", system, user)
    evaluation = response["choices"][0]["message"]["content"]

    return evaluation
