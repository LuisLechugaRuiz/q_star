from q_star.utils.llm_call import load_prompt, get_response, fill_messages


def fill_evaluation_prompt(task, history, min_score, max_score):
    """
    Fill the evaluation prompt with the task, history, and new step.

    :param task: The initial task or problem statement.
    :param history: The history of reasoning traces up to this point.
    :param min_score: The minimum score for the evaluation.
    :param max_score: The maximum score for the evaluation.
    :return: The filled policy prompt.
    """
    evaluation_inputs = {
        "task": task,
        "history": history,
        "min_score": min_score,
        "max_score": max_score,
    }
    system = load_prompt("evaluation", **evaluation_inputs)
    user = "Remember to return only the numeric evaluation of the current history."
    return system, user


def generate_evaluation(task, history, min_score=0.0, max_score=1.0):
    """
    Use the LLM to evaluate a reasoning trace.

    :param task: The initial task or problem statement.
    :param history: The full history of reasoning traces up to this point.
    :return: A value between [..range..] that represents the evaluation of the reasoning trace.
    """
    system, user = fill_evaluation_prompt(task, history, min_score, max_score)

    response = get_response(system, user)
    evaluation = response["choices"][0]["message"]["content"]

    return evaluation


def fill_evaluation_training_data(task, history, min_score=0.0, max_score=1.0):
    system, user = fill_evaluation_prompt(task, history, min_score, max_score)
    return fill_messages(system, user)
