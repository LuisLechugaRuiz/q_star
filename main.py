from pathlib import Path

from algorithm.q_star import QStarAlgorithm, Result


def print_with_color(text, color):
    if color == "red":
        str_color = "\033[91m"
    elif color == "green":
        str_color = "\033[92m"
    elif color == "blue":
        str_color = "\033[94m"
    print(str_color + text + "\033[0m")


def execute_q_star(q_star: QStarAlgorithm, training: bool):    
    iterations = 0

    while iterations < max_iterations:  # TODO: Determine end condition.
        history, result = q_star.run()
        if result == Result.IN_PROGRESS:
            print_with_color("REASONING:", "blue")
        elif result == Result.SUCCEEDED:
            print_with_color("CORRECT SOLUTION:", "green")
            if not training:
                # Just finish as we found the correct solution.
                return None
        elif result == Result.FAILED:
            print_with_color("WRONG SOLUTION:", "red")
        print(history)
        iterations += 1


def main():
    # parse arguments -> task, number of iterations, number of new branches, training..
    # max_iterations=1000,
    q_star = QStarAlgorithm(task_description=task, new_branches=new_branches, learning_rate=learning_rate)
    execute_q_star(q_star, training)

    if training:
        # save to file.
        data_path = Path(__file__).parent / "data"
        solutions = q_star.solutions # Saving all solutions, maybe we should filter to get the shortest one.
        scores = q_star.updated_scores
        # TODO: save as data_path / task_name / solutions - scores and be able to fetch later all of them with glob when we have certain tokens.

if __name__ == "__main__":
    main() # TODO: Send the arguments.