import argparse
import json
from dotenv import load_dotenv

from q_star.algorithm.q_star import QStarAlgorithm, Result
from q_star.data import get_data_path


def print_with_color(prefix, text, color):
    if color == "red":
        str_color = "\033[91m"
    elif color == "green":
        str_color = "\033[92m"
    elif color == "blue":
        str_color = "\033[94m"
    print(str_color + prefix + "\033[0m " + text)


def execute_q_star(q_star: QStarAlgorithm, training: bool, max_iterations: int):
    iterations = 0

    while iterations < max_iterations:  # TODO: Determine end condition.
        history, result, score, depth = q_star.run()
        if result == Result.IN_PROGRESS:
            print_with_color(f"REASONING with Q-Value: {score} and depth: {depth}:\n", history, "blue")
        elif result == Result.SUCCEEDED:
            print_with_color(
                f"CORRECT SOLUTION with Q-Value: {score} and depth: {depth}:", history, "green"
            )
            if not training:
                # Just finish as we found the correct solution.
                return None
        elif result == Result.FAILED:
            print_with_color(f"WRONG SOLUTION with Q-Value: {score} and depth: {depth}:", history, "red")
        iterations += 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Q Star Algorithm Parameters")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument(
        "--solution", required=False, default=None, help="The solution to the task"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--new_branches",
        type=int,
        required=False,
        default=2,
        help="Number of new branches",
    )
    parser.add_argument(
        "--learning_rate", type=float, required=False, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--training", action="store_true", help="Flag to indicate training mode"
    )
    return parser.parse_args()


def save_data(task_description, solutions, scores):
    base_path = get_data_path() / "data" / task_description
    base_path.mkdir(parents=True, exist_ok=True)

    # Save solutions
    solutions_path = base_path / "solutions.json"
    with solutions_path.open("w") as f:
        json.dump(solutions, f)

    # Save scores
    scores_path = base_path / "scores.json"
    with scores_path.open("w") as f:
        json.dump(scores, f)


def main():
    load_dotenv()  # This loads the variables from .env
    args = parse_arguments()

    q_star = QStarAlgorithm(
        task_description=args.task,
        solution=args.solution,
        new_branches=args.new_branches,
        learning_rate=args.learning_rate,
    )
    execute_q_star(q_star, args.training, args.iterations)

    solutions = q_star.solutions
    scores = q_star.updated_scores
    save_data(args.task, solutions, scores)


if __name__ == "__main__":
    main()
