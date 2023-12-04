from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import re

from q_star.algorithm.policy import (
    generate_policy_steps,
    fill_policy_training_data,
)
from q_star.algorithm.evaluator import (
    generate_evaluation,
    fill_evaluation_training_data,
)

import q_star.utils.grading.grader as grader


class Result(IntEnum):
    IN_PROGRESS = 0
    SUCCEEDED = 1
    FAILED = 2


class QStarNode:
    """
    A class to represent a node in the Q* search algorithm.
    """

    def __init__(self, uid: int, state=None, parent=None, score=0.0, depth=0):
        """
        Initialize a new Q* node.

        :param uid: The UID of the node.
        :param state: The state represented by this node.
        :param parent: The parent node of this node.
        :param score: The score of this node, representing its effectiveness or success probability.
        """
        self.uid = uid
        self.state = state
        self.parent = parent
        self.score = score
        self.depth = depth

    def get_path(self) -> List["QStarNode"]:
        """
        Construct a path from the start node to this node.
        """
        node, p = self, []
        while node:
            p.append(node)
            node = node.parent
        reversed_path = p[::-1]
        reversed_path.pop(0)  # Remove the root node
        return reversed_path

    def get_history(self, add_current_state=True):
        """
        Get the history of the node, accumulating the states from the root node to this node using natural language.
        """
        path = self.get_path()

        if len(path) > 0 and not add_current_state:
            path.pop(-1)  # Remove the current state

        history = ""
        for node in path:
            history += f"{node.state}\n"
        return history

    def update(self, new_score, learning_rate):
        """
        Simple update of the node score using the learning rate.
        """
        print("DEBUG - Initial score: ", self.score)
        self.score = self.score + learning_rate * (new_score - self.score)
        print("DEBUG - New score: ", new_score)


class QStarAlgorithm:
    """
    A class to implement the Q* algorithm, an adaptation of the A* search algorithm
    for abstract problem domains like theorem proving.
    """

    def __init__(
        self,
        task_description: str,
        solution: Optional[str] = None,
        new_branches=2,
        learning_rate=0.1,
    ):
        """
        Initialize the Q* algorithm.

        :param task_description: A description of the problem.
        :param new_branches: The number of new branches to generate at each iteration.
        :param learning_rate: The learning rate to use when updating the score of the nodes.
        """
        self.task_description = task_description
        self.solution = solution
        self.new_branches = new_branches
        self.learning_rate = learning_rate

        self.node_counter = 0
        self.depth = 0
        self.branches: List[QStarNode] = [
            QStarNode(self.node_counter, None, None, 0, 0)
        ]  # Initial state with empty history.

        self.solutions: List[Tuple[str, str]] = []
        self.updated_scores: Dict[int, Tuple[str, float]] = {}

    def sort_branches(self):
        """
        Sort the branches by score, and in case of a tie, by depth.
        """
        self.branches.sort(key=lambda n: (n.score, n.depth), reverse=True)

    def run(self) -> (str, Result):
        """
        Run the Q* algorithm to find a path from the start state to a potential solution.

        :return: A tuple with the current history and a Result object with the current state.
        """
        current_node = self.branches.pop(0)  # Node with the highest score
        history = current_node.get_history()
        result = Result.IN_PROGRESS
        answer = None
        self.depth += 1

        # Generate successors and evaluate them
        for successor in self.get_successors(history):
            updated_history = history + successor + "\n"
            evaluation = self.evaluate_state(updated_history)
            if evaluation is None:
                evaluation = 0.0  # Error extracting evaluation from model response...

            successor_node = QStarNode(
                self.node_counter,
                successor,
                current_node,
                evaluation,
                self.depth,
            )
            self.branches.append(successor_node)
            self.node_counter += 1

            final_answer = self.extract_answer(successor)
            if final_answer:
                answer = final_answer
                if self.solution is None:
                    success = True  # If not solution provided, we set to success to True to just return the first answer.
                else:
                    success = grader.grade_answer(final_answer, self.solution)
                    self.backpropagate(
                        successor_node, int(success)
                    )  # 1 if the solution is correct, 0 otherwise.
                if success:
                    result = Result.SUCCEEDED
                    self.save_success_path(successor_node)
                else:
                    result = Result.FAILED
        # Sort branches by score
        self.sort_branches()

        if answer is not None:
            return answer, result, self.branches[0].score, self.branches[0].depth

        return self.branches[0].get_history(), result, self.branches[0].score, self.branches[0].depth

    def get_successors(self, history):
        """
        Call the LLM generator to obtain the next traces.

        This function relies on the variations provided by OpenAI's API, we can modify it to control it manually: increasing the temperature, modifying the seed.
        """
        return generate_policy_steps(
            task=self.task_description, history=history, n=self.new_branches
        )

    def evaluate_state(self, history):
        """
        Call the LLM evaluator to obtain the evaluation of a state.
        """
        evaluation_str = generate_evaluation(
            task=self.task_description, history=history
        )
        match = re.search(r"[-+]?\d*\.\d+|\d+", evaluation_str)
        evaluation_float = float(match.group()) if match else None
        return evaluation_float

    def backpropagate(self, current_node: QStarNode, reward: int):
        """
        Backpropagate the reward to the parent nodes.
        """
        print(f"Backpropagating reward: {reward}")
        path = current_node.get_path()
        path_length = current_node.depth
        for node in path:
            new_score = reward / path_length * node.depth
            node.update(new_score, self.learning_rate)
            evaluation_input = fill_evaluation_training_data(
                task=self.task_description,
                history=node.get_history(),
            )
            self.updated_scores[node.uid] = (evaluation_input, node.score)

    def save_success_path(self, last_node: QStarNode):
        """
        Save the current path as solution.
        """
        path = last_node.get_path()

        for node in path:
            previous_history = node.get_history(add_current_state=False)
            policy_training_input = fill_policy_training_data(
                task=self.task_description, history=previous_history
            )
            self.solutions.append((policy_training_input, node.state))

    def extract_answer(self, text):
        """
        Extracts and returns the content after 'Answer: ' in the provided text.
        If 'Answer: ' is not found, returns an empty string.
        """
        keyword = "Answer: "
        start_index = text.find(keyword)

        if start_index == -1:
            return None

        # Extract content after the keyword
        start_index += len(keyword)
        answer = text[start_index:].strip()

        return answer
