from enum import IntEnum
from typing import Dict, List

from .policy import (
    generate_policy_steps,
    fill_policy_training_data,
)  # TODO: Relative to package.
from .evaluator import (
    generate_evaluation,
    fill_evaluation_training_data,
)  # TODO: Relative to package.


class Result(IntEnum):
    IN_PROGRESS = 0
    SUCCEEDED = 1
    FAILED = 2


class QStarNode:
    """
    A class to represent a node in the Q* search algorithm.
    """

    def __init__(self, id: int, state=None, parent=None, score=0.0):
        """
        Initialize a new Q* node.

        :param id: The ID of the node.
        :param state: The state represented by this node.
        :param parent: The parent node of this node.
        :param score: The score of this node, representing its effectiveness or success probability.
        """
        self.id = id
        self.state = state
        self.parent = parent
        self.score = score

    def get_path(self) -> List["QStarNode"]:
        """
        Construct a path from the start node to this node.
        """
        node, p = self, []
        while node:
            p.append(node.state)
            node = node.parent
        return p[::-1]  # Return reversed path

    def get_history(self, add_current_state=True):
        """
        Get the history of the node, accumulating the states from the root node to this node using natural language.
        """
        path = self.get_path()
        path.pop(0)  # Remove the root node

        if not add_current_state:
            path.pop(-1)  # Remove the current state

        history = ""
        for index, state in enumerate(path):
            history += f"- {index}: {state}\n"
        return history

    def update(self, new_score, learning_rate):
        """
        Simple update of the node score using the learning rate.
        """
        self.score = self.score + learning_rate * (new_score - self.score)


class QStarAlgorithm:
    """
    A class to implement the Q* algorithm, an adaptation of the A* search algorithm
    for abstract problem domains like theorem proving.
    """

    def __init__(
        self,
        task_description: str,
        new_branches=5,
        learning_rate=0.1,
    ):
        """
        Initialize the Q* algorithm.

        :param task_description: A description of the problem.
        :param new_branches: The number of new branches to generate at each iteration.
        :param learning_rate: The learning rate to use when updating the score of the nodes.
        """
        self.task_description = task_description
        self.new_branches = new_branches
        self.learning_rate = learning_rate

        self.node_counter = 0  # Counter to assign an ID to each node.
        self.branches: List[QStarNode] = [QStarNode(None, None, 0)]  # Initial state with empty history.
        self.solutions: List[
            (str, str)
        ] = (
            []
        )  # Natural language solutions -> Should have format: input ("Task + History") -> output ("Reasoning Trace")
        # TODO: Maybe we should save only the fastest solution?
        self.updated_scores: Dict[
            (int, (str, float))
        ] = (
            []
        )  # Updated scores. Should be a dict as we need to update node on each propagation.

    def sort_branches(self):
        """
        Sort the branches by score.
        """
        self.branches.sort(
            key=lambda n: n.score, reverse=True
        )

    def run(self) -> (str, Result):
        """
        Run the Q* algorithm to find a path from the start state to a potential solution.

        :return: A tuple with the current history and a Result object with the current state.
        """
        current_node = self.branches.pop(0)  # Node with the highest score
        history = current_node.get_history()
        result = Result.IN_PROGRESS

        # Generate successors and evaluate them
        for successor in self.get_successors(history):
            evaluation = self.evaluate_state(successor)

            successor_node = QStarNode(
                self.node_counter,
                successor,
                current_node,
                evaluation,  # Save as score only the new evaluation which is based on the full history. TBD...
            )
            self.branches.append(successor_node)
            self.node_counter += 1

            if self.is_goal_state(evaluation):
                success = self.run_external_validation(successor_node.get_history())
                self.backpropagate(successor_node, int(success))  # 1 if the solution is correct, 0 otherwise.
                if success:
                    result = Result.SUCCEEDED
                    self.save_success_path(successor_node)
                else:
                    result = Result.FAILED
        # Sort branches by score
        self.sort_branches()

        return self.branches[0].get_history(), result

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
        return generate_evaluation(task=self.task_description, history=history)

    def is_goal_state(self, evaluation: int):
        """
        Determine if the current state is a potential goal state.
        """
        # This method should determine if the current state represents a potential solution.
        # TODO: How to determine if the current state is a potential solution? Potential ideas:
        #       - 1: Ask the evaluator to return a value of -1
        #       - 2: Ask the policy to provide both the next step and a flag with is_goal_state
        return evaluation == -1

    def backpropagate(self, node: QStarNode, reward: int):
        """
        Backpropagate the reward to the parent nodes.
        """
        node = self
        while node:
            node.update(reward, self.learning_rate)
            evaluation_input = fill_evaluation_training_data(
                task=self.task_description,
                history=node.get_history(exclude_current_state=True),
                new_step=node.state,
            )
            self.updated_scores[node.id] = (evaluation_input, node.score)

            node = node.parent

    def save_success_path(self, node: QStarNode):
        """
        Save the current path as solution.
        """
        while node:
            previous_history = node.get_history(add_current_state=False)
            policy_training_input = fill_policy_training_data(
                task=self.task_description, history=previous_history
            )
            self.solutions.append(policy_training_input, node.state)

            node = node.parent

    def run_external_validation(self, history: str):
        """
        Run an external validation to ensure that the solution is correct.
        """
        pass
