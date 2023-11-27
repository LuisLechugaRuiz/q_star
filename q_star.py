from typing import List

from .policy import generate_policy_steps # TODO: Relative to package.
from .evaluator import generate_evaluation # TODO: Relative to package.


class QStarNode:
    """
    A class to represent a node in the Q* search algorithm.
    """

    def __init__(self, state=None, parent=None, cost=0.0):
        """
        Initialize a new Q* node.

        :param state: The state represented by this node.
        :param parent: The parent node of this node.
        :param cost: The accumulated cost of getting to this node, as determined by the Q* evaluator.
        """
        self.state = state
        self.parent = parent
        self.cost = cost

    def get_path(self) -> List["QStarNode"]:
        """
        Construct a path from the start node to this node.
        """
        node, p = self, []
        while node:
            p.append(node.state)
            node = node.parent
        return p[::-1]  # Return reversed path

    def get_history(self):
        path = self.get_path()
        path.pop(0)  # Remove the root node
        history = ""
        for index, state in enumerate(path):
            history += f"- {index}: {state}\n"
        return history


class QStarAlgorithm:
    """
    A class to implement the Q* algorithm, an adaptation of the A* search algorithm 
    for abstract problem domains like theorem proving.
    """

    def __init__(self, task_description, evaluator_func, max_iterations=1000, new_branches=2):
        """
        Initialize the Q* algorithm.

        :param task_description: A description of the problem.
        :param evaluator_func: The Q model, the evaluation function of the fine-tuned LLM evaluator.
        :param max_iterations: The maximum number of iterations before stopping the algorithm.
        :param new_branches: The number of new branches to generate at each iteration.
        """
        self.task_description = task_description
        self.evaluator_func = evaluator_func
        self.max_iterations = max_iterations
        self.new_branches = new_branches

        self.branches: List[QStarNode] = []  # Nodes to be evaluated

    def add_branch(self, node: QStarNode):
        """
        Add a new node to the existing branches.
        """
        self.branches.append(node)
        self.branches.sort(key=lambda n: n.cost)

    def run(self):
        """
        Run the Q* algorithm to find a path from the start state to a potential solution.
        """
        iterations = 0
        start_node = QStarNode(None, None, 0)  # Initial state with empty history.
        self.add_branch(start_node)

        while iterations < self.max_iterations:
            current_node = self.branches.pop(0)  # Node with the lowest cost
            history = current_node.get_history()

            # Generate successors and evaluate them
            for successor in self.get_successors(history):
                evaluation = self.evaluate_state(successor)

                # Should we stop the algorithm when the model thinks he has reached the solution? Or should we continue? 
                # Maybe if we have a real-time testing that ensures that the solutions works...
                if self.is_goal_state(evaluation):
                    return current_node.get_history()  # Potential solution reached

                successor_node = QStarNode(
                    successor,
                    current_node,
                    current_node.cost + evaluation # Should we accumulate costs or just save the cost of the current state?
                )
                self.add_branch(successor_node)

            iterations += 1

        return None  # No path found

    def get_successors(self, history):
        """
        Call the LLM generator to obtain the next traces.

        This function relies on the variations provided by OpenAI's API, we can modify it to control it manually: increasing the temperature, modifying the seed.
        """
        return generate_policy_steps(task=self.task_description, history=history, n=self.new_branches)

    def evaluate_state(self, history):
        """
        Call the LLM evaluator to obtain the evaluation of a state.
        """
        # Things to address:
        # - How to evaluate if this is the final solution from the LLM? Maybe the evaluator does it? When he add a value of 0?
        return generate_evaluation(task=self.task_description, history=history)

    def is_goal_state(self, evaluation):
        """
        Determine if the current state is a potential goal state.
        """
        # This method should determine if the current state represents a potential solution.
        # TODO: How to determine if the current state is a potential solution? First idea: Ask the evaluator to return a value of 0.
        return evaluation == 0
