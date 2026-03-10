from ggp_statemachine import GameStateMachine
from ggp_agent import RandomAgent, MCTSAgent


class GameRunner:
    def __init__(self, game_file, agents):
        """
        Args:
            game_file: path to .kif rule file
            agents: dict role_name -> Agent instance
        """
        self.game = GameStateMachine(game_file)
        self.game_file = game_file
        self.agents = agents
        self.history = []
        self.roles = [str(role) for role in self.game.get_roles()]
        self._validate_agents()

    def _validate_agents(self):
        missing = [role for role in self.roles if role not in self.agents]
        if missing:
            raise ValueError(f"Missing agent(s) for role(s): {', '.join(missing)}")

    def run_match(self, verbose=True):
        print(f"=== Match Start: {self.game_name()} ===")
        print(f"Players: {', '.join([str(self.agents[r]) for r in self.roles])}")

        current_state = self.game.get_initial_state()
        step = 0

        while not self.game.is_terminal(current_state):
            step += 1
            if verbose:
                print(f"\n--- Step {step} ---")
                self._print_board_state(current_state)

            moves = {}
            for role in self.roles:
                agent = self.agents.get(role)
                if not agent:
                    raise ValueError(f"No agent assigned for role: {role}")

                move = agent.select_move(self.game, current_state)
                moves[role] = move
                if verbose:
                    print(f"{agent.name} selects: {move}")

            self.history.append({"state": current_state, "moves": moves})
            current_state = self.game.get_next_state(current_state, moves)

        print("\n=== Game Over ===")
        self._print_board_state(current_state)
        scores = self.get_scores(current_state)

        if len(self.roles) == 1:
            role = self.roles[0]
            print(f"Final Score: {role} = {scores.get(role, 0)}")
        else:
            print("Final Scores:", scores)
            max_score = max(scores.values())
            winners = [role for role, score in scores.items() if score == max_score]
            if len(winners) == 1:
                print(f"Winner: {winners[0]} (Score: {max_score})")
            else:
                print(f"Draw: {', '.join(winners)} (Score: {max_score})")

        return scores

    def get_scores(self, state):
        return {role: self.game.get_goal(state, role) for role in self.roles}

    def game_name(self):
        return self.game_file.replace(".kif", "")

    def _print_board_state(self, state):
        facts = [str(f) for f in state]
        cells = [fact for fact in facts if fact.startswith("cell")]
        if cells:
            print(f"Board State: {cells}")
        else:
            print(f"State: {facts}")


if __name__ == "__main__":
    game_file = "games/ticTacToe.kif"
    agents = {
        "xplayer": MCTSAgent("Bot_MCT_1", "xplayer"),
        "oplayer": MCTSAgent("Bot_MCT_2", "oplayer"),
    }

    # game_file = "games/bonaparte.kif"
    # agents = {
    #     "france": MCTSAgent("Bot_MCT_1", "france"),
    #     "germany": MCTSAgent("Bot_MCT_2", "germany"),
    #     "russia": MCTSAgent("Bot_MCT_3", "russia"),
    # }

    # game_file = "games/mediocrity.kif"
    # agents = {
    #     "a": MCTSAgent("Bot_MCT_1", "a"),
    #     "b": MCTSAgent("Bot_MCT_2", "b"),
    #     "c": MCTSAgent("Bot_MCT_3", "c"),
    # }

    # game_file = "games/connectFour.kif"
    # agents = {
    #     "red": MCTSAgent("Bot_MCT_1", "red"),
    #     "black": MCTSAgent("Bot_MCT_2", "black"),
    # }

    # game_file = "games/maze.kif"
    # agents = {
    #     "robot": MCTSAgent("Bot_MCT_1", "robot"),
    # }

    runner = GameRunner(game_file, agents)
    final_scores = runner.run_match()
