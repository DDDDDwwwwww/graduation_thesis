import time

from ggp_statemachine import GameStateMachine
from ggp_agent import RandomAgent, HeuristicMCTSAgent, PureMCTAgent


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

    def run_match(self, verbose=True, move_time_limit=1.0, perf_log=False):
        print(f"=== Match Start: {self.game_name()} ===")
        print(f"Players: {', '.join([str(self.agents[r]) for r in self.roles])}")

        # Keep cache and metrics scoped to current match.
        self.game.clear_caches()
        self.game.reset_perf_stats()

        current_state = self.game.get_initial_state()
        step = 0

        while not self.game.is_terminal(current_state):
            step += 1
            if verbose:
                print(f"\n--- Step {step} ---")
                self._print_board_state(current_state)

            moves = {}
            role_perf_logs = []

            for role in self.roles:
                agent = self.agents.get(role)
                if not agent:
                    raise ValueError(f"No agent assigned for role: {role}")

                before_stats = self.game.get_perf_stats() if perf_log else None
                start = time.time()
                move = agent.select_move(self.game, current_state, time_limit=move_time_limit)
                elapsed = time.time() - start
                after_stats = self.game.get_perf_stats() if perf_log else None

                moves[role] = move
                if verbose:
                    print(f"{agent.name} selects: {move} (decision_sec={elapsed:.3f})")

                if perf_log and before_stats and after_stats:
                    role_perf_logs.append(
                        {
                            "agent": agent.name,
                            "role": role,
                            "decision_sec": elapsed,
                            "legal_calls": after_stats["legal_calls"] - before_stats["legal_calls"],
                            "legal_cache_hits": after_stats["legal_cache_hits"] - before_stats["legal_cache_hits"],
                            "next_calls": after_stats["next_calls"] - before_stats["next_calls"],
                            "next_cache_hits": after_stats["next_cache_hits"] - before_stats["next_cache_hits"],
                        }
                    )

            self.history.append({"state": current_state, "moves": moves})
            current_state = self.game.get_next_state(current_state, moves)

            if perf_log:
                print("[Perf][Step]")
                for entry in role_perf_logs:
                    print(
                        f"  {entry['agent']}({entry['role']}): "
                        f"decision_sec={entry['decision_sec']:.3f}, "
                        f"legal={entry['legal_calls']} (hit={entry['legal_cache_hits']}), "
                        f"next={entry['next_calls']} (hit={entry['next_cache_hits']})"
                    )
                total = self.game.get_perf_stats()
                print(
                    "[Perf][Total] "
                    f"legal={total['legal_calls']} (hit={total['legal_cache_hits']}), "
                    f"next={total['next_calls']} (hit={total['next_cache_hits']})"
                )

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
    # game_file = "games/ticTacToe.kif"
    # agents = {
    #     "xplayer": HeuristicMCTSAgent("Bot_MCT_1", "xplayer"),
    #     "oplayer": HeuristicMCTSAgent("Bot_MCT_2", "oplayer"),
    # }
    # agents = {
    #     "xplayer": PureMCTAgent("Bot_PureMCT_1", "xplayer"),
    #     "oplayer": PureMCTAgent("Bot_PureMCT_2", "oplayer"),
    # }

    # game_file = "games/bonaparte.kif"
    # agents = {
    #     "france": RandomAgent("Bot_Random_1", "france"),
    #     "germany": HeuristicMCTSAgent(
    #         "Bot_MCTS_2",
    #         "germany",
    #         iterations=8,
    #         rollout_depth_limit=8,
    #         exploration_constant=15.0,
    #         fallback_legal_threshold=150,
    #     ),
    #     "russia": HeuristicMCTSAgent(
    #         "Bot_MCTS_3",
    #         "russia",
    #         iterations=8,
    #         rollout_depth_limit=8,
    #         exploration_constant=15.0,
    #         fallback_legal_threshold=150,
    #     ),
    # }

    # game_file = "games/mediocrity.kif"
    # agents = {
    #     "a": HeuristicMCTSAgent("Bot_MCT_1", "a"),
    #     "b": HeuristicMCTSAgent("Bot_MCT_2", "b"),
    #     "c": HeuristicMCTSAgent("Bot_MCT_3", "c"),
    # }

    # game_file = "games/breakthrough.kif"
    # agents = {
    #     "white": PureMCTAgent("Bot_MCT_1", "white"),
    #     "black": HeuristicMCTSAgent("Bot_MCT_2", "black")
    # }

    game_file = "games/reversi.kif"
    agents = {
        "white": PureMCTAgent("Bot_MCT_1", "white"),
        "black": RandomAgent("Bot_MCT_2", "black")
    }

    # game_file = "games/connectFour.kif"
    # agents = {
    #     "red": HeuristicMCTSAgent("Bot_MCT_1", "red"),
    #     "black": HeuristicMCTSAgent("Bot_MCT_2", "black"),
    # }

    # game_file = "games/maze.kif"
    # agents = {
    #     "robot": HeuristicMCTSAgent("Bot_MCT_1", "robot"),
    # }

    runner = GameRunner(game_file, agents)
    final_scores = runner.run_match(verbose=True, move_time_limit=1.0, perf_log=True)
