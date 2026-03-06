import time
from ggp_statemachine import GameStateMachine
from ggp_agent import RandomAgent, MCTSAgent 

class GameRunner:
    def __init__(self, game_file, agents):
        """
        Args:
            game_file: .kif 规则文件路径
            agents: 一个字典，映射角色名到 Agent 实例。
                    例如: {'white': agent1, 'black': agent2}
        """
        self.game = GameStateMachine(game_file)
        self.game_file = game_file
        self.agents = agents
        self.history = [] # 记录棋谱，方便论文里做分析

    def run_match(self, verbose=True):
        print(f"=== Match Start: {self.game_name()} ===")
        print(f"Players: {', '.join([str(a) for a in self.agents.values()])}")
        
        # 1. 获取初始状态
        current_state = self.game.get_initial_state()
        step = 0
        
        while not self.game.is_terminal(current_state):
            step += 1
            if verbose:
                print(f"\n--- Step {step} ---")
                # 简单的状态可视化 (打印出当前被标记的格子)
                self._print_board_state(current_state)

            # 2. 收集所有玩家的动作 (Simultaneous Move)
            moves = {}
            for role in self.game.get_roles():
                # role 可能是 pyswip 的 Atom 对象，转字符串比较安全
                role_str = str(role)
                agent = self.agents.get(role_str)
                
                if not agent:
                    raise ValueError(f"No agent assigned for role: {role_str}")
                
                # 调用 Agent 的思考接口
                move = agent.select_move(self.game, current_state)
                moves[role_str] = move
                
                if verbose:
                    print(f"{agent.name} selects: {move}")

            # 记录历史
            self.history.append({'state': current_state, 'moves': moves})

            # 3. 状态转移
            current_state = self.game.get_next_state(current_state, moves)

        # 4. 游戏结束，结算分数
        print("\n=== Game Over ===")
        self._print_board_state(current_state)
        scores = self.get_scores(current_state)
        print("Final Scores:", scores)
        
        winner = max(scores, key=scores.get)
        print(f"Winner: {winner} (Score: {scores[winner]})")
        return scores

    def get_scores(self, state):
        scores = {}
        for role in self.game.get_roles():
            scores[str(role)] = self.game.get_goal(state, role)
        return scores

    def game_name(self):
        name = game_file.replace(".kif","")
        return name

    def _print_board_state(self, state):
        """
        一个通用的辅助函数，尝试打印出 (cell x y mark) 类型的事实，
        方便调试 Tic-Tac-Toe。
        """
        # 将 Prolog 的 Atom/Functor 转换为字符串列表
        facts = [str(f) for f in state]
        # 过滤出 cell 信息
        cells = [f for f in facts if f.startswith('cell')]
        if cells:
            print(f"Board State: {cells}")
        else:
            print(f"State: {facts}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义游戏文件
    game_file = "games/ticTacToe.kif"
    
    # 2. 初始化不同的 Agent
    # 我们也可以让一个 Random 玩，一个用 (还没实现的) MCTS 玩
    # 注意：这里角色的名字 'white' / 'black' 必须和 kif 文件里定义的 role 一致！
    bot_1 = RandomAgent("Bot_Random_1", "xplayer")
    bot_2 = RandomAgent("Bot_Random_2", "oplayer")
    
    agents = {
        "xplayer": bot_1,
        "oplayer": bot_2
    }
    
    # 3. 启动比赛
    runner = GameRunner(game_file, agents)
    final_scores = runner.run_match()