from pyswip import Prolog
from gdl_parser import SExpressionParser, GDLTranslator

class GameStateMachine:
    def __init__(self, rule_file):
        self.prolog = Prolog()
        self.translator = GDLTranslator()
        self.parser = SExpressionParser()
        
        # 加载并转换规则
        self._load_and_transform_rules(rule_file)
        
    def _load_and_transform_rules(self, filename):
        # print(f"--- Loading GDL: {filename} ---")
        with open(filename, 'r') as f:
            gdl_content = f.read()
            
        # 1. 词法与语法分析 (GDL -> Python List)
        # 这一步是纯 Python
        parsed_expressions = self.parser.parse(gdl_content)
        
        # 2. 代码生成 (Python List -> Prolog String)
        # 这一步也是纯 Python
        prolog_code = self.translator.translate(parsed_expressions)
        
        # 调试：打印翻译后的前500个字符看看是否正确
        # print("--- Translated Prolog Code (Preview) ---")
        # print(prolog_code[:500] + "...\n")
        
        # 3. 注入引擎 (Prolog String -> SWI-Prolog Engine)
        # 这里用 pyswip
        # 我们将翻译好的每一行代码 assert 到 Prolog 知识库中
        # print("--- Injecting into Prolog Engine ---")
        for rule in prolog_code.split('\n'):
            if rule.strip():
                try:
                    self.prolog.assertz(rule[:-1]) # 去掉末尾的 '.'，因为assertz会自动处理
                except Exception as e:
                    print(f"Prolog Error on line: {rule}\nError: {e}")

    # --- 下面是标准的 GGP 接口 (保持不变) ---

    def get_roles(self):
        # 对应 GDL: (role ?r) -> Prolog: role(R)
        return [sol['R'] for sol in self.prolog.query("role(R)")]

    def get_initial_state(self):
        # 对应 GDL: (init ?f) -> Prolog: init(F)
        return [sol['F'] for sol in self.prolog.query("init(F)")]

    def get_legal_moves(self, state, role):
        self._reset_state(state)
        # 对应 GDL: (legal ?r ?m) -> Prolog: legal(role, M)
        # 注意：Prolog 中变量是大写的，所以 role 如果是 'white' 需保持小写作为原子
        query = f"legal({role}, M)"
        solutions = [sol['M'] for sol in self.prolog.query(query)]
        # Prolog 可能因为多条规则都允许同一个动作，导致返回多次，用set去重
        return list(set(solutions))

    def get_next_state(self, state, moves):
        self._reset_state(state)
        self._inject_moves(moves)
        # 对应 GDL: (next ?f) -> Prolog: next(F)
        next_state = [sol['F'] for sol in self.prolog.query("next(F)")]
        # next(F) 会因为 GDL 中的 OR 分支返回重复事实，必须用set去重
        unique_state = list(set(next_state))
        self._clean_moves() # 清理 does，防止污染下一步
        return unique_state

    def is_terminal(self, state):
        self._reset_state(state)
        return list(self.prolog.query("terminal")) != []

    def get_goal(self, state, role):
        self._reset_state(state)
        # 对应 GDL: (goal ?r ?v) -> Prolog: goal(role, V)
        q = f"goal({role}, V)"
        res = list(self.prolog.query(q))
        if res: return res[0]['V']
        return 0

    # --- 辅助函数：上下文管理 ---

    def _reset_state(self, state):
        """
        清空上一回合的 true(...)，注入当前回合的 true(...)
        """
        self.prolog.retractall("true(_)")
        for fact in state:
            # 这里的 fact 是 pyswip 的 Functor 对象或 Atom
            # 我们直接转字符串拼进去，或者直接传对象（如果 pyswip 版本支持）
            self.prolog.assertz(f"true({fact})")

    def _inject_moves(self, moves):
        """
        moves: {'white': 'mark(1,1)', 'black': 'noop'}
        """
        for role, move in moves.items():
            self.prolog.assertz(f"does({role}, {move})")

    def _clean_moves(self):
        self.prolog.retractall("does(_,_)")


# 核心设计原理解析（对应你的参考论文）
# 为什么需要 _inject_state？:

# 根据参考论文 General Game Playing Overview...  中提到，GDL 使用关键字 true(<p>) 来表示当前状态下的事实。

# Prolog 是逻辑引擎，它本身不存储“当前到了第几回合”。
# 所以每次查询 legal 或 next 之前，我们必须把 Python 对象中的 state 列表，转换成 Prolog 里的 true(...) 事实。
# 这被称为 Context Injection。


# # --- 验证环节 ---
# if __name__ == "__main__":
#     # 确保目录下有 tictactoe.kif
#     game = GameStateMachine("games/ticTacToe.kif")
    
#     print("\n=== Game Loaded Successfully ===")
#     roles = game.get_roles()
#     print(f"Roles: {roles}")
    
#     state = game.get_initial_state()
#     print(f"Init State Size: {len(state)}")
    
#     # 测试一下合法走步
#     # 注意：Prolog解析后，role可能是 Atom('white')，打印出来没引号
#     current_role = roles[0] 
#     moves = game.get_legal_moves(state, current_role)
#     print(f"Legal moves for {current_role}: {moves}")