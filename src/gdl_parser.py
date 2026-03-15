import re
import os
import sys
import argparse

class SExpressionParser:
    """
    负责将 GDL 原始文本解析为 Python 的嵌套列表结构 (S-Expressions)。
    不涉及任何 Prolog 语法转换，只做结构解析。
    """
    
    @staticmethod
    def tokenize(text):
        """词法分析：将字符串拆解为 token 流"""
        # 1. 移除注释 (; 后面的内容直到行尾)
        text = re.sub(r';.*', '', text)
        # 2. 将括号周围加上空格，方便 split
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        # 3. 分割成 tokens
        return text.split()

    @staticmethod
    def read_from_tokens(tokens):
        """
        语法分析：递归构建抽象语法树 (AST),用Python的列表（list）来存储和表示树形结构。
        注意：对于复杂的 GGP 游戏，推荐使用专门的 S-Expression Parser 库来生成 AST，再转 Prolog
        """
        if len(tokens) == 0:
            raise SyntaxError('Unexpected EOF')
        
        token = tokens.pop(0)
        
        if token == '(':
            L = []
            while tokens[0] != ')':
                L.append(SExpressionParser.read_from_tokens(tokens))
            tokens.pop(0)  # 弹出 ')'
            return L
        elif token == ')':
            raise SyntaxError('Unexpected )')
        else:
            # 原子 (Atom) 处理
            return SExpressionParser.atom(token)

    @staticmethod
    def atom(token):
        """处理原子类型：数字转 int/float，其他保留字符串"""
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    @classmethod
    def parse(cls, text):
        """入口函数：解析整个 GDL 文件内容"""
        tokens = cls.tokenize(text)
        expressions = []
        while tokens:
            expressions.append(cls.read_from_tokens(tokens))
        return expressions


class GDLTranslator:
    """
    将解析好的 S-Expression 转换为标准的 Prolog 代码字符串。
    """
    
    def __init__(self):
        self.prolog_rules = []

    def translate(self, expressions):
        '''入口函数'''
        self.prolog_rules = []
        for exp in expressions:
            self.prolog_rules.append(self._visit(exp) + ".")
        return "\n".join(self.prolog_rules)

    def _visit(self, exp):
        """
        递归遍历 S-Expression 树
        原子节点（叶节点）直接处理，列表节点（分支节点）根据操作符类型分别处理
        """
        # 根据节点类型进行不同的转换
        if isinstance(exp, str):  # 字符串节点
            return self._handle_atom(exp)
        if isinstance(exp, int) or isinstance(exp, float):  # 数字节点
            return str(exp)
        
        # 处理AST中的空节点
        if not exp: return ""
        
        head = exp[0]
        
        # --- 处理特殊关键字节点 ---
        if head == '<=':
            # 规则: (<= head body1 body2 ...) -> head :- body1, body2 ...
            rule_head = self._visit(exp[1])
            if len(exp) > 2:
                body_parts = [self._visit(sub) for sub in exp[2:]]
                return f"{rule_head} :- {', '.join(body_parts)}"
            else:
                # 事实: (<= (truth)) -> truth.
                return rule_head
        
        elif head == 'or':
            # 逻辑或: (or a b) -> (a ; b)
            args = [self._visit(sub) for sub in exp[1:]]
            return f"({' ; '.join(args)})"
            
        elif head == 'not':
            # 否定: (not a) -> (\+ a)
            return f"(\+ {self._visit(exp[1])})"
            
        elif head == 'distinct':
            # 不等: (distinct a b) -> a \= b
            return f"{self._visit(exp[1])} \= {self._visit(exp[2])}"
            
        # --- 处理普通谓词节点 ---
        # (cell ?x ?y b) -> cell(VarX, VarY, b)
        else:
            pred_name = self._visit(head)
            if len(exp) > 1:
                args = [self._visit(arg) for arg in exp[1:]]
                return f"{pred_name}({', '.join(args)})"
            else:
                return pred_name

    def _handle_atom(self, token):
        """处理原子符号（变量、常量）的Prolog语法转换"""
        if token.startswith('?'):
            # 变量：?x -> VarX (Prolog变量必须大写开头)
            # 我们使用 Var 前缀来确保不会和某些内置关键词冲突
            return 'Var' + token[1:].capitalize()
        else:
            # 普通原子：保持小写 (GDL不区分大小写，但Prolog原子通常小写)
            # 处理特殊符号，如 <= 在前面已经处理了，这里处理普通的 atom
            return str(token).lower()


def translate_gdl_to_prolog(input_file, output_file='prolog'):
    """
    将GDL文件转换为Prolog文件
    
    Args:
        input_file: 输入的GDL/KIF文件路径
        output_file: 输出的Prolog文件路径（默认在prolog文件夹下）
    """
    # 读取GDL文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            gdl_content = f.read()
        print(f"读取文件: {input_file}")
    except FileNotFoundError:
        print(f"Can't find file: {input_file}")
        return False
    except Exception as e:
        print(f"Something wrong when reading file: {e}")
        return False
    
    # 解析GDL
    parser = SExpressionParser()
    try:
        parsed_sexpr = parser.parse(gdl_content)
        print(f"成功解析 {len(parsed_sexpr)} 个表达式")
    except Exception as e:
        print(f"解析GDL时出错: {e}")
        return False
    
    # 转换为Prolog
    translator = GDLTranslator()
    try:
        prolog_code = translator.translate(parsed_sexpr)
    except Exception as e:
        print(f"转换到Prolog时出错: {e}")
        return False
    
    # 确定输出文件路径
    # 如果没有prolog文件夹，则创建一个
    if not os.path.exists("prolog"):
        os.makedirs("prolog")
        
    # 使用输入文件的名称
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join("prolog", f"{base_name}.pl")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存Prolog文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prolog_code)
        print(f"Prolog code save to: {output_file}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

# --- 命令行接口 ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='将GDL/KIF文件转换为Prolog文件')
    parser.add_argument('-i', '--input', default='games/breakthrough.kif', 
                       help='输入的GDL文件路径')
    parser.add_argument('-o', '--output', 
                       help='输出的Prolog文件路径 (默认: prolog/<输入文件名>.pl)')
    
    args = parser.parse_args()
    
    success = translate_gdl_to_prolog(args.input, args.output)
    if success:
        print("已成功将GDL转换成Prolog")