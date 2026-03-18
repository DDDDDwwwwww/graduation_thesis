"""GDL/KIF 解析与 Prolog 翻译工具。

本模块分两层：
1. `SExpressionParser`：把 GDL 文本解析为 Python 嵌套列表（S-Expression）。
2. `GDLTranslator`：把 S-Expression 翻译为可执行的 Prolog 规则。
"""

from __future__ import annotations

import argparse
import os
import re


class SExpressionParser:
    """S-Expression 解析器（仅做结构解析，不做语义推理）。"""

    @staticmethod
    def tokenize(text):
        """将 GDL 文本切分为 token 流。"""
        # 1) 去掉分号注释（`;` 到行尾）。
        text = re.sub(r";.*", "", text)
        # 2) 让括号成为独立 token。
        text = text.replace("(", " ( ").replace(")", " ) ")
        return text.split()

    @staticmethod
    def read_from_tokens(tokens):
        """递归下降解析 token 流，返回一个 S-Expression 节点。"""
        if len(tokens) == 0:
            raise SyntaxError("Unexpected EOF")

        token = tokens.pop(0)
        if token == "(":
            node = []
            while tokens[0] != ")":
                node.append(SExpressionParser.read_from_tokens(tokens))
            tokens.pop(0)  # 弹出 ')'
            return node
        if token == ")":
            raise SyntaxError("Unexpected )")
        return SExpressionParser.atom(token)

    @staticmethod
    def atom(token):
        """把原子 token 转成 int/float/str。"""
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    @classmethod
    def parse(cls, text):
        """解析完整文本，返回顶层表达式列表。"""
        tokens = cls.tokenize(text)
        expressions = []
        while tokens:
            expressions.append(cls.read_from_tokens(tokens))
        return expressions


class GDLTranslator:
    """将 S-Expression 翻译为 Prolog 代码。"""

    def __init__(self):
        self.prolog_rules = []

    def translate(self, expressions):
        """翻译入口：把表达式列表转成多行 Prolog 规则。"""
        self.prolog_rules = []
        for exp in expressions:
            self.prolog_rules.append(self._visit(exp) + ".")
        return "\n".join(self.prolog_rules)

    def _visit(self, exp):
        """递归访问 AST 节点并生成对应 Prolog 代码片段。"""
        if isinstance(exp, str):
            return self._handle_atom(exp)
        if isinstance(exp, (int, float)):
            return str(exp)
        if not exp:
            return ""

        head = exp[0]

        # 规则：(<= head body1 body2 ...) -> head :- body1, body2 ...
        if head == "<=":
            rule_head = self._visit(exp[1])
            if len(exp) > 2:
                body_parts = [self._visit(sub) for sub in exp[2:]]
                return f"{rule_head} :- {', '.join(body_parts)}"
            return rule_head

        # 逻辑或：(or a b c) -> (a ; b ; c)
        if head == "or":
            args = [self._visit(sub) for sub in exp[1:]]
            return f"({' ; '.join(args)})"

        # 否定：(not a) -> (\+ a)
        if head == "not":
            return f"(\\+ {self._visit(exp[1])})"

        # 不同：(distinct a b) -> a \= b
        if head == "distinct":
            return f"{self._visit(exp[1])} \\= {self._visit(exp[2])}"

        # 普通谓词：(cell ?x ?y b) -> cell(VarX, VarY, b)
        pred_name = self._visit(head)
        if len(exp) > 1:
            args = [self._visit(arg) for arg in exp[1:]]
            return f"{pred_name}({', '.join(args)})"
        return pred_name

    def _handle_atom(self, token):
        """处理原子符号（变量/常量）的 Prolog 形式。"""
        if token.startswith("?"):
            # GDL 变量统一转成 Prolog 变量（大写开头）。
            return "Var" + token[1:].capitalize()
        # 非变量统一小写，便于作为 Prolog 原子。
        return str(token).lower()


def translate_gdl_to_prolog(input_file, output_file="prolog"):
    """把 GDL/KIF 文件转换为 `.pl` 文件并保存。"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            gdl_content = f.read()
        print(f"读取文件: {input_file}")
    except FileNotFoundError:
        print(f"Can't find file: {input_file}")
        return False
    except Exception as e:
        print(f"Something wrong when reading file: {e}")
        return False

    parser = SExpressionParser()
    try:
        parsed_sexpr = parser.parse(gdl_content)
        print(f"成功解析 {len(parsed_sexpr)} 个表达式")
    except Exception as e:
        print(f"解析GDL时出错: {e}")
        return False

    translator = GDLTranslator()
    try:
        prolog_code = translator.translate(parsed_sexpr)
    except Exception as e:
        print(f"转换到Prolog时出错: {e}")
        return False

    if not os.path.exists("prolog"):
        os.makedirs("prolog")

    # 默认输出到 `prolog/<输入文件名>.pl`
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join("prolog", f"{base_name}.pl")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(prolog_code)
        print(f"Prolog code save to: {output_file}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="把 GDL/KIF 文件转换为 Prolog 文件")
    arg_parser.add_argument(
        "-i",
        "--input",
        default="games/breakthrough.kif",
        help="输入 GDL 文件路径",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        help="输出 Prolog 文件路径（默认: prolog/<输入文件名>.pl）",
    )

    args = arg_parser.parse_args()
    success = translate_gdl_to_prolog(args.input, args.output)
    if success:
        print("已成功将 GDL 转换成 Prolog")
