import inspect
import importlib
import numpy
import re
import ast


def parse_aliases(source_code):
    all_imports = re.findall(r'(import .*(\n|))', source_code)
    return ''.join([i[0] for i in all_imports])


def get_methods(source_code):
    tree = ast.parse(source_code)

    methods = []
    imports = []

    class MethodVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check if the function call is an attribute (e.g., torch.tensor)
            if isinstance(node.func, ast.Attribute):
                # Get the full method name (e.g., torch.tensor)
                method_name = f"{node.func.value.id}.{node.func.attr}"
                methods.append(method_name)
            # Check if the function call is a name (e.g., np.array)
            elif isinstance(node.func, ast.Name):
                method_name = node.func.id
                methods.append(method_name)
            # Continue traversing the AST
            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                imports.append(alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                imports.append(f"{node.module}.{alias.name}")
            self.generic_visit(node)

    # Create an instance of the visitor and visit the AST
    visitor = MethodVisitor()
    visitor.visit(tree)

    return methods, imports



def parse_args(source_code, methods: list):
    aliases = parse_aliases(source_code)
    torch = exec(aliases)
    for method in methods:
        try:
            tensor_func = eval("numpy.choose")
            docs = (tensor_func.__doc__)
            signature = inspect.signature(tensor_func)
            parameters = signature.parameters
            return docs, [param.name for param in parameters.values()]
        except (Exception) as ex:
            print(ex)
            print('-------------')

# example use
# a = parse_args(
#     '''
# import numpy as np

# def func(a):
#     return np.array((a, a))
# ''', methods=['np.array'])
# print(a[0])
