import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from graphviz import Digraph
    import marimo as mo

    return Digraph, mo, np, plt


@app.function
def func(x):
    return 3 * x**2 - 4 * x + 5


@app.cell
def _():
    func(3.0)
    return


@app.cell
def _(np, plt):
    xs = np.arange(-5, 5, 0.25)
    ys = func(xs)
    plt.plot(xs, ys)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `Value` class
    A wrapper around scalar floats for autodiff later. Needs to store:
    - `data`: the value.
    - `grad`: the gradient value.
    - `prev`: the previous nodes in the computational graph for traversal during gradient calculation.

    There are a couple util attrs
    - `op`: the string label of the operation leading to this node
    """)
    return


@app.class_definition
class Value:
    E = 2.718281828459045
    def __init__(self, data, label="", _children=(), _op=""):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        ez = self.E ** (2 * self.data)
        t = (ez - 1) / (ez + 1)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out


@app.cell
def _():
    a = Value(2.0, "a")
    b = Value(-3.0, "b")
    c = Value(10.0, "c")

    e = a * b
    d = e + c
    e.label = "e"
    d.label = "d"
    f = Value(-2.0, "f")
    L = d * f
    L.label = "L"

    L.grad = 1.0
    f.grad = d.data
    d.grad = f.data
    c.grad = f.data
    e.grad = f.data
    a.grad = f.data * b.data
    b.grad = f.data * a.data
    return L, a, b, c


@app.cell
def _(a, b, c):
    h = 1e-5
    d_plus = (a.data + h) * b.data + c.data
    d_minus = (a.data - h) * b.data + c.data
    grad_a_numerical = (d_plus - d_minus) / (2 * h)
    print(f"Numerical ∂d/∂a: {grad_a_numerical:.6f}")  # -3.000000
    print(f"Analytical ∂d/∂a: {b.data:.6f}")  # -3.000000
    return


@app.cell
def _():
    def lol():
        h = 1e-4

        a = Value(2.0, "a")
        b = Value(-3.0, "b")
        c = Value(10.0, "c")
        e = a * b
        d = e + c
        e.label = "e"
        d.label = "d"
        f = Value(-2.0, "f")
        L = d * f
        L1 = L.data

        a = Value(2.0, "a")
        b = Value(-3.0, "b")
        c = Value(10.0, "c")
        e = a * b
        d = e + c
        e.label = "e"
        d.label = "d"
        f = Value(-2.0, "f")
        L = d * f
        L2 = L.data

        return (L2 - L1) / h


    print(lol())
    return


@app.cell
def _(Digraph):
    def trace(root):
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges


    def draw_dot(root):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        nodes, edges = trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }"
                % (getattr(n, "label", ""), n.data, n.grad),
                shape="record",
            )
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot

    return (draw_dot,)


@app.cell
def _(L, draw_dot):
    draw_dot(L)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A single neuron
    """)
    return


@app.cell
def _(draw_dot):
    def neuron():
        # inputs x1, x2
        x1 = Value(2.0, label="x1")
        x2 = Value(0.0, label="x1")

        # weights w1, w2
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.1, label="w2")

        # bias
        b = Value(7.0, label="b")

        x1w1 = x1 * w1; x1w1.label = "x1w1"
        x2w2 = x2 * w2; x2w2.label = "x2w2"
        x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
        n = x1w1x2w2 + b; n.label = "n"
        o = n.tanh(); o.label = "o"
        o.grad = 1.0
        o._backward()
        n._backward()
        x1w1x2w2._backward()
        x1w1._backward()
        x2w2._backward()
        return o

    draw_dot(neuron())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Topological sorting

    To ensure that backpropagation is correct, the chain of gradient backward operations needs to be correct. Doing so requires sorting the computation graph into a DAG. The algorithm to achieve this result is **topological sorting**.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
