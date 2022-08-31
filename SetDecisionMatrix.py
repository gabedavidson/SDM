"""
A Set Decision Matrix is a representation of an indefinite series. Each value in the SDM is unique.

The SDM operates such that it is used alongside an indefinite series. The series may be any length.
When a unique value is added to the SDM a node is created containing that value and some identifying key.
The operation(s) taken to determine the key can be inputted upon declaration. A default key formula exists.

One node is counted as the current node. When a new unique value is added it becomes the current node.
When a non-unique value is added, the existing node becomes the current node.

Each node has a parent node and children node(s).

The SDM contains a RollbackCondition, which is a condition which triggers a rollback.
The RollbackCondition is used to determine when to rollback and when to stop rolling back.
"""
from __future__ import annotations

import math
import random


class SetDecisionMatrix:
    class SDM:
        def __init__(self, key_formula: SetDecisionMatrix.KeyFormula = None):
            self.root = None
            self.curr = None
            self.length = 0

            if key_formula is None:
                self.formula = SetDecisionMatrix.DefaultKeyFormula()
            else:
                self.formula = key_formula

            print(self.formula.compute(2))

        def insert(self, value):
            check = self.find(self.formula.compute(value))  # try to find it in the SDM
            if check is not None:
                self.curr = check
            else:
                if self.root is None:  # is the first node added
                    self.root = SetDecisionMatrix.SDMNode(self.formula.compute(value), value, None, root=True)
                    self.curr = self.root
                else:  # add to SDM
                    to_add = SetDecisionMatrix.SDMNode(self.formula.compute(value), value, self.curr)
                    self.curr.add(to_add)
                    self.curr = to_add
                    self.length += 1

        def find(self, key):
            if self.length == 0:
                return None
            return self.search(key, self.root)

        def search(self, key, node):
            """
            Returns the node whose key matches the inputted one or None if not found.
            :param key: The key to find
            :param node: The node to find from
            :return: SDMNode or None
            """
            if node.key == key:
                return node
            if len(node.children) == 0:
                return None
            for _node in node.children:
                result = self.search(key, _node)
                if result is not None:
                    return result
            return None

        def rollback(self, delete_prev=True):
            if self.curr.key == self.root.key:
                self.curr = None
                self.delete()
            else:
                key = self.curr.key
                self.curr = self.curr.parent
                if delete_prev:
                    self.delete(key)

        def delete(self, key=None):
            node = self.find(key) if key is None else self.curr
            if node is None:
                return
            ind = random.choice([i for i in range(len(node.children))])  # choose random child of node to be root
            for i in range(len(node.children)):
                if i == ind:
                    continue
                node.children[ind].add(node.children[i])
                node.children[i].parent = node.children[ind]
            node.children[ind].parent = node.parent
            node.children[ind].parent.remove(node)

    class TargetSDM(SDM):
        def __init__(self, target: float = None, key_formula: SetDecisionMatrix.KeyFormula = None):
            SetDecisionMatrix.SDM.__init__(self, key_formula)

            self.prev = self.root
            self.target = target

        def tinsert(self, value):
            super(SetDecisionMatrix.TargetSDM, self).insert(value)
            self.prev = self.curr.parent

    class SDMNode:
        def __init__(self, key: float, value, parent, root=False):
            """
            :param key: some float that characterizes the unique value contained.
            :param value: some generic object
            :param parent: some SDMNode
            """
            self.key = key
            self.value = value

            if parent is None and not root: raise Exception(
                "SDMNode: (value -> {0}, key -> {1}) unrecognized reference to parent.".format(value, key))
            self.parent = parent
            self.children = []

        def add(self, other):
            if type(other) is not SetDecisionMatrix.SDMNode:
                raise Exception("Cannot add non-SDMNode value to children.")
            self.children.append(other)

        def remove(self, other):
            if type(other) is not SetDecisionMatrix.SDMNode:
                raise Exception("Cannot remove non-SDMNode value to children.")
            self.children.remove(other)

    class KeyFormula:
        @staticmethod
        def compute(value):
            return value

    class DefaultKeyFormula(KeyFormula):
        def __init__(self, limit: float = 1000):
            self.limit = limit

        def compute(self, value: float | int):
            return math.sqrt(value / self.limit)

    class RollbackCondition:
        @staticmethod
        def __check__(condition):
            return True if condition else False

    class DefaultRollbackCondition(RollbackCondition):
        @staticmethod
        def __check__(condition):
            return condition is True


def main():
    sdm = SetDecisionMatrix.SDM(key_formula=SetDecisionMatrix.DefaultKeyFormula(1000))
    # k = 0
    # for i in range(2):
    #     for j in range(2):
    #         k += l[i][j] * (i * 2 + j) / (l[i][j] + 1)
    # print(k)
    pass


if __name__ == "__main__":
    main()
