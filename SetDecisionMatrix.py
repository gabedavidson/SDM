"""
See README on github repo.
"""
from __future__ import annotations

import random
from itertools import chain

import numpy as np


class SetDecisionMatrix:
    class SDM:
        def __init__(self, key_formula: SetDecisionMatrix._KeyFormula = None):
            self.root = None
            self.curr = None
            self.count = 0

            if key_formula is None:
                self.formula = SetDecisionMatrix.DefaultKeyFormula()
            else:
                self.formula = key_formula

        def insert(self, value):
            check = self.get(value)  # try to find it in the SDM
            if value > self.formula.limit:
                self.formula.limit *= 2
                self.update_keys()
            if check is not None:  # this value already exists
                self.curr = check
            else:  # this value does not exist yet
                if self.root is None:  # is the first node added
                    self.root = SetDecisionMatrix.SDMNode(self.formula.compute(value), value, None, root=True)
                    self.curr = self.root
                else:  # add to SDM
                    to_add = SetDecisionMatrix.SDMNode(self.formula.compute(value), value, self.curr)
                    self.curr.add_child(to_add)
                    self.curr = to_add
                self.count += 1

        def get(self, value):
            if self.count == 0:
                return None
            return self._search(self.formula.compute(value), self.root)

        def _search(self, key, node):
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
                result = self._search(key, _node)
                if result is not None:
                    return result
            return None

        def remove(self, value=None):
            node = self.get(self.formula.compute(value)) if value is not None else self.curr
            if node is None:
                return

            for child in node.children:
                node.parent.add_child(child)  # automatically sets child's parent to node.parent
            del node

        def print(self, print_value=False, print_structure=False):
            if self.count == 0:
                print("SDM is empty.")
                return
            self._rec_print(self.root, print_value, print_structure)

        def _rec_print(self, node, print_value=False, print_structure=False):
            print("Key:", node.key)
            if print_value:
                print("Value:", node.value)
            if print_structure:
                print("ROOT" if node.parent is None else "Parent: " + str(node.parent.key))
            print("====================")
            for child in node.children:
                self._rec_print(child, print_value, print_structure)

        def update_keys(self):
            self.rec_update_key(self.root)

        def rec_update_key(self, node):
            node.key = self.formula.compute(node.value)
            if len(node.children) != 0:
                for child in node.children:
                    self.rec_update_key(child)

    class TargetSDM(SDM):
        """
        This version of the SDM prioritizes inputs that are closer to the target.
        You can choose to save only improving inputs or sort every inputted value.
        """

        def __init__(self, target: any, key_formula: SetDecisionMatrix._KeyFormula = None, imp_only=True):
            SetDecisionMatrix.SDM.__init__(self, key_formula)

            self.improving_only = imp_only
            self.target = self.formula.compute(target)
            self.root = None  # SetDecisionMatrix.TSDMNode(0, self.target, None, True)
            self.curr = None
            self.improved_last = False

            self.keys = []

        def __bool__(self):
            try:
                return self.curr.key == self.target
            except TypeError:
                return False

        def best(self, true=True):
            return self.curr.value if true else self.curr

        def improved(self) -> bool:
            return self.improved_last

        def insert(self, value):
            key = self.formula.compute(value)
            lngth = self.count
            if self.root is None:
                self.root = SetDecisionMatrix.TSDMNode(key, value, None, True)
                self.curr = self.root
                self.count += 1
            else:
                if key not in self.keys:
                    if self.improving_only:
                        self.imp_insert(value)
                    else:
                        self.all_insert(value)
                else:
                    return
            self.keys.append(key)
            if self.count > lngth:
                while self.formula.compute(value) > 1:
                    self.formula.limit *= 2
                    self.update_keys()
            if key == self.target:
                print("Target reached.")

        def imp_insert(self, value):
            key = self.formula.compute(value)
            curr = abs(self.target - self.curr.key)
            inputted = abs(self.target - key)
            if curr < inputted:  # curr is better than inputted value
                self.improved_last = False
                return
            else:  # inputted value is better than curr
                if key == self.curr.key:
                    return
                node = SetDecisionMatrix.TSDMNode(key, value, self.curr)
                self.curr.good = node
                self.curr = node
                self.count += 1
                self.improved_last = True

        def all_insert(self, value):
            node = SetDecisionMatrix.TSDMNode(self.formula.compute(value), value, detach=True)
            self.rec_add(value, self.root, node)
            if node.is_better_than(self.curr.key, self.target):
                self.curr = node
                self.improved_last = True
            else:
                self.improved_last = False

        def rec_add(self, value, curr, new):
            if new.key == curr.key:
                return None
            if curr.is_better_than(new.key, self.target):  # parent > added node –> __bad__
                if curr.bad is None:
                    curr.__bad__(new)
                    self.count += 1
                    return None
                else:
                    if self.rec_add(value, curr.bad, new) is None:
                        return None
            else:  # added node > parent –> __good__
                if curr.good is None:
                    curr.__good__(new)
                    self.count += 1
                    return None
                else:
                    if self.rec_add(value, curr.good, new) is None:
                        return None

        def get_keys(self, curr: SetDecisionMatrix.TSDMNode, keys=None):
            if keys is None:
                keys = []
            if curr.good is not None:
                self.get_keys(curr.good, keys=keys)
            keys.append(curr.key)
            if curr.bad is not None:
                self.get_keys(curr.bad, keys=keys)
            self.keys = keys

        def get(self, value) -> SetDecisionMatrix.TSDMNode:
            return self._search(self.formula.compute(value), self.root)

        def _search(self, key, node):
            while node.key != key:
                if node.is_better_than(key, self.target):
                    node = node.bad
                else:
                    node = node.good
                if node is None:
                    return None
            return node

        def update_keys(self):
            self.rec_update_key(self.root)
            self.get_keys(self.root)

        def rec_update_key(self, node):
            node.key = self.formula.compute(node.value)
            if node.good is not None:
                self.rec_update_key(node.good)
            if node.bad is not None:
                self.rec_update_key(node.bad)

        def print(self, print_value=False, print_structure=False):
            if self.count == 0:
                print("TargetSDM is empty.")
                return
            self._rec_print(self.root, print_value, print_structure)
            if not self.improving_only:
                print("\nBest: {0} ({1})".format(self.best(), self.curr.key))

        def _rec_print(self, node, print_value=False, print_structure=False):
            print("Node key: " + str(node.key))
            if print_value:
                print("Node value: " + str(node.value))
            if print_structure:
                print("Parent: " + str(node.parent.key)) if node.parent is not None else print("Parent is None")
            print("========================")
            if node.good is not None:
                self._rec_print(node.good, print_value, print_structure)
            if node.bad is not None:
                self._rec_print(node.bad, print_value, print_structure)

    class SDMNode:
        """
        Node class for SDM structure.
        """

        def __init__(self, key: float, value, parent=None, root=False):
            """
            :param key: some float that characterizes the unique value contained.
            :param value: some generic object
            :param parent: some SDMNode
            """
            self.key = key
            self.value = value

            if parent is None and not root:
                raise Exception(
                    "SDMNode: (value -> {0}, key -> {1}) unrecognized reference to parent.".format(value, key))
            self.parent = parent
            self.children = []

        def add_child(self, other):
            """
            Adds an SDMNode to self.children
            :param other: SDMNode to add
            :return: None
            """
            if type(other) is not SetDecisionMatrix.SDMNode:
                raise Exception("Cannot add non-SDMNode value to children.")
            self.children.append(other)
            other.parent = self

        def remove_child(self, other):
            """
            Removes an SDMNode from self.children
            :param other: SDMNode to remove
            :return: None
            """
            if type(other) is not SetDecisionMatrix.SDMNode:
                raise Exception("Cannot remove non-SDMNode value to children.")
            try:
                ind = self.children.index(other)
                if len(self.children[ind].children) > 0:
                    for child in self.children[ind].children:
                        child.parent = self.parent
                self.children.pop(ind)
            except ValueError():
                return False

    class TSDMNode(SDMNode):
        """
        Node class for TargetSDM structure.

        Derives from SDMNode. Implements left and right (good and bad) children as opposed to
        and indeterminable number of children.
        """

        def __init__(self, key: float, value, parent=None, detach: bool = False):
            """
            Init method for TSDMNode
            :param key: the key to be assigned to this node
            :param value: the original value
            :param parent: some TSDMNode
            :param detach: whether to allow None parent
            """
            SetDecisionMatrix.SDMNode.__init__(self, key, value, parent, detach)
            self.good, self.bad = None, None

        def __bad__(self, other):
            """
            Set method for self.bad
            :param other: set value
            :return: None
            """
            self.bad = other
            other.parent = self

        def __good__(self, other):
            """
            Set method for self.good
            :param other: set value
            :return: None
            """
            self.good = other
            other.parent = self

        def is_better_than(self, key, target, use_implicit=False):
            """
            Returns whether one node is "better" (closer to the target) than another using their key
            :param key: The key of the node to compare against
            :param target: The target value of TargetSDM
            :param use_implicit: If target is 0, can simplify computation
            :return: bool –> If self.key is better than inputted key then True, else False
            """
            if use_implicit:
                if target == 0:
                    return (self.key < key) & (self.key >= 0)
                return self.is_better_than(key, target, False)
            if abs(target - self.key) < abs(target - key):
                return True
            return False

    class _KeyFormula(object):
        def __init__(self, limit: float = 1000, override_exception=False):
            self.limit = limit
            if not override_exception:
                raise Exception("(KeyFormulaBase) Must implement inherited object.")

        @staticmethod
        def compute(value):
            return value

    class DefaultKeyFormula(_KeyFormula):
        """
        Useful for int and float based (Target)SDMs
        """

        def __init__(self, limit: float = 1000):
            SetDecisionMatrix._KeyFormula.__init__(self, limit, override_exception=True)

        def compute(self, value):
            return value / self.limit

    """
    Below are a number of prebuilt KeyFormulas for certain input types.
    """

    # 1D grid –> list[float | int]
    class OneDimGridKeyFormula(DefaultKeyFormula):
        def __init__(self, solution: list[float | int] = None, limit: float = 1000):
            self.targeted = True if solution is None else False
            self.dimension = len(solution) if solution is not None else -1
            self.known = {}
            if self.targeted:
                self.__load__(solution)
            SetDecisionMatrix.DefaultKeyFormula.__init__(self, limit=limit)

        def compute(self, value: list[float | int]):
            total = 0
            for i in range(len(value)):
                try:
                    d = self.__distance__(value, i) if self.targeted else (i + 1) * value[i] / self.limit
                    total += d
                except IndexError:
                    raise Exception("Inputted grid dimensions differ from target.")
            return super(self).compute(total)

        def __load__(self, solution):
            for i in range(self.dimension):
                self.known[solution[i]] = i

        def __distance__(self, value, i):
            return abs(i - self.known[value])

        def __delta__(self, value: any, curr_pos: int, new_pos: int):
            return self.__distance__(value, curr_pos) - self.__distance__(value, new_pos)

    # 2D grid –> list[list[float | int]]
    class TwoDimGridKeyFormula(DefaultKeyFormula):
        """
        Useful for 2D array based (Target)SDMs whose dimensions are equal.
        SetDecisionMatrix.TargetSDM.target is always 0 when using this KeyFormula.
        """

        def __init__(self, solution: list[list[float | int]] = None,
                     limit: float = 1000):
            """
            If no solution, then a non-targeted SDM is assumed.
            :param solution: 2D grid (None if non-targeted)
            :param limit: KeyFormula limit. Arbitrary.
            """
            self.targeted = True if solution is not None else False
            self.dimension = len(solution)
            self.known = {}
            if self.targeted:
                self.__load__(self.dimension, solution)
            SetDecisionMatrix.DefaultKeyFormula.__init__(self, limit=limit)

        def compute(self, value: list[list[float | int]]):
            """
            Override of base–>compute(). Computes the total of the distances of each tile in the inputted
            grid to their true positions as inputted in self.__init__() parameter *solution
            :param value: inputted grid
            :return: float –> such that the closer to 0 the value, the closer to the solution this grid is.
            """
            total = 0
            for i in range(self.dimension):
                for k in range(self.dimension):
                    try:
                        d = self.__distance__(value[i][k], i, k) if self.targeted else (((i + 1) * (k + 1)) + value[i][
                            k]) / self.limit
                        total += d
                    except IndexError:
                        raise Exception("Inputted grid dimensions differ from solution.")
            return super(SetDecisionMatrix.TwoDimGridKeyFormula, self).compute(total)

        def __load__(self, dimension, solution):
            """
            Prefinds the coordinates of each true value in self.solution to reduce runtime for future calls
            of self.compute()
            :param dimension: the dimension of the grid
            :return: None
            """
            for i in range(dimension):
                for k in range(dimension):
                    self.known[solution[i][k]] = [i, k]

        def __distance__(self, value, i, k):
            """
            Finds the distance between two coordinate points, assuming non-diagonal travel.
            IMPORTANT: This function assumes that *value lives within self.solution.
            :param value: The value which lives on the coordinate to find
            :param i: Coordinate[0]
            :param k: Coordinate[1]
            :return: int –> the distance between the true position of the value and the coordinate position (i, k)
            """
            try:
                true = self.known[value]
            except Exception:
                raise Exception("Cannot compute input against solution.")
            return abs(i - true[0]) + abs(k - true[1])

        def __delta__(self, value, curr_pos: list[int], new_pos: list[int]):
            """
            Finds the change in distance if a value were to be moved elsewhere in the grid.
            IMPORTANT: This function assumes that *value lives within self.solution.
            :param value: the value to move
            :param curr_pos: list –> [x, y]
            :param new_pos: list –> [x, y]
            :return: float –> such that: if float < 0 then curr is better else new is better
            """
            return self.__distance__(value, curr_pos[0], curr_pos[1]) - self.__distance__(value, new_pos[0],
                                                                                          new_pos[1])

    # String or char –> str | chr
    class StringKeyFormula(DefaultKeyFormula):
        def __init__(self, limit: float = 1000):
            self.mark = str
            SetDecisionMatrix.DefaultKeyFormula.__init__(self, limit=limit)

        def compute(self, value: str):
            return super(self).compute(sum([ord(char) for char in value]))

    # Dictionary –> dict
    class DictKeyFormula(DefaultKeyFormula):
        def __init__(self, weight: list[any] = None, limit: int = 1000):
            SetDecisionMatrix.DefaultKeyFormula.__init__(self, limit=limit)
            self.weight = weight if weight is not None else [.5, .5]
            if sum(self.weight) != 1:
                raise Exception("Weightings must sum to 1.0")

        def compute(self, value: dict):
            total = 0
            for k, v in value.items():
                total += ((super(self).compute(k) * self.weight[0]) + (super(self).compute(v) * self.weight[1]))
            return total


def make_random_2d_grid(dim: int = 3):
    return [[random.random() * 10 for _ in range(dim)] for _ in range(dim)]


def randomize_2d_grid(grid):
    values = list(chain.from_iterable(grid))
    random.shuffle(values)
    return [[values[i + j * len(grid)] for i in range(len(grid))] for j in range(len(grid))]


def get_avg_dev_2d(v1, v2):
    assert len(v1) == len(v2)
    length = len(v1)
    return sum([sum([abs(v1[j][i] - v2[j][i]) for i in range(length)]) for j in range(length)]) / length


def main():
    """print("Target: " + str(sdm.target) + "\n")

    sdm.insert(5)
    sdm.insert(19)
    sdm.insert(12)
    sdm.insert(14)
    sdm.insert(15)
    sdm.insert(11)
    sdm.insert(17)
    sdm.insert(17)
    sdm.insert(16)

    sdm.print(print_value=True, print_structure=True)

    node1 = sdm.get(12)
    print(node1.value)
    node2 = sdm.get(11)
    print(node2.value)"""

    solution = make_random_2d_grid()
    tsdm = SetDecisionMatrix.TargetSDM(target=solution, imp_only=False,
                                       key_formula=SetDecisionMatrix.TwoDimGridKeyFormula(solution=solution))

    iters, max_iters = 0, 500
    while iters < max_iters:
        tsdm.insert(randomize_2d_grid(solution))
        if tsdm.best(true=False) == tsdm.formula.compute(solution):
            print(f"Found after {iters} iterations.")
            break
        iters += 1
    else:
        print("Could not find solution.")

    tsdm.print(print_value=True, print_structure=True)
    print("\nSolution:", solution)
    print("Best:", tsdm.best())
    print(f"Average deviation: {get_avg_dev_2d(tsdm.best(), solution)}")


if __name__ == "__main__":
    main()


