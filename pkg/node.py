from enum import IntEnum

from state import State


class NodeType(IntEnum):
    UNKNOWN = 0
    OBSTACLE = 1
    EMPTY = 2
    VICTIM = 3
    SAVED = 4


class Node:
    def __init__(self, type: NodeType, row, col):
        self.type = type
        self.state = State(row, col)
