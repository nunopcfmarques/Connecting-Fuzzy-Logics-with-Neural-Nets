from collections import deque
class Node:
    def __init__(self, data: str, depth: int) -> None:
        self.left = None
        self.right = None
        self.data = data
        self.depth = depth

def level_order_traversal(root: Node) -> list[(str, int)]:
    if root is None:
        return []

    result = []
    queue = deque()
    queue.append(root)

    while queue:
        node = queue.popleft()
        result.append((node.data, node.depth))

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result

def print_tree(root: Node) -> Node:
    if root is None:
        return

    queue = deque()
    queue.append(root)

    while queue:
        node_count = len(queue)
        while node_count > 0:
            node = queue.popleft()
            print(f"{node.data} ({node.depth})", end=" ")
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            node_count -= 1
        print()

def get_children(node: Node) -> list[Node]:
        children = []
        if node.left is not None:
            children.append(node.left)
        if node.right is not None:
            children.append(node.right)
        return children
