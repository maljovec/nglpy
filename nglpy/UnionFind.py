
class Singleton(object):
    def __init(self, id):
        self.id = id
        self.parent = id
        self.rank = 0


class UnionFind(object):
    def __init__(self):
        self.sets = {}

    def make_set(self, id):
        if id not in self.sets:
            self.sets[id] = Singleton(id)
        return self.sets[id]

    def find(self, id):
        if id not in self.sets:
            self.make_set(id)

        if self.sets[id].parent == id:
            return id
        else:
            self.sets[id].parent = self.find(self.sets[id].parent)
            return self.sets[id].parent

    def union(self, x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)

        if xRoot == yRoot:
            return

        if (self.sets[xRoot].rank < self.sets[yRoot].rank) or \
           (self.sets[xRoot].rank < self.sets[yRoot].rank and xRoot < yRoot):
            self.sets[xRoot].parent = yRoot
            self.sets[yRoot].rank = self.sets[yRoot].rank + 1
        else:
            self.sets[yRoot].parent = xRoot
            self.sets[xRoot].rank = self.sets[xRoot].rank + 1

    def count_components(self):
        return len(self.get_component_representatives())

    def get_component_representatives(self):
        roots = set()
        for key in self.sets:
            root = self.find(key)
            if root not in roots:
                roots.add(root)
        return roots

    def get_component_items(self, rep):
        items = []
        for key in self.sets:
            root = self.find(key)
            if (rep == root):
                items.append(key)

        return items
