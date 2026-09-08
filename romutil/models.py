import enum

class Direction(enum.IntEnum):
    north = 0
    east = 1
    up = 2
    south = 3
    west = 4
    down = 5
    mod = 6

    def invert(self):
        return Direction((self + self.mod // 2) % self.mod)

direction_matrix = [
    Direction.north,
    Direction.east,
    Direction.south,
    Direction.west,
    Direction.up,
    Direction.down,
]

class Exit:
    def __init__(self, e, source, distance=1):
        self.src = source
        self.dst = e[1]
        self.direction = Direction(direction_matrix[e[0]])
        self.distance = distance
        self.one_way = False

    def __eq__(self, other):
        if not isinstance(other, Exit):
            return False
        if self.src == other.src and self.dst == other.dst and self.direction == other.direction:
            return True
        if self.src == other.dst and self.dst == other.src and self.direction == other.direction.invert():
            return True
        return False

    def __contains__(self, room):
        return room in (self.src, self.dst)

    def __repr__(self):
        return f'{self.src} -> {self.dst} ({self.distance} {self.direction.name})'

    def __hash__(self):
        r0, r1, d = (self.src, self.dst, self.direction) if self.src < self.dst \
            else (self.dst, self.src, self.direction.invert())
        return hash(f'{r0} {r1} {d}')

class Room:
    def __init__(self, r):
        self.vnum = r[0]
        self.name = r[1]
        self.desc = r[2]
        self.exits = [] if not r[3] else [Exit(e, self.vnum) for e in r[3] if e is not None]
        self.fixups = []
        self.dummy = False
        self.x = None
        self.y = None
        self.z = None

    def replace_exit(self, orig, replacement, distance):
        for e in self.exits:
            if e.dst == orig:
                e.dst = replacement
                e.distance += distance

    def __repr__(self):
        return f'[{self.vnum}: {self.name}] {{{self.exits}}}'
