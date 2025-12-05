import os
import numpy as np


class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j


class GridGraph:
    def __init__(self, file_path=None, width=-1, height=-1, origin=(0, 0),
                 meters_per_cell=0, cell_odds=None, collision_radius=0.15, threshold=-100):

        if file_path:
            self._load(file_path)
        else:
            self.width = width
            self.height = height
            self.origin = origin
            self.meters_per_cell = meters_per_cell
            self.cell_odds = cell_odds

        self.threshold = threshold
        self._setup_collision(collision_radius)

        self.visited_cells = []
        self.parent = {}
        self.distance = {}

    def as_string(self):
        hdr = f"{self.origin[0]} {self.origin[1]} {self.width} {self.height} {self.meters_per_cell}"
        flat = ' '.join(' '.join(row) for row in self.cell_odds.astype(str))
        return f"{hdr} {flat}"

    def _load(self, file_path):
        if not os.path.isfile(file_path):
            raise RuntimeError(f"Could not load map file: {file_path}")

        with open(file_path, "r") as f:
            ox, oy, w, h, res = map(float, f.readline().split())
            self.origin = (ox, oy)
            self.width, self.height = int(w), int(h)
            self.meters_per_cell = res

            self.cell_odds = np.zeros((self.height, self.width), dtype=np.int8)

            for r in range(self.height):
                vals = f.readline().split()
                for c in range(self.width):
                    self.cell_odds[r, c] = np.int8(vals[c])

    def pos_to_cell(self, x, y):
        ci = int((x - self.origin[0]) // self.meters_per_cell)
        cj = int((y - self.origin[1]) // self.meters_per_cell)
        return Cell(ci, cj)

    def cell_to_pos(self, i, j):
        x = self.origin[0] + (i + 0.5) * self.meters_per_cell
        y = self.origin[1] + (j + 0.5) * self.meters_per_cell
        return x, y

    def is_cell_in_bounds(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    def is_cell_occupied(self, i, j):
        return self.cell_odds[j, i] >= self.threshold

    def _setup_collision(self, r):
        cells = int(np.ceil(r / self.meters_per_cell))
        dim = 2 * cells - 1

        rr, cc = np.indices((dim, dim))
        center = cells - 1
        mask = (rr - center) ** 2 + (cc - center) ** 2 <= (cells - 1) ** 2

        self._coll_ind_j, self._coll_ind_i = np.where(mask)
        self.collision_radius = r
        self.collision_radius_cells = cells

    def check_collision(self, i, j):
        js = self._coll_ind_j + j - (self.collision_radius_cells - 1)
        is_ = self._coll_ind_i + i - (self.collision_radius_cells - 1)

        valid = (js >= 0) & (js < self.height) & (is_ >= 0) & (is_ < self.width)
        return np.any(self.is_cell_occupied(is_[valid], js[valid]))

    def get_parent(self, cell):
        return self.parent.get((cell.i, cell.j))

    def init_graph(self):
        self.visited_cells = []
        self.parent.clear()
        self.distance.clear()

    def find_neighbors(self, i, j):
        nbrs = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i + dx, j + dy
            if self.is_cell_in_bounds(ni, nj) and not self.check_collision(ni, nj):
                nbrs.append(Cell(ni, nj))
        return nbrs
