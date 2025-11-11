import csv
import numpy as np

class TransportProblem:
    """
    TransportProblem encapsulates a transportation problem:
      - costs: 2D array (m x n)
      - supplies: length-m array
      - demands: length-n array

    Features:
      - automatic balancing (adds dummy row/col if needed)
      - vogelInitialSolution() for initial basic feasible solution
      - solveByPotentials() implements the method of potentials (u/v), cycle improvement
      - helper functions for degeneracy handling and cycle finding
    All method/variable names use camelCase.
    """

    def __init__(self, costs, supplies, demands, tol=1e-9):
        self.costs = np.array(costs, dtype=float)
        self.supplies = np.array(supplies, dtype=float)
        self.demands = np.array(demands, dtype=float)
        self.tol = tol
        self.m, self.n = self.costs.shape
        self._balanceProblem()
        # allocation uses np.nan for empty cells; basic cells are non-nan
        self.allocation = np.full((self.m, self.n), np.nan)

    def _balanceProblem(self):
        """Balance supplies and demands by adding a dummy row or column with zero cost if necessary."""
        totalSupply = float(self.supplies.sum())
        totalDemand = float(self.demands.sum())
        if abs(totalSupply - totalDemand) < self.tol:
            return
        if totalSupply > totalDemand:
            # add dummy demand (column)
            dummyCol = np.zeros((self.m, 1))
            self.costs = np.hstack((self.costs, dummyCol))
            self.demands = np.append(self.demands, totalSupply - totalDemand)
        else:
            # add dummy supply (row)
            dummyRow = np.zeros((1, self.n))
            self.costs = np.vstack((self.costs, dummyRow))
            self.supplies = np.append(self.supplies, totalDemand - totalSupply)
        self.m, self.n = self.costs.shape

    def setAllocation(self, allocation):
        """Set allocation matrix directly (use np.nan for empty cells)."""
        self.allocation = np.array(allocation, dtype=float)

    def vogelInitialSolution(self):
        """
        Vogel's approximation method to produce an initial feasible (basic) solution.
        Sets self.allocation with allocations (and then ensures non-degeneracy by adding zero basics).
        """
        supplies = self.supplies.copy()
        demands = self.demands.copy()
        allocation = np.full((self.m, self.n), np.nan)
        remainingRows = set(range(self.m))
        remainingCols = set(range(self.n))

        while remainingRows and remainingCols:
            # compute penalties for rows
            rowPenalties = {}
            for i in remainingRows:
                costsRow = [self.costs[i, j] for j in remainingCols]
                if len(costsRow) >= 2:
                    sortedCosts = sorted(costsRow)
                    rowPenalties[i] = sortedCosts[1] - sortedCosts[0]
                else:
                    rowPenalties[i] = costsRow[0]

            # compute penalties for columns
            colPenalties = {}
            for j in remainingCols:
                costsCol = [self.costs[i, j] for i in remainingRows]
                if len(costsCol) >= 2:
                    sortedCosts = sorted(costsCol)
                    colPenalties[j] = sortedCosts[1] - sortedCosts[0]
                else:
                    colPenalties[j] = costsCol[0]

            # find the maximum penalty
            maxRow = max(rowPenalties.items(), key=lambda x: x[1])
            maxCol = max(colPenalties.items(), key=lambda x: x[1])

            if maxRow[1] >= maxCol[1]:
                i = maxRow[0]
                # choose the cheapest column in row i among remainingCols
                j = min(remainingCols, key=lambda jj: self.costs[i, jj])
            else:
                j = maxCol[0]
                i = min(remainingRows, key=lambda ii: self.costs[ii, j])

            qty = min(supplies[i], demands[j])
            allocation[i, j] = qty
            supplies[i] -= qty
            demands[j] -= qty

            if supplies[i] <= self.tol:
                remainingRows.discard(i)
            if demands[j] <= self.tol:
                remainingCols.discard(j)

        self.allocation = allocation
        self._makeNonDegenerateBasis()

    def _basicPositions(self):
        """Return list of (i,j) indices that are basic (i.e., allocation not nan)."""
        return [(i, j) for i in range(self.m) for j in range(self.n) if not np.isnan(self.allocation[i, j])]

    def _makeNonDegenerateBasis(self):
        """
        Ensure at least m + n - 1 basic cells by inserting zero-valued basics into empty cells
        (choose cheapest cost empty cells).
        """
        basicPositions = self._basicPositions()
        needed = self.m + self.n - 1 - len(basicPositions)
        if needed <= 0:
            return
        freeCells = [(i, j) for i in range(self.m) for j in range(self.n) if np.isnan(self.allocation[i, j])]
        # sort free cells by cost (prefer low cost to be added as zero basics)
        freeCellsSorted = sorted(freeCells, key=lambda cell: self.costs[cell])
        for k in range(needed):
            i, j = freeCellsSorted[k]
            self.allocation[i, j] = 0.0

    def computePotentials(self):
        """
        Compute potentials u (for rows) and v (for cols) satisfying u_i + v_j = c_ij for all basic cells.
        Returns u (length m) and v (length n).
        """
        basic = self._basicPositions()
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        # fix reference u[0] = 0
        u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for (i, j) in basic:
                if np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = self.costs[i, j] - v[j]
                    changed = True
                elif not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = self.costs[i, j] - u[i]
                    changed = True
        # fill any remaining NaNs with zeros (disconnected piece because of degeneracy)
        u = np.where(np.isnan(u), 0.0, u)
        v = np.where(np.isnan(v), 0.0, v)
        return u, v

    def computeDeltas(self, u, v):
        """Compute reduced costs (deltas) for all cells: delta_ij = c_ij - (u_i + v_j)."""
        return self.costs - (u.reshape(-1, 1) + v.reshape(1, -1))

    def isOptimal(self, deltas):
        """For a minimization problem, optimality holds when all deltas >= 0."""
        return np.all(deltas >= -self.tol)

    def _findCycle(self, startCell):
        """
        Find an alternating closed cycle (horizontal/vertical alternation) that uses only basic cells,
        treating startCell as temporarily basic. Returns ordered list of cells forming the cycle
        (first cell repeated at end is possible).
        """
        basics = set(self._basicPositions())
        basics.add(startCell)

        # neighbors per row/col among basic cells
        rowNeighbors = {i: [j for (ii, j) in basics if ii == i] for i in range(self.m)}
        colNeighbors = {j: [i for (i, jj) in basics if jj == j] for j in range(self.n)}

        # depth-first search building alternating path
        def dfs(path, visited, expectRowMove):
            current = path[-1]
            i, j = current
            # if we have returned to startCell after at least 4 steps and the alternation finishes, accept
            if len(path) >= 4 and current == startCell and expectRowMove is None:
                return path[:]
            if expectRowMove is None:
                # initial step: can move either along row or along column
                for jj in rowNeighbors.get(i, []):
                    if jj == j: 
                        continue
                    nxt = (i, jj)
                    if len(path) == 1 or nxt != path[-2]:
                        res = dfs(path + [nxt], visited | {nxt}, False)
                        if res:
                            return res
                for ii in colNeighbors.get(j, []):
                    if ii == i: 
                        continue
                    nxt = (ii, j)
                    if len(path) == 1 or nxt != path[-2]:
                        res = dfs(path + [nxt], visited | {nxt}, True)
                        if res:
                            return res
                return None
            if expectRowMove:
                # must move within same row
                for jj in rowNeighbors.get(i, []):
                    if jj == j:
                        continue
                    nxt = (i, jj)
                    if nxt == startCell and len(path) >= 4:
                        return path + [nxt]
                    if nxt not in visited:
                        res = dfs(path + [nxt], visited | {nxt}, False)
                        if res:
                            return res
            else:
                # must move within same column
                for ii in colNeighbors.get(j, []):
                    if ii == i:
                        continue
                    nxt = (ii, j)
                    if nxt == startCell and len(path) >= 4:
                        return path + [nxt]
                    if nxt not in visited:
                        res = dfs(path + [nxt], visited | {nxt}, True)
                        if res:
                            return res
            return None

        cycle = dfs([startCell], {startCell}, None)
        # if cycle ends with startCell appended, keep it; otherwise None
        return cycle

    def _updateAllocationsAlongCycle(self, cycle):
        """
        Given a cycle (list of cells, possibly ending with the start again),
        apply +/- alternation: '+' on first, '-' on second, etc.
        Find theta = minimum allocation on '-' positions and update allocations.
        Remove a zero basic if basis size exceeds m+n-1.
        """
        if not cycle:
            raise RuntimeError("Cycle is None when attempting to update allocations.")
        # compress if last equals first
        if cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        # assign signs
        signs = {}
        for idx, cell in enumerate(cycle):
            signs[cell] = '+' if idx % 2 == 0 else '-'
        # compute theta = min allocation among '-' cells (treat nan as 0)
        minusCells = [cell for cell, s in signs.items() if s == '-']
        minusVals = []
        for (i, j) in minusCells:
            val = self.allocation[i, j]
            if np.isnan(val):
                val = 0.0
            minusVals.append(val)
        if not minusVals:
            theta = 0.0
        else:
            theta = min(minusVals)
        # update cells
        for cell, s in signs.items():
            i, j = cell
            oldVal = 0.0 if np.isnan(self.allocation[i, j]) else self.allocation[i, j]
            if s == '+':
                newVal = oldVal + theta
            else:
                newVal = oldVal - theta
            # keep small values as exact zero (to allow degeneracy handling)
            if abs(newVal) <= self.tol:
                newVal = 0.0
            self.allocation[i, j] = newVal
        # If we now have too many basics, remove one zero-basic (not ideal cell) to keep basis size m+n-1
        basicPositions = self._basicPositions()
        if len(basicPositions) > (self.m + self.n - 1):
            zeroBasics = [cell for cell in basicPositions if abs(self.allocation[cell]) <= self.tol]
            if zeroBasics:
                # remove the first zero-basic encountered
                cellToRemove = zeroBasics[0]
                self.allocation[cellToRemove] = np.nan

    def solveByPotentials(self, maxIterations=1000, verbose=False):
        """
        Solve the transport problem using the method of potentials.
        If allocation is empty, automatically uses Vogel's method for initial solution.

        Returns:
          allocation matrix, totalCost (float), iterationsUsed (int)
        """
        # Ensure initial solution exists
        if np.all(np.isnan(self.allocation)):
            self.vogelInitialSolution()

        iterCount = 0
        while iterCount < maxIterations:
            iterCount += 1
            u, v = self.computePotentials()
            deltas = self.computeDeltas(u, v)
            if self.isOptimal(deltas):
                if verbose:
                    print(f"Optimal found at iteration {iterCount}")
                break
            # pick most negative delta (most negative = best improvement)
            i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
            if deltas[i, j] >= -self.tol:
                break  # numerical safety
            # find improvement cycle with (i,j) as entering cell
            cycle = self._findCycle((i, j))
            if not cycle:
                raise RuntimeError("Unable to find improvement cycle. Degeneracy or basis issue.")
            self._updateAllocationsAlongCycle(cycle)
            # ensure non-degenerate basis for next iteration
            self._makeNonDegenerateBasis()
            if verbose:
                print(f"Iteration {iterCount}: entered cell {(i,j)} delta={deltas[i,j]:.6f}, cost={self.totalCost():.6f}")
        return self.allocation, self.totalCost(), iterCount

    def totalCost(self):
        """Compute total cost corresponding to current allocation (nan treated as zero)."""
        alloc = np.where(np.isnan(self.allocation), 0.0, self.allocation)
        return float(np.sum(alloc * self.costs))

    def printSolution(self):
        """Print allocation matrix and total cost."""
        print("Allocation (nan means empty):")
        print(self.allocation)
        print("Total cost:", self.totalCost())



def readTransportCSV(filePath):
    """
    Reads a transport problem from a CSV file.
    
    Expected format:
        - First row: supplies
        - Second row: demands
        - Next len(supplies) rows: cost matrix (each row corresponds to a supplier)
    
    Returns:
        costs   -> 2D numpy array (m x n)
        supplies -> 1D numpy array (length m)
        demands  -> 1D numpy array (length n)
    """
    rows = []
    with open(filePath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Strip whitespace, ignore empty cells
            cleanRow = [cell.strip() for cell in row if cell.strip() != '']
            if cleanRow:
                try:
                    rows.append([float(x) for x in cleanRow])
                except ValueError as e:
                    raise ValueError(f"Non-numeric data found in file {filePath}: {row}") from e

    if len(rows) < 3:
        raise ValueError(
            f"CSV must contain at least 3 rows (supplies, demands, and cost rows). Found {len(rows)} rows."
        )

    supplies = np.array(rows[0])
    demands = np.array(rows[1])
    costRows = rows[2:]

    m = len(supplies)
    n = len(demands)

    if len(costRows) != m:
        raise ValueError(
            f"Number of cost rows ({len(costRows)}) must equal number of supplies ({m})."
        )
    for r in costRows:
        if len(r) != n:
            raise ValueError(
                f"Each cost row must have {n} elements (same as number of demands). Row: {r}"
            )

    costs = np.array(costRows)
    return costs, supplies, demands

if __name__ == "__main__":
  for file in ["res/example1.csv","res/example2.csv","res/example3.csv"]:
    print(f"--------------------{file}-------------------------------")
    costs,supplies,demands = readTransportCSV(file)
    print("Supplies:")
    print(supplies)
    print("Demands:")
    print(demands)
    print("Costs: ")
    print(costs)
    

    tp = TransportProblem(costs, supplies, demands)
    allocation, totalcost, iterations = tp.solveByPotentials(verbose=False)
    tp.printSolution()
    print("iterations used:", iterations)
    print(f"--------------------{file}-------------------------------")