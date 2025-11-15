import numpy as np
import pandas as pd

class TransportProblem:
    def __init__(self, costs, supplies, demands):
        self.costs = np.array(costs, dtype=float)
        self.supplies = np.array(supplies, dtype=float)
        self.demands = np.array(demands, dtype=float)
        self.m = len(supplies)
        self.n = len(demands)
        
        self._balance_problem()
        
        self.allocation = None
        self.total_cost = 0
        self.iterations = 0
    
    def _balance_problem(self):
        """Балансировка транспортной задачи"""
        total_supply = np.sum(self.supplies)
        total_demand = np.sum(self.demands)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                self.demands = np.append(self.demands, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.m)])
                self.n += 1
                print(f"Добавлен фиктивный потребитель со спросом {total_supply - total_demand:.1f}")
            else:
                self.supplies = np.append(self.supplies, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.n)])
                self.m += 1
                print(f"Добавлен фиктивный поставщик с предложением {total_demand - total_supply:.1f}")
    
    def _north_west_corner(self):
        """Метод северо-западного угла для начального решения"""
        allocation = np.zeros((self.m, self.n))
        i, j = 0, 0
        supply_remaining = self.supplies.copy()
        demand_remaining = self.demands.copy()
        
        while i < self.m and j < self.n:
            if supply_remaining[i] < 1e-10:
                i += 1
                continue
            if demand_remaining[j] < 1e-10:
                j += 1
                continue
                
            amount = min(supply_remaining[i], demand_remaining[j])
            allocation[i, j] = amount
            supply_remaining[i] -= amount
            demand_remaining[j] -= amount
            
            if supply_remaining[i] < 1e-10:
                i += 1
            else:
                j += 1
                
        return allocation
    
    def _get_basis_indices(self, allocation):
        """Получить индексы базисных переменных"""
        basis = []
        for i in range(self.m):
            for j in range(self.n):
                if allocation[i, j] > 1e-10:
                    basis.append((i, j))
        return basis
    
    def _calculate_potentials(self, basis):
        """Вычисление потенциалов"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0
        
        changed = True
        max_iter = self.m + self.n
        
        while changed and max_iter > 0:
            max_iter -= 1
            changed = False
            
            for i, j in basis:
                if not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = self.costs[i, j] - u[i]
                    changed = True
                elif np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = self.costs[i, j] - v[j]
                    changed = True
        
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)
        
        return u, v
    
    def _find_entering_variable(self, u, v, basis_set):
        """Найти вводимую переменную с отрицательной оценкой"""
        enter_var = None
        min_delta = 0
        
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in basis_set:
                    delta_ij = self.costs[i, j] - u[i] - v[j]
                    if delta_ij < min_delta - 1e-10:
                        min_delta = delta_ij
                        enter_var = (i, j)
        
        return enter_var, min_delta
    
    def _find_cycle(self, allocation, enter_var):
        """Найти цикл пересчета для вводимой переменной"""
        basis = self._get_basis_indices(allocation)
        basis_set = set(basis)
        basis_set.add(enter_var)
        
        from collections import deque
        
        start_i, start_j = enter_var
        queue = deque()
        queue.append((start_i, start_j, [(start_i, start_j)], True))
        
        visited = set()
        visited.add((start_i, start_j, True))
        
        while queue:
            i, j, path, is_row_move = queue.popleft()
            
            if is_row_move:
                for next_j in range(self.n):
                    if (i, next_j) in basis_set and (i, next_j) not in [p[:2] for p in path]:
                        if next_j == start_j and len(path) >= 3:
                            return path + [(i, next_j)]
                        if (i, next_j, False) not in visited:
                            visited.add((i, next_j, False))
                            queue.append((i, next_j, path + [(i, next_j)], False))
            else:
                for next_i in range(self.m):
                    if (next_i, j) in basis_set and (next_i, j) not in [p[:2] for p in path]:
                        if next_i == start_i and len(path) >= 3:
                            return path + [(next_i, j)]
                        if (next_i, j, True) not in visited:
                            visited.add((next_i, j, True))
                            queue.append((next_i, j, path + [(next_i, j)], True))
        
        return None
    
    def _update_allocation(self, allocation, cycle):
        """Обновить распределение по найденному циклу"""
        plus_cells = cycle[0::2]
        minus_cells = cycle[1::2]
        
        theta = float('inf')
        for i, j in minus_cells:
            if allocation[i, j] < theta:
                theta = allocation[i, j]
        
        new_allocation = allocation.copy()
        for i, j in plus_cells:
            new_allocation[i, j] += theta
        for i, j in minus_cells:
            new_allocation[i, j] -= theta
        
        return new_allocation
    
    def solveByPotentials(self, verbose=False):
        """Решение транспортной задачи методом потенциалов"""
        allocation = self._north_west_corner()
        self.iterations = 0
        
        if verbose:
            print("Начальное решение:")
            print(allocation)
            initial_cost = np.sum(self.costs * allocation)
            print(f"Начальная стоимость: {initial_cost:.2f}")
        
        max_iterations = 50
        
        while self.iterations < max_iterations:
            self.iterations += 1
            
            basis = self._get_basis_indices(allocation)
            basis_set = set(basis)
            
            while len(basis_set) < self.m + self.n - 1:
                added = False
                for i in range(self.m):
                    for j in range(self.n):
                        if (i, j) not in basis_set and allocation[i, j] == 0:
                            basis_set.add((i, j))
                            if len(basis_set) == self.m + self.n - 1:
                                added = True
                                break
                    if added:
                        break
            
            u, v = self._calculate_potentials(list(basis_set))
            
            enter_var, delta = self._find_entering_variable(u, v, basis_set)
            
            if verbose:
                print(f"\nИтерация {self.iterations}:")
                print(f"u = {u}")
                print(f"v = {v}")
                if enter_var:
                    print(f"Вводимая переменная: ({enter_var[0]}, {enter_var[1]}), Δ = {delta:.6f}")
            
            if enter_var is None:
                break
            
            cycle = self._find_cycle(allocation, enter_var)
            
            if cycle is None:
                if verbose:
                    print("Цикл не найден!")
                break
            
            if verbose:
                print(f"Цикл: {cycle}")
            
            allocation = self._update_allocation(allocation, cycle)
            
            if verbose:
                current_cost = np.sum(self.costs * allocation)
                print(f"Новое распределение:")
                print(allocation)
                print(f"Стоимость: {current_cost:.2f}")
        
        self.allocation = allocation
        self.total_cost = np.sum(self.costs * allocation)
        
        return allocation, self.total_cost, self.iterations
    
    def printSolution(self):
        """Вывод решения"""
        if self.allocation is None:
            print("Задача не решена!")
            return
        
        print("\nОптимальное решение:")
        print("Распределение:")
        for i in range(self.m):
            row = []
            for j in range(self.n):
                if abs(self.allocation[i, j]) > 1e-10:
                    row.append(f"{self.allocation[i, j]:6.1f}")
                else:
                    row.append("   nan")
            print("  ".join(row))
        
        print(f"\nОбщая стоимость: {self.total_cost:.1f}")


def readTransportCSV(filename):
    """Чтение транспортной задачи из CSV файла
    Формат: 
    - первая строка: supplies
    - вторая строка: demands  
    - остальные строки: costs
    """
    try:
        df = pd.read_csv(filename, header=None)
        
        df = df.fillna(0)
        
        supplies = df.iloc[0, :].values.astype(float)
        supplies = supplies[~np.isclose(supplies, 0)]
        
        demands = df.iloc[1, :].values.astype(float)
        demands = demands[~np.isclose(demands, 0)]
        
        costs = df.iloc[2:2+len(supplies), :len(demands)].values.astype(float)
        
        return costs, supplies, demands
        
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")

if __name__ == "__main__":
    files = ["res/example1.csv", "res/example2.csv", "res/example3.csv"]
    
    for file in files:
        print(f"\n{'='*60}")
        print(f"ОБРАБОТКА ФАЙЛА: {file}")
        print(f"{'='*60}")
        
        try:
            costs, supplies, demands = readTransportCSV(file)
            print("Supplies:", supplies)
            print("Demands:", demands)
            print("Costs:")
            print(costs)
            
            tp = TransportProblem(costs, supplies, demands)
            allocation, totalcost, iterations = tp.solveByPotentials(verbose=False)
            tp.printSolution()
            print("iterations used:", iterations)
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")
