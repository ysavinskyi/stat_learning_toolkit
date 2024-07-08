from ortools.sat.python import cp_model


class LinearProgrammingModel:
    def __init__(self, min_value, max_value, number_of_x):
        self._number_of_x = number_of_x
        self._min_value = min_value
        self._max_value = max_value
        self._model = cp_model.CpModel()
        self._variables = {}

    def set_constraints(self, target_func, constraint_list):
        for i in range(self._number_of_x):
            var_name = f'X{i + 1}'
            self._variables[var_name] = self._model.NewIntVar(self._min_value, self._max_value, var_name)

        for constraint in constraint_list:
            try:
                self._model.Add(eval(constraint, {}, self._variables))
            except Exception as e:
                print(f"Error adding constraint {constraint}: {e}")

        func, condition = target_func
        try:
            if 'minimize' in condition.lower():
                self._model.Minimize(eval(func, {}, self._variables))
            elif 'maximize' in condition.lower():
                self._model.Maximize(eval(func, {}, self._variables))
        except Exception as e:
            print(f"Error setting objective function {func}: {e}")

    def get_solutions(self):
        solver = cp_model.CpSolver()
        print("Solving the model...")
        status = solver.Solve(self._model)
        print("Solver status:", status)

        if status == cp_model.OPTIMAL:
            print(f'Оптимальне значення цільової функції: {solver.ObjectiveValue()}')
            for var_name, var in self._variables.items():
                print(f'{var_name} = {solver.Value(var)}')
        else:
            print('Оптимальне рішення не знайдено.')