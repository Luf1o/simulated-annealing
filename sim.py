import functions as mvf
import pandas as pd
import random
import math


def make_bounds(dim, low, high):
    return [(low, high)] * dim
functions = {
    "Ackley": (mvf.Ackley, [(-5.12, 5.12), (-5.12, 5.12)]),
    "Beale": (mvf.Beale, [(-4.5, 4.5), (-4.5, 4.5)]),
    "Booth": (mvf.Booth, [(-10, 10), (-10, 10)]),
    "Bohachevsky1": (mvf.Bohachevsky1, [(-100, 100), (-100, 100)]),
    "Bohachevsky2": (mvf.Bohachevsky2, [(-100, 100), (-100, 100)]),
    "BoxBetts": (mvf.BoxBetts, [(0.9, 1.2), (9, 11.2), (0.9, 1.2)]),
    "Branin1": (mvf.Branin1, [(-5, 10), (0, 15)]),
    "Branin2": (mvf.Branin2, [(-5, 10), (0, 15)]),
    "Camel3": (mvf.Camel3, [(-5, 5), (-5, 5)]),
    "Camel6": (mvf.Camel6, [(-3, 3), (-2, 2)]),
    "Chichinadze": (mvf.Chichinadze, [(-30, 30), (-30, 30)]),
    "Cola": (mvf.Cola, [(-10, 10), (-10, 10), (-10, 10)]),
    "Colville": (mvf.Colville, [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]),
    "Corona": (mvf.Corona, [(0.001, 5), (0.001, 5)]),  # Avoid origin
   
    "Easom": (mvf.Easom, [(-100, 100), (-100, 100)]),
    "EggHolder": (mvf.EggHolder, [(-512, 512), (-512, 512)]),
    "Exp2": (mvf.Exp2, [(-1, 1), (-1, 1)]),  # Limited range to prevent overflow
    "Hansen": (mvf.Hansen, [(-10, 10), (-10, 10)]),
    "Hartmann3": (mvf.Hartmann3, [(0, 1), (0, 1), (0, 1)]),
    "Hartmann6": (mvf.Hartmann6, [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]),
    "Himmelblau": (mvf.Himmelblau, [(-5, 5), (-5, 5)]),
    "Holzman": (mvf.Holzman, [(-10, 10), (-10, 10)]),
    "Hosaki": (mvf.Hosaki, [(0, 5), (0, 6)]),
    "Hyperellipsoid": (mvf.Hyperellipsoid, [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]),

    "Katsuura": (mvf.Katsuura, [(-100, 100), (-100, 100)]),
    "Kowalik": (mvf.Kowalik, [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]),
    "Langermann": (mvf.Langermann, [(0, 10), (0, 10)]),
    "LennardJones": (mvf.LennardJones, [(-2, 2), (-2, 2)]),
    "Leon": (mvf.Leon, [(-1.2, 1.2), (-1.2, 1.2)]),
    "Levy": (mvf.Levy, [(-10, 10), (-10, 10), (-10, 10)]),
    
    "Matyas": (mvf.Matyas, [(-10, 10), (-10, 10)]),
    "MaxFold": (mvf.MaxFold, [(-10, 10), (-10, 10), (-10, 10)]),
    "McCormick": (mvf.McCormick, [(-1.5, 4), (-3, 4)]),
    "Michalewicz": (mvf.Michalewicz, [(0, 3.14159), (0, 3.14159)]),
    "Multimod": (mvf.Multimod, [(-10, 10), (-10, 10)]),
    
    "Paviani": (mvf.Paviani, [
        (2.001, 9.999), (2.001, 9.999), (2.001, 9.999), (2.001, 9.999), (2.001, 9.999),
        (2.001, 9.999), (2.001, 9.999), (2.001, 9.999), (2.001, 9.999), (2.001, 9.999)
    ]),
    "Powell": (mvf.Powell, [(-4, 5), (-4, 5), (-4, 5), (-4, 5)]),
    "Price": (mvf.Price, [(0.01, 10), (0.01, 10), (0.01, 10)]),
    
    "Quartic": (mvf.Quartic, [(-1.28, 1.28), (-1.28, 1.28), (-1.28, 1.28)]),
    "QuarticNoise": (mvf.QuarticNoise, [(-1.28, 1.28), (-1.28, 1.28)]),
    "Rana": (mvf.Rana, [(-500, 500), (-500, 500)]),
    "Rastrigin": (mvf.Rastrigin, [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]),
    "Rosenbrock": (mvf.Rosenbrock, [(-5, 10), (-5, 10), (-5, 10)]),
    "RosenbrockGeneralized": (mvf.RosenbrockGeneralized, [(-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048)]),

    "Schaffer": (mvf.Schaffer, [(-100, 100), (-100, 100)]),
    "Schwefel": (mvf.Schwefel, [(-500, 500), (-500, 500), (-500, 500)]),
    "ShekelFoxholes": (mvf.ShekelFoxholes, [(-65.536, 65.536), (-65.536, 65.536)]),
    "Shubert": (mvf.Shubert, [(-10, 10), (-10, 10)]),
    "Sphere": (mvf.Sphere, [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]),
    "Step": (mvf.Step, [(-100, 100), (-100, 100), (-100, 100)]),
    "StretchedV": (mvf.StretchedV, [(-10, 10), (-10, 10), (-10, 10)]),
    "SumSquares": (mvf.SumSquares, [(-10, 10), (-10, 10), (-10, 10)]),
    
    "Trecanni": (mvf.Trecanni, [(-5, 5), (-5, 5)]),
    "Trefethen4": (mvf.Trefethen4, [(-6.5, 6.5), (-4.5, 4.5)]),
    "Watson": (mvf.Watson, [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]),
    "XOR": (mvf.XOR, [(-6.28, 6.28), (-6.28, 6.28)]),
    "Zettl": (mvf.Zettl, [(-5, 10), (-5, 10)]),
    "Zimmerman": (mvf.Zimmerman, [(-10, 10), (-10, 10)]),
}

def simulated_annealing(objective_function,bounds,initial_temperature,cooling_rate,n_iterations):
    current_solution = [random.uniform(bounds[i][0],bounds[i][1]) for i in range(len(bounds))]
    best_solution = list(current_solution)
    best_value = objective_function(best_solution)
    neighor_solution = None
    
    for i in range(n_iterations):
        temperature = initial_temperature * cooling_rate ** i
        neighor_solution = list(current_solution)
        
        for j in range(len(neighor_solution)):
            perturbation = random.uniform(-temperature,temperature)
            neighor_solution[j]+=perturbation
            neighor_solution[j] = max(bounds[j][0],neighor_solution[j])
            neighor_solution[j] = min(bounds[j][1],neighor_solution[j])
        
        neighbor_value  = objective_function(neighor_solution)
        objective_value = objective_function(current_solution)
        differene = neighbor_value - objective_value

        if neighbor_value <  objective_value or random.random() < math.exp(-differene/(temperature + 1e-9)):
            current_solution = list(neighor_solution)
            current_value = objective_function(current_solution)
            if current_value < best_value:
                best_solution = list(current_solution)
                best_value = current_value
    return best_solution, best_value
    
def main():
    df = pd.DataFrame(columns=['Function Name','Best Value','Best Solution'])
    
    for name, (func,bounds) in functions.items():
        best_sol , best_val = simulated_annealing(func,bounds,100,0.95,3000)
        for i in range(len(best_sol)):
            best_sol[i] = round(best_sol[i],4)
        #print(f'{name}: \nbest_value={best_val:.6f}\tbest_solution={best_sol}')
        df.loc[len(df)] = [name,round(best_val,6),best_sol]
    
    df.to_csv('simulated.csv')

if __name__ == '__main__':
    res = input("Run Annealing? Yes/ No : ")
    res = res.lower()
    if res == 'yes':
        main()
    else:
        print(len(functions.keys()))
        print("exit")