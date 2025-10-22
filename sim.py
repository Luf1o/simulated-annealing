from functions import ackley,beale,mvfBohachevsky2,mvfBohachevsky1,booth,mvfBoxBetts,mvfColville,mvfEasom,mvfLangermann,mvfKatsuura,mvfKowalik,mvfHosaki,mvfHansen
from functions import mvfBranin1,mvfBranin2,mvfCamel3,mvfCamel6,mvfChichinadze,mvfCola,mvfCorona,mvfEggHolder,mvfExp2,mvfHolzman,mvfHyperellipsoid,mvfHimmelblau,mvfHartmann3,mvfHartmann6
import random
import math

functions = {
    "Ackley": (ackley, [(-5, 5), (-5, 5)]),
    "Beale": (beale, [(-4.5, 4.5), (-4.5, 4.5)]),
    "Booth": (booth, [(-10, 10), (-10, 10)]),
    "Bohachevsky1": (mvfBohachevsky1, [(-50, 50), (-50, 50)]),
    "Bohachevsky2": (mvfBohachevsky2, [(-50, 50), (-50, 50)]),
    "BoxBetts": (mvfBoxBetts, [(-10, 10), (-10, 10), (-10, 10)]),
    "Branin1": (mvfBranin1, [(-5, 10), (0, 15)]),
    "Branin2": (mvfBranin2, [(-5, 10), (0, 15)]),
    "Camel3": (mvfCamel3, [(-5, 5), (-5, 5)]),
    "Camel6": (mvfCamel6, [(-5, 5), (-5, 5), (-5, 5)]),
    "Chichinadze": (mvfChichinadze, [(-30, 30), (-30, 30)]),
    "Cola": (mvfCola, [(-10, 10), (-10, 10)]), # Assuming n=2
    "Colville": (mvfColville, [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]),
    "Corona": (mvfCorona, [(-5, 5), (-5, 5)]),
    "Easom": (mvfEasom, [(-10, 10), (-10, 10)]),
    "EggHolder": (mvfEggHolder, [(-512, 512), (-512, 512)]),
    "Exp2": (mvfExp2, [(-5, 5), (-5, 5)]),
    "Hansen": (mvfHansen, [(-10, 10), (-10, 10)]),
    "Hartmann3": (mvfHartmann3, [(0, 1), (0, 1), (0, 1)]),
    "Hartmann6": (mvfHartmann6, [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]),
    "Himmelblau": (mvfHimmelblau, [(-5, 5), (-5, 5)]),
    "Hyperellipsoid": (mvfHyperellipsoid, [(-5, 5), (-5, 5)]), # Assuming n=2
    "Holzman": (mvfHolzman, [(-5, 5), (-5, 5)]),
    "Hosaki": (mvfHosaki, [(0, 5), (0, 6)]),
    "Kowalik": (mvfKowalik, [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]),
    "Katsuura": (mvfKatsuura, [(-100, 100), (-100, 100)]), # Assuming n=2
    "Langermann": (mvfLangermann, [(0, 10), (0, 10)])
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
    for name, (func,bounds) in functions.items():
        best_sol , best_val = simulated_annealing(func,bounds,100,0.95,1000)
        for i in range(len(best_sol)):
            best_sol[i] = round(best_sol[i],4)
        print(f'{name}: \nbest_value={best_val:.6f}\tbest_solution={best_sol}')


if __name__ == '__main__':
    res = input("Run Annealing? Yes/ No")
    res = res.lower()
    if res == 'yes':
        main()
    else:
        print(len(functions.keys()))
        print("exit")