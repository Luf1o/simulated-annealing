import numpy as np

flags = {
    "price": 0
}


"""Ackley Function"""
def Ackley(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    n = len(x)
    if n == 0:
        raise ValueError("Input array cannot be empty")
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e


"""Beale has only 2 variable test function"""
def Beale(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Beale function requires exactly 2 variables")
    
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3


def Booth(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Booth function requires exactly 2 variables")
    
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def Bohachevsky1(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Bohachevsky1 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 + 2.0*x1**2 - 0.3*np.cos(3*np.pi*x0) 
            - 0.4*np.cos(4*np.pi*x1) + 0.7)

    
def Bohachevsky2(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Bohachevsky2 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 + 2.0*x1**2 
            - 0.3*np.cos(3*np.pi*x0)*np.cos(4*np.pi*x1) + 0.3)

    
def BoxBetts(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 3:
        raise ValueError("BoxBetts function requires exactly 3 variables")
    
    x1, x2, x3 = x[0], x[1], x[2]
    total = 0
    for i in range(1, 11):
        term = (np.exp(-0.1*(i+1)*x1) - np.exp(-0.1*(i+1)*x2)
                - (np.exp(-0.1*(i+1)) - np.exp(-(i+1)))
                * x3)
        total += term**2
    return total


def Branin1(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Branin1 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    term1 = x1 - (5.1*x0**2)/(4*np.pi**2) + 5*x0/np.pi - 6
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x0) + 10
    return term1**2 + term2


def Branin2(x):
    """
    Modified Branin2 Function
    Typically evaluated on [-5, 10] × [0, 15]
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Branin2 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    term1 = 1.0 - 2.0*x1 + np.sin(4.0*np.pi*x1)/20.0 - x0
    term2 = x1 - np.sin(2.0*np.pi*x0)/2.0
    return term1**2 + term2**2


def Camel3(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Camel3 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 2*x0**2 - 1.05*x0**4 + x0**6/6 + x0*x1 + x1**2


def Camel6(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Camel6 function requires exactly 2 variables (was 3)")
    
    x0, x1 = x[0], x[1]
    t0 = (4 - 2.1*x0**2 + x0**4/3)*x0**2
    t1 = x0*x1
    t2 = (-4 + 4*x1**2)*x1**2
    return t0 + t1 + t2


def Chichinadze(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Chichinadze function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 - 12*x0 + 11 + 10*np.cos(np.pi*x0/2) 
            + 8*np.sin(5*np.pi*x0) 
            - np.sqrt(0.2)*np.exp(-0.5*(x1 - 0.5)**2))


def Cola(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    n = len(x)
    if n < 2:
        raise ValueError("Cola function requires at least 2 variables")
    
    sum1 = sum([(x[i] - x[i-1])**2 for i in range(1, n)])
    sum2 = sum([(x[i] - 1)**2 for i in range(n)])
    return sum1 + sum2


def Colville(x):
    """Colville is a 4-d function"""
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 4:
        raise ValueError("Colville function requires exactly 4 variables")
    
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (100*(x2 - x1**2)**2 + (1 - x1)**2 +
            90*(x4 - x3**2)**2 + (1 - x3)**2 +
            10.1*((x2 - 1)**2 + (x4 - 1)**2) +
            19.8*(x2 - 1)*(x4 - 1))


def Corona(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Corona function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    r_sq = x1**2 + x2**2
    if r_sq < 1e-10:
        raise ValueError("Corona function undefined at origin")
    
    return (1 - 1/r_sq)**2 + 0.1*(r_sq - 1)**2


def Easom(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Easom function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return -np.cos(x1)*np.cos(x2)*np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))


def EggHolder(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("EggHolder function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    # Handle potential negative values in sqrt
    term1 = -(x1 + 47)*np.sin(np.sqrt(np.abs(x1 + x0/2 + 47)))
    term2 = -x0*np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    return term1 + term2


def Exp2(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Exp2 function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    exponent = x1**2 + x2**2
    # Prevent overflow
    if exponent > 700:
        return np.inf
    return np.exp(exponent)


def Hansen(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Hansen function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    term1 = sum([(i+1)*np.cos(i*x1 + i + 1) for i in range(5)])
    term2 = sum([(i+1)*np.cos((i+2)*x2 + i + 1) for i in range(5)])
    return term1 * term2


def Hartmann3(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 3:
        raise ValueError("Hartmann3 function requires exactly 3 variables")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35]
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def Hartmann6(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 6:
        raise ValueError("Hartmann6 function requires exactly 6 variables")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def Himmelblau(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Himmelblau function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def Hyperellipsoid(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    
    return np.sum(np.arange(1, len(x) + 1) * x**2)


def Holzman(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Holzman function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return (x1**2 + x2**2 - np.cos(18*x1) - np.cos(18*x2))**2


def Hosaki(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Hosaki function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    # Prevent overflow in exp
    if x2 > 700:
        return 0  # exp(-x2) approaches 0
    
    return ((1 - 8*x1 + 7*x1**2 - (7/3)*x1**3 + 0.25*x1**4) 
            * x2**2 * np.exp(-x2))


def Kowalik(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 4:
        raise ValueError("Kowalik function requires exactly 4 variables")
    
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    a = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
                  0.0456, 0.0342, 0.0323, 0.0235])
    b = 1 / np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    
    total = 0.0
    for i in range(10):
        denom = b[i]**2 + b[i]*x3 + x4
        if np.abs(denom) < 1e-10:
            return np.inf  # Division by zero protection
        numerator = a[i] - ((x1*(b[i]**2 + b[i]*x2)) / denom)
        total += numerator**2
    return total


def Katsuura(x):
    """
    Katsuura Function
    Domain: -100 ≤ xi ≤ 100
    Global minimum: f(0,...,0) = 1
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    prod = 1
    for i in range(d):
        term = 0
        for j in range(1, 33):
            term += np.abs(2**j * x[i] - np.round(2**j * x[i])) / 2**j
        prod *= (1 + (i+1)*term)**(10/d**1.2)
    return (10 / d**2) * (prod - 1)


def Langermann(x):
    """
    Langermann Function
    Domain: 0 ≤ xi ≤ 10
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Langermann function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    m = 5
    a = np.array([3, 5, 2, 1, 7])
    b = np.array([5, 2, 1, 4, 9])
    c = np.array([1, 2, 5, 2, 3])

    result = 0
    for i in range(m):
        d_sq = (x1 - a[i])**2 + (x2 - b[i])**2
        result += c[i] * np.exp(-d_sq/np.pi) * np.cos(np.pi*d_sq)
    return result


def LennardJones(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("LennardJones function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    r_squared = x0**2 + x1**2 + 1e-12
    inv_r_squared = 1.0 / r_squared
    inv_r_6 = inv_r_squared**3
    inv_r_12 = inv_r_6**2
    
    return 4.0 * (inv_r_12 - inv_r_6)


def Leon(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Leon function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 100*(x1 - x0**3)**2 + (1.0 - x0)**2


def Levy(x):
    """
    Levy Function
    Domain: -10 ≤ xi ≤ 10
    Global minimum: f(1,...,1) = 0
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi*w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    
    if d == 1:
        return term1 + term3
    
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    
    return term1 + term2 + term3


def Matyas(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Matyas function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 0.26*(x0**2 + x1**2) - 0.48*x0*x1


def MaxFold(x):
    """Maximum of absolute values"""
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    return np.max(np.abs(x))


def McCormick(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("McCormick function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return np.sin(x0 + x1) + (x0 - x1)**2 - 1.5*x0 + 2.5*x1 + 1


def Michalewicz(x, m=10):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * (np.sin(i*x**2/np.pi))**(2*m))


def Multimod(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    
    abs_x = np.abs(x)
    return np.sum(abs_x) * np.prod(abs_x)


def Paviani(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 10:
        raise ValueError("Paviani function requires exactly 10 variables")
    
    # Domain check: 2 < xi < 10
    if np.any(x <= 2.0) or np.any(x >= 10.0):
        return np.inf
    
    term1 = np.sum(np.log(x - 2.0)**2 + np.log(10.0 - x)**2)
    term2 = np.prod(x)**0.2
    return term1 - term2

import numpy as np

"""Ackley Function"""
def ackley(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    n = len(x)
    if n == 0:
        raise ValueError("Input array cannot be empty")
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e


"""Beale has only 2 variable test function"""
def beale(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Beale function requires exactly 2 variables")
    
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3


def booth(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Booth function requires exactly 2 variables")
    
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def Bohachevsky1(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Bohachevsky1 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 + 2.0*x1**2 - 0.3*np.cos(3*np.pi*x0) 
            - 0.4*np.cos(4*np.pi*x1) + 0.7)

    
def Bohachevsky2(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Bohachevsky2 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 + 2.0*x1**2 
            - 0.3*np.cos(3*np.pi*x0)*np.cos(4*np.pi*x1) + 0.3)

    
def BoxBetts(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 3:
        raise ValueError("BoxBetts function requires exactly 3 variables")
    
    x1, x2, x3 = x[0], x[1], x[2]
    total = 0
    for i in range(1, 11):
        term = (np.exp(-0.1*(i+1)*x1) - np.exp(-0.1*(i+1)*x2)
                - (np.exp(-0.1*(i+1)) - np.exp(-(i+1)))
                * x3)
        total += term**2
    return total


def Branin1(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Branin1 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    term1 = x1 - (5.1*x0**2)/(4*np.pi**2) + 5*x0/np.pi - 6
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x0) + 10
    return term1**2 + term2


def Branin2(x):
    """
    Modified Branin2 Function
    Typically evaluated on [-5, 10] × [0, 15]
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Branin2 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    term1 = 1.0 - 2.0*x1 + np.sin(4.0*np.pi*x1)/20.0 - x0
    term2 = x1 - np.sin(2.0*np.pi*x0)/2.0
    return term1**2 + term2**2


def Camel3(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Camel3 function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 2*x0**2 - 1.05*x0**4 + x0**6/6 + x0*x1 + x1**2


def Camel6(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Camel6 function requires exactly 2 variables (was 3)")
    
    x0, x1 = x[0], x[1]
    t0 = (4 - 2.1*x0**2 + x0**4/3)*x0**2
    t1 = x0*x1
    t2 = (-4 + 4*x1**2)*x1**2
    return t0 + t1 + t2


def Chichinadze(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Chichinadze function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return (x0**2 - 12*x0 + 11 + 10*np.cos(np.pi*x0/2) 
            + 8*np.sin(5*np.pi*x0) 
            - np.sqrt(0.2)*np.exp(-0.5*(x1 - 0.5)**2))


def Cola(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    n = len(x)
    if n < 2:
        raise ValueError("Cola function requires at least 2 variables")
    
    sum1 = sum([(x[i] - x[i-1])**2 for i in range(1, n)])
    sum2 = sum([(x[i] - 1)**2 for i in range(n)])
    return sum1 + sum2


def Colville(x):
    """Colville is a 4-d function"""
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 4:
        raise ValueError("Colville function requires exactly 4 variables")
    
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (100*(x2 - x1**2)**2 + (1 - x1)**2 +
            90*(x4 - x3**2)**2 + (1 - x3)**2 +
            10.1*((x2 - 1)**2 + (x4 - 1)**2) +
            19.8*(x2 - 1)*(x4 - 1))


def Corona(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Corona function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    r_sq = x1**2 + x2**2
    if r_sq < 1e-10:
        raise ValueError("Corona function undefined at origin")
    
    return (1 - 1/r_sq)**2 + 0.1*(r_sq - 1)**2


def Easom(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Easom function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return -np.cos(x1)*np.cos(x2)*np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))


def EggHolder(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("EggHolder function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    # Handle potential negative values in sqrt
    term1 = -(x1 + 47)*np.sin(np.sqrt(np.abs(x1 + x0/2 + 47)))
    term2 = -x0*np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    return term1 + term2


def Exp2(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Exp2 function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    exponent = x1**2 + x2**2
    # Prevent overflow
    if exponent > 700:
        return np.inf
    return np.exp(exponent)


def Hansen(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Hansen function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    term1 = sum([(i+1)*np.cos(i*x1 + i + 1) for i in range(5)])
    term2 = sum([(i+1)*np.cos((i+2)*x2 + i + 1) for i in range(5)])
    return term1 * term2


def Hartmann3(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 3:
        raise ValueError("Hartmann3 function requires exactly 3 variables")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35]
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def Hartmann6(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 6:
        raise ValueError("Hartmann6 function requires exactly 6 variables")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def Himmelblau(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Himmelblau function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def Hyperellipsoid(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    
    return np.sum(np.arange(1, len(x) + 1) * x**2)


def Holzman(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Holzman function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    return (x1**2 + x2**2 - np.cos(18*x1) - np.cos(18*x2))**2


def Hosaki(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Hosaki function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    # Prevent overflow in exp
    if x2 > 700:
        return 0  # exp(-x2) approaches 0
    
    return ((1 - 8*x1 + 7*x1**2 - (7/3)*x1**3 + 0.25*x1**4) 
            * x2**2 * np.exp(-x2))


def Kowalik(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 4:
        raise ValueError("Kowalik function requires exactly 4 variables")
    
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    a = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
                  0.0456, 0.0342, 0.0323, 0.0235])
    b = 1 / np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    
    total = 0.0
    for i in range(10):
        denom = b[i]**2 + b[i]*x3 + x4
        if np.abs(denom) < 1e-10:
            return np.inf  # Division by zero protection
        numerator = a[i] - ((x1*(b[i]**2 + b[i]*x2)) / denom)
        total += numerator**2
    return total


def Katsuura(x):
    """
    Katsuura Function
    Domain: -100 ≤ xi ≤ 100
    Global minimum: f(0,...,0) = 1
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    prod = 1
    for i in range(d):
        term = 0
        for j in range(1, 33):
            term += np.abs(2**j * x[i] - np.round(2**j * x[i])) / 2**j
        prod *= (1 + (i+1)*term)**(10/d**1.2)
    return (10 / d**2) * (prod - 1)


def Langermann(x):
    """
    Langermann Function
    Domain: 0 ≤ xi ≤ 10
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Langermann function requires exactly 2 variables")
    
    x1, x2 = x[0], x[1]
    m = 5
    a = np.array([3, 5, 2, 1, 7])
    b = np.array([5, 2, 1, 4, 9])
    c = np.array([1, 2, 5, 2, 3])

    result = 0
    for i in range(m):
        d_sq = (x1 - a[i])**2 + (x2 - b[i])**2
        result += c[i] * np.exp(-d_sq/np.pi) * np.cos(np.pi*d_sq)
    return result


def LennardJones(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("LennardJones function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    r_squared = x0**2 + x1**2 + 1e-12
    inv_r_squared = 1.0 / r_squared
    inv_r_6 = inv_r_squared**3
    inv_r_12 = inv_r_6**2
    
    return 4.0 * (inv_r_12 - inv_r_6)


def Leon(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Leon function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 100*(x1 - x0**3)**2 + (1.0 - x0)**2


def Levy(x):
    """
    Levy Function
    Domain: -10 ≤ xi ≤ 10
    Global minimum: f(1,...,1) = 0
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi*w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    
    if d == 1:
        return term1 + term3
    
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    
    return term1 + term2 + term3


def Matyas(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("Matyas function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return 0.26*(x0**2 + x1**2) - 0.48*x0*x1


def MaxFold(x):
    """Maximum of absolute values"""
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    return np.max(np.abs(x))


def McCormick(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 2:
        raise ValueError("McCormick function requires exactly 2 variables")
    
    x0, x1 = x[0], x[1]
    return np.sin(x0 + x1) + (x0 - x1)**2 - 1.5*x0 + 2.5*x1 + 1


def Michalewicz(x, m=10):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * (np.sin(i*x**2/np.pi))**(2*m))


def Multimod(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    
    abs_x = np.abs(x)
    return np.sum(abs_x) * np.prod(abs_x)


def Paviani(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 10:
        raise ValueError("Paviani function requires exactly 10 variables")
    
    # Domain check: 2 < xi < 10
    if np.any(x <= 2.0) or np.any(x >= 10.0):
        return np.inf
    
    term1 = np.sum(np.log(x - 2.0)**2 + np.log(10.0 - x)**2)
    term2 = np.prod(x)**0.2
    return term1 - term2


def Powell(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d == 0:
        raise ValueError("Input array cannot be empty")
    if d % 4 != 0:
        raise ValueError(f"Powell function requires dimension to be a multiple of 4, got {d}")
    
    total = 0.0
    for i in range(0, d, 4):
        if i + 3 >= d:
            break
        term1 = (x[i] + 10*x[i+1])**2
        term2 = 5*(x[i+2] - x[i+3])**2
        term3 = (x[i+1] - 2*x[i+2])**4
        term4 = 10*(x[i] - x[i+3])**4
        total += term1 + term2 + term3 + term4
    
    return total


def Price(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    if len(x) != 3:
        raise ValueError("Price function requires exactly 3 variables")
    
    x1, x2, x3 = x[0], x[1], x[2]
    
    # Check for domain violations that cause log of negative numbers
    if x1 <= 0 or x2 <= 0:
        import warnings
        warnings.warn("Price function: x1 and x2 should be positive for log computation")
        return np.inf
    
    term1 = (2*x3 - np.log(x1))**2
    term2 = (x2 - np.log(x1))**2
    if flags["price"] < 1:
        print('Debug: clipping enabled due to RuntimeWarning: Overflow ')
        flags["price"] = 1
    term3 = (np.exp(np.clip(x2/x1, -100, 100)) - x3)**2

    
    return term1 + term2 + term3


def Quartic(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    i = np.arange(1, d + 1)
    return np.sum(i * x**4)


def QuarticNoise(x, noise_scale=0.1):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    base_value = Quartic(x)
    noise = np.random.normal(0, noise_scale)
    
    return base_value + noise


def Rana(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d < 2:
        raise ValueError("Rana function requires at least 2 variables")
    
    total = 0.0
    for i in range(d):
        x_i = x[i]
        x_next = x[(i + 1) % d]  # Wrap around for last element
        
        # Compute square root terms safely
        sqrt_term1 = np.sqrt(np.abs(x_next + x_i + 1))
        sqrt_term2 = np.sqrt(np.abs(x_next - x_i + 1))
        
        term1 = x_i * np.sin(sqrt_term1) * np.cos(sqrt_term2)
        term2 = (x_next + 1) * np.cos(sqrt_term1) * np.sin(sqrt_term2)
        
        total += term1 + term2
    
    return total


def Rastrigin(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d == 0:
        raise ValueError("Input array cannot be empty")
    
    A = 10
    return A*d + np.sum(x**2 - A*np.cos(2*np.pi*x))


def Rosenbrock(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d < 2:
        raise ValueError("Rosenbrock function requires at least 2 variables")
    
    total = 0.0
    for i in range(d - 1):
        total += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return total


def RosenbrockGeneralized(x):
    x = np.atleast_1d(np.array(x, dtype=float))
    d = len(x)
    
    if d < 2:
        raise ValueError("Rosenbrock function requires at least 2 variables")
    
    x0 = x[:-1]
    x1 = x[1:]
    
    return np.sum(100*(x1 - x0**2)**2 + (1 - x0)**2)

def Schaffer(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Schaffer function is 2-dimensional")
    num = np.sin(np.sqrt(x[0]**2 + x[1]**2))**2 - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + num / den

def Schwefel(x):
    x = np.array(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def Shubert(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Shubert function is 2-dimensional")
    i = np.arange(1, 6)
    sum1 = np.sum(i * np.cos((i + 1) * x[0] + i))
    sum2 = np.sum(i * np.cos((i + 1) * x[1] + i))
    return sum1 * sum2

def ShekelFoxholes(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Shekel's Foxholes function is 2-dimensional")

    # Create grid of constants a1, a2
    a = np.array([-32, -16, 0, 16, 32])
    a1, a2 = np.meshgrid(a, a)
    a1 = a1.flatten()
    a2 = a2.flatten()

    sum_term = 0
    for j in range(25):
        sum_term += 1 / (j + 1 + (x[0] - a1[j])**6 + (x[1] - a2[j])**6)
    
    return (1 / (1/500 + sum_term))  # Reciprocal of the sum

def Sphere(x):
    x = np.array(x)
    return np.sum(x**2)

def Step(x):
    x = np.array(x)
    return np.sum(np.floor(x + 0.5)**2)

def StretchedV(x):
    x = np.array(x)
    s = 0
    for i in range(len(x) - 1):
        term = (x[i]**2 + x[i+1]**2)**0.25
        s += term * (np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2 + 0.1)
    return s

def SumSquares(x):
    x = np.array(x)
    i = np.arange(1, len(x) + 1)
    return np.sum(i * x**2)

def Trecanni(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Trecanni function is 2-dimensional")
    return x[0]**4 - 4*x[0]**3 + 4*x[0]**2 + x[1]**2

def XOR(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("XOR function is 2-dimensional")
    s1, s2 = np.sin(x[0]), np.sin(x[1])
    return s1 + s2 - 2 * s1 * s2


def Watson(x):
    x = np.array(x)
    n = len(x)
    a = np.arange(1, 30) / 29.0
    total = 0
    for i in range(29):
        term1 = np.sum([(j - 1) * (a[i] ** (j - 2)) * x[j - 1] for j in range(2, n + 1)])
        term2 = np.sum([(a[i] ** (j - 1)) * x[j - 1] for j in range(1, n + 1)])
        total += (term1 - term2**2 - 1)**2
    return total + x[0]**2


def Trefethen4(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Trefethen No.4 function is 2-dimensional")
    x1, x2 = x
    return (
        np.exp(np.sin(50 * x1))
        + np.sin(60 * np.exp(x2))
        + np.sin(70 * np.sin(x1))
        + np.sin(np.sin(80 * x2))
        - np.sin(10 * (x1 + x2))
        + 0.25 * (x1**2 + x2**2)
    )
import numpy as np

def Zettl(x):
    x = np.array(x)
    x1, x2 = x[0], x[1]
    return (x1**2 + x2**2 - 2*x1)**2 + 0.25*x1


def Zimmerman(x):
    x = np.array(x)
    x1, x2 = x[0], x[1]
    return np.max([abs(x1), abs(x2), abs(x1 + x2 - 2), abs(x1 - x2 + 1)])


__all__ = [
    'Ackley',
    'Beale',
    'Bohachevsky1',
    'Bohachevsky2',
    'Booth',
    'BoxBetts',
    'Branin1',
    'Branin2',
    'Camel3',
    'Camel6',
    'Chichinadze',
    'Cola',
    'Colville',
    'Corona',
    'Easom',
    'EggHolder',
    'Exp2',
    'Hansen',
    'Hartmann3',
    'Hartmann6',
    'Himmelblau',
    'Holzman',
    'Hosaki',
    'Hyperellipsoid',
    'Katsuura',
    'Kowalik',
    'Langermann',
    'LennardJones',
    'Leon',
    'Levy',
    'Matyas',
    'MaxFold',
    'McCormick',
    'Michalewicz',
    'Multimod',
    'Paviani',
    'Powell',
    'Price',
    'Quartic',
    'QuarticNoise',
    'Rana',
    'Rastrigin',
    'Rosenbrock',
    'RosenbrockGeneralized',
    'Schaffer',
    'Schwefel',
    'ShekelFoxholes',
    'Shubert',
    'Sphere',
    'Step',
    'StretchedV',
    'SumSquares',
    'Trecanni',
    'Trefethen4',
    'Watson',
    'XOR',
    'Zettl',
    'Zimmerman',
    'ackley',
    'beale',
    'booth',
]