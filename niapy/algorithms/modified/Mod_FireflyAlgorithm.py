import logging
import math
import numpy as np
import random
from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean
from scipy.special import gamma

__all__ = ['Mod_FireflyAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class Mod_FireflyAlgorithm(Algorithm):
    Name = ['Mod_FireflyAlgorithm', 'ModFA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46."""

    def __init__(self, population_size=20, alpha_0=0.5, beta0=0.1, gamma_sym=1, theta=0.9, eta = 0.4,beta_chaos = 0.5,tau = 1.5,*args, **kwargs):
        """Initialize Mod_FireflyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha_0 (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma_sym (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta0 = beta0
        self.gamma_sym = gamma_sym
        self.theta = theta
        self.eta = eta
        self.beta_chaos = beta_chaos 
        self.tau = tau
        self.sigma_u = ( gamma(1+self.tau)*np.sin(np.pi*self.tau/2) ) / ( gamma((1+self.tau)/2)*self.tau*2**((self.tau-1)/2) )**(1/self.tau)
        self.sigma_v = 1

    def set_parameters(self, population_size=20, alpha_0=0.5, beta0=0.1, gamma_sym=1, theta=0.9,eta = 4,beta_chaos = 0.5,tau = 1.5, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha_0 (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma_sym (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.alpha_0 = alpha_0
        self.beta0 = beta0
        self.gamma_sym = gamma_sym
        self.theta = theta
        self.eta = eta
        self.beta_chaos = beta_chaos 
        self.tau = tau
        self.sigma_u = ( gamma(1+self.tau)*np.sin(np.pi*self.tau/2) ) / ( gamma((1+self.tau)/2)*self.tau*2**((self.tau-1)/2) )**(1/self.tau)
        self.sigma_v = 1
    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'alpha': self.alpha,
            'beta0': self.beta0,
            'gamma_sym': self.gamma_sym,
            'theta': self.theta,
            'eta':self.eta,
            'beta_chaos':self.beta_chaos,
            'tau':self.tau
        })
        return params
    


    def init_ffa(self,task):
        Fireflies = np.zeros((self.population_size,task.dimension))
        Fitness = np.ones((self.population_size))
        for j in range(task.dimension):
            Fireflies[0][j] = random.uniform(0, 1) * (task.upper[j]- task.lower[j]) + task.lower[j] #X0
       
        for i in range(1,self.population_size):
            Fireflies[i] = self.eta * Fireflies[i-1] * (1-Fireflies[i-1]) #logistic map
            
        Fitness = np.apply_along_axis(task.eval, 1, Fireflies)#*task.optimization_type.value
        return Fireflies,Fitness
            
    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * alpha_0 (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        fireflies, intensity = self.init_ffa(task)
        # print("init_population function")
        # fireflies, intensity, _ = super().init_population(task)
        return fireflies, intensity, {'dummy': 0}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Firefly Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments:
                    * alpha_0 (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.basic.FireflyAlgorithm.move_ffa`

        """
        print('-----------------Iteration:',task.iters,'-------------------')
        # alpha_0 = params.pop('alpha_0') * self.theta
        alpha = self.alpha_0 * self.theta
        self.theta = self.theta*self.theta
        # print(self.theta)
        if self.beta_chaos == 0:
            self.beta_chaos = 0
        else:             
            self.beta_chaos = (1 / self.beta_chaos) % 1
        for i in range(self.population_size):
            for j in range(self.population_size):
                if population_fitness[i] > population_fitness[j]:
                    rij_2 = euclidean(population[i], population[j])
                    beta = (self.beta_chaos - self.beta0) * math.exp(-self.gamma_sym*rij_2) + self.beta0
                    levy_step = np.random.normal(0,self.sigma_u,task.dimension) / (np.abs((np.random.normal(0,self.sigma_v,task.dimension)))**(1/self.tau))
                    steps = alpha * (self.random(task.dimension) - 0.5) * levy_step
                    population[i] += beta * (population[j] - population[i]) + steps
                    population[i] = task.repair(population[i])
                    population_fitness[i] = task.eval(population[i])
                    # print('Fitness ',population_fitness[i])
                    best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'alpha': alpha}