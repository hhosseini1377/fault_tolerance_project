from UUnifast import UUniFastDiscard
import random as rand
from math import pow, exp, ceil, log10
from numpy import random 
import numpy as np
from math import gcd
import matplotlib.pyplot as plt
import pandas as pd
# System parameters
n = 20
m_set = [4]
task_sets_num = 2000
frequency_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
alpha = 0.01
lambda_0 = 0.0001
d = 4
scaling_factor = 1

# Generate task sets
def generate_tasks():
    """
    This function generates n tasks with random cumulative utilization between 0 and 1.
    generated utilizations will be scaled up with factor m, if possible, otherwise they will be discarded
    NOTE: It also returns m
    """
    sets = []
    m_sets = []
    while len(sets) < task_sets_num:

        m = rand.choice(m_set)
        per_core_utilization = 3.2/m

        set = UUniFastDiscard(n, per_core_utilization, 1)[0]
        utilization_exceeded = False

        # Check if each utilization is less than 1. If this condition is met the set will be added to the task_sets.
        for i in range(len(set)):
            set[i] *= m
            if set[i] > 1:
                # Discard a task if its utilization exceeds 1.
                utilization_exceeded = True
                continue

        if not utilization_exceeded:            
            sets.append(set)
            m_sets.append(m)

    return m_sets, sets

# Return fault rate for a given frequency level
def calculate_lambda(f):
    return lambda_0*pow(10, (d*(1-f))/1-min(frequency_levels))

# Return reliability for given frequency and WCET
def calculate_Pof(frequency: float, WCET: float):
    fault_rate = calculate_lambda(frequency)
    return 1 - exp(-fault_rate*WCET/frequency)

def calculate_power(frequency: float):
    """
    This function calculates total power of the task based on frequency level.
    NOTE: power is normalized based on maximum dynamic power
    """
    return frequency*0.2 + pow(frequency, 3)

def calculate_energy(frequency: float, replica_count: int, WCET: float):
    """
    This function calculates total energy of the replicas based on frequency level.
    NOTE: energy is normalized based on maximum energy
    """
    power = calculate_power(frequency=frequency)
    energy = (power*WCET/frequency)*replica_count
    return energy

class Task:
    """
    Class for tasks in the system
    NOTE: PoF target is randomly chosen based on DO178B
    """

    def __init__(self, utilization) -> None:
        self.period = random.uniform(1, 10, 1)[0].astype(np.int64)*10
        self.WCET = utilization*self.period
        self.job_level_PoF = calculate_Pof(1, self.WCET)
        self.utilization = utilization

    def calculate_jobs_in_hyper_period(self, hyper_period):
        self.jobs_in_hyper_period = hyper_period//self.period

    def calcualte_task_level_PoF(self, ):
        task_level_reliability = 1
        for _ in range(self.jobs_in_hyper_period):
            task_level_reliability *= (1-self.job_level_PoF)
        self.task_level_PoF = 1- task_level_reliability
    
    def calculate_targeted_job_level_PoF(self, ):
        targeted_task_level_PoF = self.task_level_PoF * scaling_factor
        self.PoF_target = 1 - (1-targeted_task_level_PoF)** (1/self.jobs_in_hyper_period)


    def generate_EFR_table(self, ) -> list:
        """
        This function returns EFR_table for the task.
        NOTE: elements that are not proper are removed
        """

        EFR_table = []
        for f in frequency_levels:
            frequency_PoF = calculate_Pof(frequency=f, WCET=self.WCET)
            try: 
                replica_count = ceil(log10(self.PoF_target)/log10(frequency_PoF))
            except Exception as E:
                break
            total_execution_time = replica_count/f*self.WCET
            total_energy = calculate_energy(frequency=f, replica_count=replica_count, WCET=self.WCET)
            EFR_element = {
                'frequency': f,
                'replica_count': replica_count,
                'total_energy': total_energy,
                'total_execution_time': total_execution_time,
            }
            EFR_table.append(EFR_element)

        index = 1
        # Remove row with higher energies and higher execution times
        while index < len(EFR_table) and len(EFR_table) > 1:
            if EFR_table[index]['total_energy'] > EFR_table[index-1]['total_energy']:
                del EFR_table[index]
                index = 1
                continue
            index += 1
        self.EFR_table = EFR_table
        return EFR_table

class Workload:
    """
    This class is created for the workload of the system comprising task_set and cores
    Args:
        task_sets: set of the tasks running on the system
        m: number of cores in the system
    """
    def __init__(self, task_set, m) -> None:
        self.task_set = task_set
        self.m = m
        self.cores = []
        self.hyper_period = self.calculate_hyper_period()
        for _ in range(m):
            self.cores.append(Core())

        # total utlization of the workload
        self.utilization = 0
        for task in self.task_set:
            self.utilization += task.utilization
        
        # Job level targeted PoF for each task
        for task in self.task_set:
            task.calculate_jobs_in_hyper_period(self.hyper_period)
            task.calcualte_task_level_PoF()
            task.calculate_targeted_job_level_PoF()

    def calculate_hyper_period(self):
        lcm = 1
        for task in self.task_set:
            lcm = lcm*task.period//gcd(lcm, task.period)
        return lcm

    def LPF(self, is_fixed, settings, ):
        """
        This function determines the next settings using LPF policy
        Args:
            is_fixed: the input whether a setting can be changed, settings: settings of the replicas
        """
        is_changed = False
        change_index = -1
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                energy_difference = self.task_set[i].EFR_table[settings[i]]['total_energy'] - self.task_set[i].EFR_table[settings[i]+1]['total_energy']
                time_difference = self.task_set[i].EFR_table[settings[i]+1]['total_execution_time'] - self.task_set[i].EFR_table[settings[i]]['total_execution_time']
                max_difference = energy_difference/time_difference
                break
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                energy_difference = self.task_set[i].EFR_table[settings[i]]['total_energy'] - self.task_set[i].EFR_table[settings[i]+1]['total_energy']
                time_difference = self.task_set[i].EFR_table[settings[i]+1]['total_execution_time'] - self.task_set[i].EFR_table[settings[i]]['total_execution_time']
                difference = energy_difference/time_difference
                if difference >= max_difference:
                    max_difference = difference
                    change_index = i
                    is_changed = True
        if is_changed:
            settings[change_index] += 1
        return change_index

    def LEF(self, is_fixed, settings, ):
        """
        This function determines the next settings using LEF policy
        Args:
            is_fixed: the input whether a setting can be changed, settings: settings of the replicas
        """
        is_changed = False
        change_index = -1
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                max_energy_difference = self.task_set[i].EFR_table[settings[i]]['total_energy'] - self.task_set[i].EFR_table[settings[i]+1]['total_energy']
                break
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                if self.task_set[i].EFR_table[settings[i]]['total_energy'] - self.task_set[i].EFR_table[settings[i]+1]['total_energy'] >= max_energy_difference:
                    max_energy_difference = self.task_set[i].EFR_table[settings[i]]['total_energy'] - self.task_set[i].EFR_table[settings[i]+1]['total_energy']
                    change_index = i
                    is_changed = True
        if is_changed:
            settings[change_index] += 1
        return change_index

    def LUF(self, is_fixed, settings, ):
        """
        This function determines the next settings using LUF policy
        Args:
            is_fixed: determines whether a setting can be changed, settings: settings of the replicas
        """
        is_changed = False
        change_index = -1
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                max_execution_time_difference = self.task_set[i].EFR_table[settings[i]]['total_execution_time']
                break
        for i in range(len(self.task_set)):
            if not is_fixed[i] and settings[i] < len(self.task_set[i].EFR_table) - 1:
                if self.task_set[i].EFR_table[settings[i]]['total_execution_time'] >= max_execution_time_difference:
                    max_execution_time_difference = self.task_set[i].EFR_table[settings[i]]['total_execution_time']
                    change_index = i
                    is_changed = True
        if is_changed:
            settings[change_index] += 1
        return change_index

    def energy_saving(self, ):
        """
        This function is create to calculate amount of energy serving using system settings
        """
        total_energy = 0
        total_base_energy = 0
        for index in range(len(self.task_set)):
            total_energy += self.task_set[index].EFR_table[self.settings[index]]['total_energy']
            total_base_energy += self.task_set[index].EFR_table[0]['total_energy']
            energy_saving = 1 - total_energy/total_base_energy
        return energy_saving
    def GEERP(self, policy):
        """
        This function maps the replicas on the cores.
        If the replicas are not schedulable using this policy, It returns false
        NOTE: the settings of the first rows are used for mapping
        Args:
            policy: the policy of relaxing the tasks. It can be either LEF, LPF or LUF
        """
        is_scheduled = False
        change_index = None
        is_ended = False
        settings = []
        is_fixed = []
        for _ in range(len(self.task_set)):
            settings.append(0)
            is_fixed.append(False)

        while not is_ended:
            cores = []
            for _ in range(self.m):
                cores.append(Core())
            replicas = []
            index = 0
            for task in self.task_set:
                for _ in range(task.EFR_table[settings[index]]['replica_count']):
                    replicas.append(Replica(task=task, f = task.EFR_table[0]['frequency']))
                index += 1
            replicas.sort(key=lambda x: x.utilization, reverse=True)
            replicas_assigned = 0
            for replica in replicas:
                for core in cores:
                    if replica.utilization/replica.f < 1 - core.utilization:
                        is_replica_used = False
                        for replica_ in core.replica_set:
                            if replica_.task == replica.task:
                                is_replica_used = True
                                break
                        if not is_replica_used:
                            core.replica_set.append(replica)
                            core.utilization += replica.utilization/replica.f
                            replicas_assigned += 1
                            break
            # if the system was schedulable
            if len(replicas) == replicas_assigned:
                self.cores = cores.copy() 
                is_scheduled = True
                # Determine the next task to be relaxed
                if policy == 'LEF':
                    change_index = self.LEF(is_fixed=is_fixed, settings=settings)
                elif policy == 'LPF':
                    change_index = self.LPF(is_fixed=is_fixed, settings=settings)
                elif policy == 'LUF':
                    change_index = self.LUF(is_fixed=is_fixed, settings=settings)
                if change_index == -1:
                    is_ended = True
            # If the system was not schedulable 
            else:
                
                if change_index == None:
                    is_ended = True
                else:
                    settings[change_index] -= 1
                    is_fixed[change_index] = True
        # Return false if system is not schedulable 
        if not is_scheduled:
            return False
        # Return true if system is schedulable
        self.settings = settings
        return True


class Core:
    """
    This class is created for each core of the system.
    """
    def __init__(self) -> None:
        self.replica_set = []
        self.utilization = 0

class Replica():
    def __init__(self, task, f) -> None:
        self.task = task
        self.utilization = task.utilization
        self.f = f


# def main():
m_sets, utilization_sets = generate_tasks()
# Create the workload
energy_savings = []
while scaling_factor <10000:
    print(scaling_factor)
    energy_saving = {
        'scaling_factor': scaling_factor,
        'energy_saving': 0
    }
    scheduled = 0
    index = 0
    while index < len(utilization_sets):
        
        utilization_set = utilization_sets[index]
        m = m_sets[index]
        task_set = []
        for utilization in utilization_set:
            task = Task(utilization=utilization)
            task_set.append(task)
        work_load = Workload(task_set=task_set, m=m)
        for task in task_set:
            task.generate_EFR_table()
        try:
            if work_load.GEERP(policy='LUF'):
                # print('System is schedulable')
                # print(f'Total energy saving of the system: {work_load.energy_saving()}')
                # print(f'Cumulativate utilization of the workload: {work_load.utilization}')
                # print(f'Per core utilization of the system: {work_load.utilization/work_load.m} \n')
                energy_saving['energy_saving'] += work_load.energy_saving()
                scheduled += 1
                index += 1
        except Exception as e:
            continue                
    print(scheduled)
    try:
        energy_saving['energy_saving'] /= scheduled
    except Exception as E:
        pass
    energy_savings.append(energy_saving)
    scaling_factor *= 10
print(energy_savings)

