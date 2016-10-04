import numpy as np

class DE_optimizer(object):

    def __init__(self,
               evaluator,
               population_size,
               f,
               cr,
               max_iter,
               strategy = 9,
               vtr = -1e99,
               eps = 0,
               show_progress = 1,
               monitor_cycle = 5):

        self.show_progress = show_progress
        self.evaluator = evaluator
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.max_iter = max_iter
        self.strategy = strategy
        if strategy > 5 :
            self.st = strategy - 5
        else :
            self.st = strategy
        self.monitor_cycle = max_iter +1 #### temp
        self.D = evaluator.D
        self.eps = eps
        self.vtr = vtr
        self.population = np.zeros((self.population_size,self.D))
        self.scores = 1e99 * np.ones(self.population_size)
        self.optimize()

    def optimize(self):
    # initialise the population
        self.make_random_population()
    # score the population
        self.score_population()

    #Initial pop features
        self.best_val = np.min(self.scores)
        self.best_ind = self.population[np.argmin(self.scores),:]
        self.mean_val = np.mean(self.scores)
        self.count = 0
        self.nfeval = self.population_size
        self.monitor()

        while not self.terminate():

            self.evolve()
        #Calculate the generation features
            self.mean_val = np.mean(self.scores)
            if self.mean_val == 0 :
                self.ObjProgr == 0
            else :
                self.ObjProgr = (self.mean_val - self.best_val) / self.mean_val

            self.count += 1
        #Monitoring
            if self.count%self.monitor_cycle == 0:
                self.monitor()

        #Reporting
            if self.show_progress != 0 :
                if self.count % self.show_progress == 0 :
                    self.print_status()

        self.final_print()


    def make_random_population(self):
        for ii in range(0,self.D):
            delta  = self.evaluator.domain[ii,1] - self.evaluator.domain[ii,0]
            offset = self.evaluator.domain[ii,0]
            random_values = np.random.rand(self.population_size) * delta + offset
            self.population[:, ii] = random_values

    def score_population(self):
        for ii in range(0,self.population_size) :
            if any(self.population[ii, :] < self.evaluator.domain[:,0]) or any(self.population[ii, :] > self.evaluator.domain[:,1]):
                self.scores[ii] = 1e99
            else :
                self.scores[ii] = self.evaluator.target(self.population[ii, :])
                self.print_ind(self.scores[ii],self.population[ii, :],ii)


    def monitor(self):
        if self.count == 0 :
            self.monitor_score = np.min(self.scores)
            self.monitor_indiv = self.population[np.argmin(self.scores),:]
            self.monitor_mean  = np.mean(self.scores)
            self.monitor_obj_change = 1e99
        else :
            self.monitor_score = self.best_val
            self.monitor_indiv = self.best_ind
            self.monitor_mean  = self.mean_val
            if self.monitor_score == 0 :
                self.monitor_obj_change = 0
            else :
                self.monitor_obj_change = (self.monitor_score - self.best_val )/self.monitor_score

    def evolve(self):
    #Birth
        ui = self.cal_trials()
    #Selection
        for i in range(0, self.population_size) :
            if any(ui[i, :] < self.evaluator.domain[:,0]) or any(ui[i, :] > self.evaluator.domain[:,1]):
                tempval = 1e99
            else :
                if all(ui[i,:] == self.population[i,:]):
                    tempval = self.scores[i]
                else:
                    tempval = self.evaluator.target(ui[i,:])
                    self.nfeval += 1
            if (tempval < self.scores[i]) :
                self.population[i,:] = ui[i,:]
                self.scores[i] = tempval
                if (tempval < self.best_val) :
                    self.best_val = tempval
                    self.best_ind = ui[i,:]
            self.print_ind(self.scores[i],self.population[i, :],i)

    def cal_trials(self):
        popold = self.population

        rot = np.arange(0, self.population_size)
        rotd= np.arange(0, self.D)
        ind    = np.random.permutation(4)

        a1 = np.random.permutation(self.population_size)
        rt = (rot + ind[0]) % self.population_size
        a2 = a1[rt]
        rt = (rot + ind[1]) % self.population_size
        a3 = a2[rt]
        rt = (rot + ind[2]) % self.population_size
        a4 = a3[rt]
        rt = (rot + ind[3]) % self.population_size
        a5 = a4[rt]

        pm1 = self.population[a1,:]
        pm2 = self.population[a2,:]
        pm3 = self.population[a3,:]
        pm4 = self.population[a4,:]
        pm5 = self.population[a5,:]

        pop_of_best_ind = np.zeros((self.population_size,self.D))
        for i in range(0, self.population_size) :
            pop_of_best_ind[i] = self.best_ind

        cr_decision = np.random.rand(self.population_size,self.D) < self.cr

        if (self.strategy > 5) :
            cr_decision = np.sort(np.transpose(cr_decision))
            for i in range(0, self.population_size) :
                n = np.floor(np.random.rand(1) * self.D)
                if n > 0 :
                    rtd = (rotd + n) % self.D
                    rtd = rtd.astype(int)
                    cr_decision[:,i] = cr_decision[rtd, i]
            cr_decision = np.transpose(cr_decision)
        mpo = cr_decision < 0.5
        ui = 0
        if (self.st == 1) :
            dif = self.f * (pm1 - pm2)
            ui = pop_of_best_ind + dif
            ui = self.positivize(ui,pop_of_best_ind, dif)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif(self.st == 2) :
            dif = self.f * (pm1 - pm2)
            ui = pm3 + dif
            ui = self.positivize(ui,pop_of_best_ind, dif)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif(self.st == 3) :
            dif = self.f * (pop_of_best_ind - popold + pm1 - pm2)
            ui = popold + dif
            ui = self.positivize(ui,pop_of_best_ind, dif)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif(self.st == 4) :
            dif = self.f * (pm1 - pm2 + pm3 - pm4)
            ui = pop_of_best_ind + dif
            ui = self.positivize(ui,pop_of_best_ind, dif)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        elif(self.st == 5) :
            dif = self.f * (pm1 - pm2 + pm3 - pm4)
            ui = pm5 + dif
            ui = self.positivize(ui,pop_of_best_ind, dif)
            ui = np.multiply(popold, mpo) + np.multiply(ui, cr_decision)
        return ui


    def terminate(self):
    #Termination 1 : By maxiter
        if self.count >= self.max_iter:
            self.termination = 1
            return True
    #Termination 2 : By Value to reach
        if self.best_val < self.vtr :
            self.termination = 2
            return True
    #Termination 3 : By monitor cycle change in the objective
        if  self.monitor_obj_change < self.eps:
            self.termination = 3
            return True
        return False

    def positivize(self,ui, pop_of_best_ind, dif):
        ui = np.where(ui >= 0, ui, ui - 2*dif)
        return np.where(ui >= 0, ui, pop_of_best_ind)


    def print_status(self):
        print('********************************************************************************************')
        print('iter {}:best performance: {}, best individual: {}, objective progress: {}'.format(self.count,self.best_val * -1,self.best_ind,self.ObjProgr))
        print('')

    def print_ind(self, score, ind, ind_no):
        print('{}: Performance: {}, F_weight: {}'.format(ind_no +1,-1 * score,ind))

    def final_print(self):
        print('')
        print('* Problem Specifications')
        print('Date           : {}'.format(self.evaluator.scheduler.Date))
        print('Initial field  : {}'.format(self.evaluator.scheduler.init_id))
        print('Preferences    : {}'.format(self.evaluator.scheduler.preferences))
        print('')
        print('* DE parameters')
        print('F          : {}'.format(self.f))
        print('Cr         : {}'.format(self.cr))
        print('Pop Size   : {}'.format(self.population_size))
        print('No. of eval: {}'.format(self.nfeval))
        print('No. of Iter: {}'.format(self.count))
        print('Termination: {}'.format(self.termination))
        print('eps        : {}'.format(self.eps))
        print('vtr        : {}'.format(self.vtr))
        print('')