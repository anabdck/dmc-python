import numpy as np
import cvxopt as cp


class DMC:
    def __init__(self, N1=1, N2=10, Nu=5, Nss=10, lamda=0.5, T_s=1, y0 = 0, model=None):

        self.N1 = N1
        self.N2 = N2

        self.Nu = Nu
        self.Nss = Nss
        self.lamda = lamda

        self.model = model

        self.T_s = T_s
        self.y0 = y0

        
        if self.N2 > (Nss):
            self.N2 = Nss 

        self.N = self.N2 - self.N1 + 1

        self.last_Delta_u = np.zeros((self.Nu, 1))
        self.last_u = 0
        self.k = 0

        self.init_matrices()

    def set_G(self):
        m = self.model
        m[-1] = 0
        # np.hstack((m, 0))
        # print(m[-1])
        for d in range(0, self.Nu):
            mask = np.arange(self.N1-d-1, self.N2-d, 1)
            # print(mask)

            mask = np.clip(mask, -1, None)
            # print(mask)
            if d == 0:
                G = np.array([m[mask]]).T
            else:
                G = np.hstack((G, np.array([m[mask]]).T))
        # print(G)
        return G
    
    def set_g(self):
        g = np.array([self.model[:self.Nss]]).T
        return g
    
    def set_Qu(self):
        I = np.eye(self.Nu)

        return self.lamda*I
        
    def set_Hqp(self):
        Hqp = 2*(self.G.T@self.G + self.Qu)
        return Hqp
    
    def init_hat_y0(self):
        hat_y0 = self.y0*np.ones((self.Nss, 1))
        # hat_y0[0,0] = self.model[0] # considerando que o primeiro elemento é o valor da saída do sistema em t=0
        return hat_y0
    
    def update_hat_y0(self):
        self.hat_y0 = self.hat_y0 + self.g*self.last_Delta_u[0].flatten()[0]
        self.hat_y0_0 = self.hat_y0[0]
        self.hat_y0 = np.vstack((self.hat_y0[1:], self.hat_y0[-1])) # 
            
    def update_f(self, y):
        self.f = np.array([self.hat_y0[self.N1:self.N2+1,0]]).T + (y - self.hat_y0_0.flatten()[0])
        

    def update_fqp(self, r):
        
        self.fqp = -2*self.G.T@(r - self.f)
         
    def init_matrices(self):
        self.G = self.set_G()
        self.g = self.set_g()
        self.Qu = self.set_Qu()
        self.Hqp = self.set_Hqp()
        self.hat_y0 = self.init_hat_y0()
        # self.update_f(self.y0)

    def update_matrices(self, y, r):

        self.update_hat_y0()
        self.update_f(y)
        self.update_fqp(r)

    def generate_constraints(self, constraints):

        R_ineq = []
        r_ineq = []

        # Restrições em Delta_u 
        if "delta_u_max" in constraints and "delta_u_min" in constraints:
            delta_u_max = constraints["delta_u_max"]
            delta_u_min = constraints["delta_u_min"]
            R_ineq_delta_u = np.vstack([np.eye(self.Nu), -np.eye(self.Nu)])  
            r_ineq_delta_u = np.hstack([delta_u_max * np.ones(self.Nu), -delta_u_min * np.ones(self.Nu)])
            R_ineq.append(R_ineq_delta_u)
            r_ineq.append(r_ineq_delta_u)

        # Restrições em u 
        if "u_max" in constraints and "u_min" in constraints:
            # print("Restrições em u")
            # print(constraints)
            u_max = constraints["u_max"]
            u_min = constraints["u_min"]
            # Restrições nas entradas do controle
            R_ineq_u = np.vstack([np.tril(np.ones((self.Nu, self.Nu))), -np.tril(np.ones((self.Nu, self.Nu)))])  
            r_ineq_u = np.hstack([(u_max - self.last_u) * np.ones(self.Nu), (-u_min + self.last_u) * np.ones(self.Nu)])
            R_ineq.append(R_ineq_u)
            r_ineq.append(r_ineq_u)

        # Restrições em y 
        if "y_max" in constraints and "y_min" in constraints:
            y_max = constraints["y_max"]
            y_min = constraints["y_min"]
            # Restrições na saída
            R_ineq_y = np.vstack([self.G, -self.G])  # I para max, -I para min
            r_ineq_y = np.hstack([(y_max * np.ones(self.N) - (self.f).flatten()), (-y_min * np.ones(self.N) + (self.f).flatten())])
            R_ineq.append(R_ineq_y)
            r_ineq.append(r_ineq_y)

        # Concatenar as restrições de todas as variáveis (se existirem)
        if len(R_ineq) > 0 and len(r_ineq) > 0:
            R_ineq = np.vstack(R_ineq)
            r_ineq = np.hstack(r_ineq)
        else:
            R_ineq = None
            r_ineq = None

        return R_ineq, r_ineq
            
    def solve_qp(self, R=None, r=None):
        Hqp = cp.matrix(self.Hqp)
        fqp = cp.matrix(self.fqp)

        if self.R_ineq is not None:
            R = cp.matrix(self.R_ineq)
            r = cp.matrix(self.r_ineq)

        cp.solvers.options['show_progress'] = False

        sol = cp.solvers.qp(Hqp, fqp, R, r, None, None)
        Delta_u = sol['x']
        return np.array(Delta_u)
        # Delta_u = -np.linalg.inv(self.Hqp)@self.fqp
        # # J = 0.5*Delta_u.T@self.Hqp@Delta_u + self.fqp.T@Delta_u
        # return Delta_u
           
    def get_u(self, y, r, constraints=None):

        if isinstance(r, (int, float)):
            r = np.array([[r]*self.N]).T

        self.update_matrices(y, r)

        # Se as restrições forem passadas, gere as matrizes de restrição
        self.R_ineq = None
        self.r_ineq = None
        
        if constraints:
            self.R_ineq, self.r_ineq = self.generate_constraints(constraints)

        Delta_u = self.solve_qp()
        
        if Delta_u is None:
            print("Erro ao resolver o problema de otimização.")
            return None

        self.k += 1
        self.last_Delta_u = Delta_u
        self.last_u += Delta_u[0]

        return Delta_u[0]

