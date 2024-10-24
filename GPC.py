from controller import BaseController
import numpy as np

class GPC(BaseController):
    def __init__(self, N1=1, N2=10, Nu=5, Nss=10, lamda=0.5, T_s=1, y0 = 0, step_response_coeffs=None):
        super().__init__(N1, N2, Nu, Nss, lamda, T_s, y0, step_response_coeffs)

        self.hat_y0 = self.init_hat_y0()
        
    def init_hat_y0(self):
        hat_y0 = self.y0*np.ones((self.Nss, 1))
        # hat_y0[0,0] = self.step_response_coeffs[0] # considerando que o primeiro elemento é o valor da saída do sistema em t=0
        return hat_y0
    
    def update_hat_y0(self):
        self.hat_y0 = self.hat_y0 + self.g*self.last_Delta_u[0].flatten()[0]
        self.hat_y0_0 = self.hat_y0[0]
        self.hat_y0 = np.vstack((self.hat_y0[1:], self.hat_y0[-1])) # 
            
    def update_f(self, y):
        self.f = np.array([self.hat_y0[self.N1:self.N2+1,0]]).T + (y - self.hat_y0_0.flatten()[0])
        

    def update_fqp(self, r):
        self.fqp = -2*self.G.T@(r - self.f)