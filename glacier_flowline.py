import numpy as np
import xarray as xr
import math

class GlacierFlowline(object):
    def __init__(self, nx, dx, dt, **kwargs):
        self.secperyr = 31556926
        self.rho_ice = 910.0
        self.g = 9.81
        self.glen_n = 3.
        self.ice_softness = 5e-17/self.secperyr # 1e-16 Bueler notes 2018
        self.sliding_constant = 1e-15 # 7e-16, 5e-15 Herman 2011 epsl hydro paper
        #self.sliding_constant = 3e-5 # m = 1
        #self.sliding_constant = 2e-53 # m = 10
        self.weertman_m = 3.
        
        self.erosion_k = 5e-4
        self.erosion_l = 1
        
        self.slope_threshold = 30
        
        self.uplift_rate = kwargs.get('uplift_rate', 0.)
        
        self.deform_e = 1.0
        self.sliding_e = 1.0
        
        self.n_ghost = 2
        self.nx = nx
        self.dx = dx
        
        try:
            self.flux_in = kwargs['flux_in']/self.secperyr
        except KeyError:
            self.flux_in = 0
        
        self.num_of_nodes = self.nx + 2 * self.n_ghost
        self.num_of_links = self.num_of_nodes - 1
        self.core_nodes = np.arange(self.n_ghost, self.n_ghost+self.nx)
        self.core_links = np.arange(self.n_ghost, self.n_ghost+self.nx-1)
        
        self.at_node = {}
        self.at_link = {}
        
        self.at_node['topg'] = np.zeros(self.nx+2*self.n_ghost)
        self.at_node['thk'] = np.zeros(self.nx+2*self.n_ghost) + 1e-9
        
        try:
            self.at_node['topg'][self.core_nodes] = kwargs['initial_topg']
        except KeyError:
            pass
        
        try:
            self.at_node['thk'][self.core_nodes] = kwargs['initial_thk']
        except KeyError:
            pass
        for i in range(self.n_ghost):
            self.at_node['thk'][i] = self.at_node['thk'][self.n_ghost]
        
        self.at_node['surf'] = self.at_node['topg'] + self.at_node['thk']
        
        self.at_link['deform_vel'] = np.zeros(self.num_of_links)
        self.at_link['sliding_vel'] = np.zeros(self.num_of_links)
        
        self.at_node['mb'] = np.zeros(self.nx+2*self.n_ghost)
        try:
            self.mass_balance_beta = kwargs['mass_balance_beta']
        except KeyError:
            self.mass_balance_beta = 0.005    
        try:
            self.snowfall_rate = kwargs['snowfall_rate']
        except KeyError:
            self.snowfall_rate = 1.0
        try:
            self.ela = kwargs['ela']
        except KeyError:
            self.ela = 1500
            
        self.at_node['eff_pres'] = np.zeros(self.num_of_nodes)
        
        self.model_state = None
        self.update_model_state()
        
        self.saved_results = None
        
    def comp_effective_pressure(self):
        self.at_node['eff_pres'] = 0.8 * self.rho_ice * self.g * self.at_node['thk']
        
    def comp_deformation_velocity(self):
        # staggered
        self.at_link['deform_vel'] = np.zeros(self.num_of_links)
        surf_grad = (self.at_node['surf'][1:] - self.at_node['surf'][:-1]) / self.dx
        thk_staggered = 0.5 * (self.at_node['thk'][1:] + self.at_node['thk'][:-1])
        self.at_link['deform_vel'] = -2 / (self.glen_n + 2) * self.ice_softness * \
            np.power((self.rho_ice * self.g), self.glen_n) * np.power(thk_staggered, self.glen_n+1) \
            * np.power(np.abs(surf_grad), self.glen_n-1) * surf_grad
        for i in range(self.n_ghost):
            self.at_link['deform_vel'][i] = 0.0
        
        self.at_link['deform_vel'] *= self.deform_e
    
    def comp_sliding_velocity(self):
        # staggered
        self.at_link['sliding_vel'] = np.zeros(self.num_of_links)
        self.comp_effective_pressure()
        eff_pres_staggered = 0.5 * (self.at_node['eff_pres'][1:] + self.at_node['eff_pres'][:-1])

        surf_grad = (self.at_node['surf'][1:] - self.at_node['surf'][:-1]) / self.dx
        thk_staggered = 0.5 * (self.at_node['thk'][1:] + self.at_node['thk'][:-1])
        basal_shear_staggered = self.rho_ice * self.g * thk_staggered * surf_grad
        
        self.at_link['sliding_vel'] = -self.sliding_constant / eff_pres_staggered \
            * np.power(np.abs(basal_shear_staggered), self.weertman_m-1) \
            * basal_shear_staggered
        for i in range(self.n_ghost):
            self.at_link['sliding_vel'][i] = 0.0
        
        self.at_link['sliding_vel'] *= self.sliding_e
            
    def ssa_one_step(self, N, gamma, W, alpha, beta, u_left):
        # see https://github.com/bueler/mccarthy/blob/master/mfiles/flowline.m
        rhs = self.dx**2 * beta
        rhs[0] = u_left
        #rhs[-1] = rhs[-1] - 2 * gamma * self.dx * W[-1]
        rhs[-1] = 0.0
        
        A = np.zeros((N, N))
        A[0, 0] = 1.0
        for i in range(1, N-1):
            A[i, i-1] = W[i-1]
            A[i, i] = -(W[i-1] + W[i] + alpha[i] * self.dx**2)
            A[i, i+1] = W[i]
        #A[N-1, N-2] = W[N-2] + W[N-1]
        #A[N-1, N-1] = -(W[N-2] + W[N-1] + alpha[N-1] * self.dx**2)
        A[N-1, N-1] = 1.0

        u = np.linalg.solve(A, rhs)
        
        #import pdb;pdb.set_trace()
        return u
        
    def comp_sliding_velocity_ssa(self):
        # see https://github.com/bueler/mccarthy/blob/master/mfiles/ssaflowline.m
        
        is_glacier = np.intersect1d(self.core_nodes, np.where(self.at_node['thk'] > 1e-3)[0])
        N = len(is_glacier)
        if N < 3:
            return
        
        h = self.at_node['surf'][is_glacier]
        hx = np.zeros(N)
        hx[1:-1] = (h[2:] - h[:-2]) / (2*self.dx)
        hx[0] = (h[1] - h[0]) / self.dx
        hx[-1] = (h[-1] - h[-2]) / self.dx
        
        H = self.at_node['thk'][is_glacier]
        beta = self.rho_ice * self.g * H * hx
        gamma = (0.25 * self.ice_softness**(1.0/self.glen_n) * (1.0 - self.rho_ice/1000.) \
                 * self.rho_ice * self.g * H[-1]) ** self.glen_n

        self.comp_sliding_velocity()
        initial_u = np.zeros(self.num_of_nodes)
        initial_u[1:-1] = 0.5 * (self.at_link['sliding_vel'][1:] + self.at_link['sliding_vel'][:-1])
        u = initial_u[is_glacier]
        u_left = 0.0

        Hstag = 0.5 * (H[1:] + H[:-1])
        tol = 1.0 / self.secperyr
        max_step = 10000
        eps_reg = (1.0 / self.secperyr) / (self.nx * self.dx)
        maxdiff = 1e5
        W = np.zeros(N)
        count = 0
        while maxdiff > tol and count < max_step:
            #import pdb;pdb.set_trace()
            sqr_u_reg = np.power(u, 2) + eps_reg ** 2
            alpha = np.power(self.at_node['eff_pres'][is_glacier]/self.sliding_constant, 1.0/self.weertman_m) \
                * np.power(sqr_u_reg, (1.0/self.weertman_m - 1) / 2.0)
            uxstag = (u[1:] - u[:-1]) / self.dx
            sqr_ux_reg = np.power(uxstag, 2) + eps_reg ** 2 # regularize to avoid division by zero
            W[:-1] = 2 * self.ice_softness**(-1.0/self.glen_n) * Hstag \
                * np.power(sqr_ux_reg, (1.0/self.glen_n - 1) / 2.0)
            W[-1] = W[-2]
            unew = self.ssa_one_step(N, gamma, W, alpha, beta, u_left)
            maxdiff = np.nanmax(np.abs(unew - u))
            u = unew
            count += 1
        
        if count >= max_step:
            print("Warning: SSA iteration failed to converge")
        sliding_vel = np.zeros(self.num_of_nodes)
        sliding_vel[is_glacier] = u
        self.at_link['sliding_vel'] = 0.5 * (sliding_vel[1:] + sliding_vel[:-1])
        for i in range(self.n_ghost):
            self.at_link['sliding_vel'][i] = 0.0
        
        self.at_link['sliding_vel'] *= self.sliding_e
    
    def update_thk(self, dt, sliding='traditional'):
        self.comp_mass_balance()
        self.comp_deformation_velocity()
        if sliding == 'traditional':
            self.comp_sliding_velocity()
        elif sliding == 'ssa':
            self.comp_sliding_velocity_ssa()
        
        thk_staggered = 0.5 * (self.at_node['thk'][1:] + self.at_node['thk'][:-1])
        flux = (self.at_link['deform_vel'] + self.at_link['sliding_vel']) * thk_staggered
        flux_div = np.zeros(self.nx+2*self.n_ghost)
        flux_div[1:-1] = (flux[1:] - flux[:-1]) / self.dx
        flux_div[self.n_ghost] = (flux[self.n_ghost] - self.flux_in) / self.dx  # left boundary
        self.at_node['thk'] = self.at_node['thk'] + dt * self.secperyr * (self.at_node['mb'] - flux_div)
        
        self.at_node['thk'][np.where(self.at_node['thk'] <= 0)] = 1e-9 # avoid negative thk and avoid divding by zero
        
        self.at_node['surf'] = self.at_node['topg'] + self.at_node['thk']
        
    def comp_mass_balance(self):
        self.at_node['mb'] = self.mass_balance_beta * (self.at_node['surf'] - self.ela)
        self.at_node['mb'][np.where(self.at_node['mb'] > self.snowfall_rate)] = self.snowfall_rate
        self.at_node['mb'] = self.at_node['mb'] / self.secperyr
        #import pdb;pdb.set_trace()
        
    def update_topg(self, dt):
        #import pdb;pdb.set_trace()
        sliding_vel_not_stagged = 0.5 * (self.at_link['sliding_vel'][1:] + self.at_link['sliding_vel'][:-1])
        
        self.at_node['topg'][1:-1] -= dt * np.power(np.abs(sliding_vel_not_stagged) * self.secperyr, self.erosion_l) * self.erosion_k
        self.at_node['topg'][self.at_node['thk'] > 1e-3] += dt * self.uplift_rate # only change elevation under ice
        self.at_node['surf'] = self.at_node['topg'] + self.at_node['thk']
        
    def check_stability(self):
        # JL on 2022/05/02 to JL 3 years ago: I'm really confused by the loop order here.
        # so I changed it
        
        # prevent the glacier from digging a deep hole
        # from right to left
        for i in self.core_nodes:
            if (self.at_node['surf'][i] <= self.at_node['surf'][i+1]):
                if (self.at_node['topg'][i] - self.at_node['topg'][i+1])/self.dx > math.tan(self.slope_threshold/180*math.pi):
                    self.at_node['topg'][i+1] = self.at_node['topg'][i] - math.tan(self.slope_threshold/180*math.pi)*self.dx
                
        # from left to right
        for i in self.core_nodes[::-1]:
            if (self.at_node['surf'][i] <= self.at_node['surf'][i-1]):
                if (self.at_node['topg'][i] - self.at_node['topg'][i-1])/self.dx > math.tan(self.slope_threshold/180*math.pi):
                    self.at_node['topg'][i-1] = self.at_node['topg'][i] - math.tan(self.slope_threshold/180*math.pi)*self.dx

        # prevent over steepened slope
        # from right to left
        for i in self.core_nodes:
            if (self.at_node['topg'][i+1] - self.at_node['topg'][i])/self.dx > math.tan(self.slope_threshold/180*math.pi):
                self.at_node['topg'][i+1] = math.tan(self.slope_threshold/180*math.pi)*self.dx + self.at_node['topg'][i]

        # from left to right
        for i in self.core_nodes[::-1]:
            if (self.at_node['topg'][i-1] - self.at_node['topg'][i])/self.dx > math.tan(self.slope_threshold/180*math.pi):
                self.at_node['topg'][i-1] = math.tan(self.slope_threshold/180*math.pi)*self.dx + self.at_node['topg'][i]

        self.at_node['surf'] = self.at_node['topg'] + self.at_node['thk']
        
    def run_one_step(self, dt, sliding='traditional', erosion=False):
        self.update_thk(dt, sliding=sliding)
        if erosion:
            self.update_topg(dt)
            
    def _unstagger(self, a):
        b = np.zeros(len(a)+1)
        b[1:-1] = 0.5 * (a[1:] + a[:-1])
        
        return b
    
    def update_model_state(self):
        if self.model_state is None:
            # Initialize model state
            self.model_state = xr.Dataset()
            x = self.nx*self.dx - (np.arange(self.nx)*self.dx + self.dx/2)
            self.model_state.coords['x'] = (('x'), x, {'units': 'm', 'long_name': 'upstream distance'})
            self.model_state['bedrock_elevation'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['ice_thickness'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['sliding_velocity'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['deformation_velocity'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['mass_balance'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})

        self.model_state['bedrock_elevation'].data = self.at_node['topg'][self.core_nodes]
        self.model_state['ice_thickness'].data = self.at_node['thk'][self.core_nodes]
        self.model_state['sliding_velocity'].data = self._unstagger(self.at_link['sliding_vel'])[self.core_nodes]*self.secperyr
        self.model_state['deformation_velocity'].data = self._unstagger(self.at_link['deform_vel'])[self.core_nodes]*self.secperyr
        self.model_state['mass_balance'].data = self.at_node['mb'][self.core_nodes]
    
    def save_model_state(self, t, var_list=None):
        self.update_model_state()
        if var_list is None:
            var_list = ['bedrock_elevation', 'ice_thickness']
        state_to_save = self.model_state.copy(deep=True)
        
        for k in state_to_save.keys():
            if k not in var_list:
                state_to_save = state_to_save.drop_vars(k)
        
        state_to_save = state_to_save.expand_dims('time')
        state_to_save.coords['time'] = np.array([t], dtype=float)
        if self.saved_results is None:
            self.saved_results = state_to_save
        else:
            self.saved_results = xr.concat([self.saved_results, state_to_save], 'time')
            
    def write_saved_results_to_file(self, filename):
        #self.saved_results.attrs['input_params'] = json.dumps(self._params)
        self.saved_results['time'].attrs['units'] = 'years'
        #self.saved_results['time'].attrs['calendar'] = '365_day'
        self.saved_results.to_netcdf(filename)