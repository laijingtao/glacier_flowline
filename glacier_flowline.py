import numpy as np
import xarray as xr
import math

class GlacierFlowline(object):
    def __init__(self, nx, dx, **kwargs):
        self.secperyr = 31556926
        self.rho_ice = 910.0
        self.g = 9.81
        self.glen_n = 3.
        self.ice_softness = 1e-17/self.secperyr # 1e-16 Bueler notes 2018, ~1e-17 in Cuffey and Paterson 2010.
        self.sliding_constant = 5e-15 # 7e-16, 5e-15 Herman 2011 epsl hydro paper
        #self.sliding_constant = 3e-5 # m = 1
        #self.sliding_constant = 2e-53 # m = 10
        self.weertman_m = 3.
        
        self.erosion_k = 1e-4 # 1e-4 in Humphrey and Raymond 1994
        self.erosion_l = 1
        
        self.slope_threshold = kwargs.get('slope_threshold', 30)
        
        self.uplift_rate = kwargs.get('uplift_rate', 0.)
        
        self.deform_e = 1.0
        self.sliding_e = 1.0
        
        self.n_ghost = 2
        self.nx = nx
        self.dx = dx

        self.CFL_limit = kwargs.get('CFL_limit', 0.1)
        
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

        self.at_node['glacial_erosion_rate'] = np.zeros(self.num_of_nodes)

        self.large_dt_warning = False
        
        self.model_state = None
        self.update_model_state()
        
        self.saved_results = None
        
    
    def update_thk(self, dt):
        self.update_mass_balance()

        thk, deform_vel, sliding_vel, large_dt_warning = _update_thk_numba_impl(
            self.at_node['topg'], self.at_node['thk'], self.at_node['mb'],
            self.dx, dt, self.CFL_limit, self.glen_n, self.ice_softness,
            self.rho_ice, self.g, self.sliding_constant, self.weertman_m,
            self.deform_e, self.sliding_e, self.secperyr)
    
        self.at_link['deform_vel'] = deform_vel
        self.at_link['sliding_vel'] = sliding_vel
        self.at_node['thk'] = thk
        self.at_node['surf'] = self.at_node['topg'] + self.at_node['thk']

        if large_dt_warning:
            self.large_dt_warning = True
        
    def update_mass_balance(self):
        mb = _update_mass_balance_numba_impl(
            self.at_node['topg'], self.at_node['thk'], self.mass_balance_beta,
            self.ela, self.snowfall_rate, self.secperyr)
        self.at_node['mb'] = mb
        #self.at_node['mb'] = self.mass_balance_beta * (self.at_node['surf'] - self.ela)
        #self.at_node['mb'][np.where(self.at_node['mb'] > self.snowfall_rate)] = self.snowfall_rate
        #self.at_node['mb'] = self.at_node['mb'] / self.secperyr
        #import pdb;pdb.set_trace()
        
    def update_topg(self, dt):
        #import pdb;pdb.set_trace()
        sliding_vel_not_stagged = 0.5 * (self.at_link['sliding_vel'][1:] + self.at_link['sliding_vel'][:-1])
        glacial_erosion_rate = np.power(np.abs(sliding_vel_not_stagged) * self.secperyr, self.erosion_l) * self.erosion_k
        self.at_node['glacial_erosion_rate'] = glacial_erosion_rate

        self.at_node['topg'][1:-1] -= dt * glacial_erosion_rate
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
        
    def run_one_step(self, dt, erosion=False):
        self.update_thk(dt)
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
            x = np.arange(self.nx)*self.dx + self.dx/2
            self.model_state.coords['x'] = (('x'), x, {'units': 'm', 'long_name': 'distance'})
            self.model_state['bedrock_elevation'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['ice_thickness'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['surface_elevation'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['sliding_velocity'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['deformation_velocity'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['mass_balance'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['glacial_erosion_rate'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})

        self.model_state['bedrock_elevation'].data = self.at_node['topg'][self.core_nodes]
        self.model_state['ice_thickness'].data = self.at_node['thk'][self.core_nodes]
        self.model_state['surface_elevation'].data = self.at_node['surf'][self.core_nodes]
        self.model_state['sliding_velocity'].data = self._unstagger(self.at_link['sliding_vel'])[self.core_nodes]*self.secperyr
        self.model_state['deformation_velocity'].data = self._unstagger(self.at_link['deform_vel'])[self.core_nodes]*self.secperyr
        self.model_state['mass_balance'].data = self.at_node['mb'][self.core_nodes]*self.secperyr
        self.model_state['glacial_erosion_rate'].data = self.at_node['glacial_erosion_rate'][self.core_nodes]
    
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

def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    try:
        import numba
        return numba.njit(func, cache=True)
    except ImportError:
        return func

@_speed_up
def _update_thk_numba_impl(topg, thk, mb, dx, dt, cfl_limit, glen_n, ice_softness,
                           rho_ice, g, sliding_constant, weertman_m, deform_e,
                           sliding_e, secperyr):
    large_dt_warning = False

    dt = dt*secperyr

    curr_t = 0
    while curr_t < dt:
        deform_vel, sliding_vel = _update_vel_numba_impl(
            topg, thk, dx, glen_n, ice_softness, rho_ice, g,
            sliding_constant, weertman_m, deform_e, sliding_e)
        
        vel = sliding_vel + deform_vel

        sub_dt = cfl_limit*dx/np.nanmax(np.abs(vel)) # CFL condition
        if sub_dt > dt - curr_t:
            sub_dt = dt - curr_t
        
        '''
        # flux-limiter method
        surf = thk + topg
        slope_up = surf[1:-1] - surf[:-2]
        slope_down = surf[2:] - surf[1:-1]
        slope_down[slope_down < 1e-6] = 1e-6
        r_slope = np.zeros(len(surf))
        r_slope[1:-1] = slope_up/slope_down
        limiter = np.minimum(2*r_slope, (1+r_slope)/2)
        limiter = np.minimum(limiter, np.zeros(len(limiter))+2)
        limiter = np.maximum(np.zeros(len(limiter)), limiter) # monotonized centered limiter
        '''
        # upwind scheme
        flux = 0.5 * vel * (thk[1:] + thk[:-1]) - 0.5 * np.abs(vel) * (thk[1:] - thk[:-1])
        # central scheme
        #flux = 0.5 * vel * (thk[1:] + thk[:-1])

        # Lax-Wendroff scheme
        #flux_h = 0.5 * vel * (thk[1:] + thk[:-1]) - 0.5 * np.power(vel, 2) * sub_dt/dx * (thk[1:] - thk[:-1])
        #flux = flux_l + limiter[:-1]*(flux_h - flux_l)

        
        # modify flux to prevent negative thk
        # Section 5.10.1, Numerical Methods for Fluid Dynamics, 2nd, 2010
        flux_out = np.zeros(len(thk))
        flux_out[1:-1] = np.maximum(np.zeros(len(flux)-1), flux[1:]) - np.minimum(np.zeros(len(flux)-1), flux[:-1])
        flux_out += 1e-9
        flux_out_allowed = thk * dx / sub_dt
        r_flux = np.minimum(np.ones(len(flux_out)), flux_out_allowed/flux_out)
        corrector = np.zeros(len(flux))
        for k in range(len(corrector)):
            if flux[k] >= 0:
                corrector[k] = r_flux[k]
            else:
                corrector[k] = r_flux[k+1]
        flux = flux * corrector
        

        flux_div = np.zeros(len(thk))
        flux_div[1:-1] = (flux[1:] - flux[:-1]) / dx
        #flux_div[self.n_ghost] = (flux[self.n_ghost] - self.flux_in) / self.dx  # left boundary
        
        # update based on flux
        thk = thk + sub_dt * (0 - flux_div)
    
        if len(thk[thk < -1e-5]) > 0:
            large_dt_warning = True
            #print("Warning: negative thickness value possibility due to large dt!")

        # update surface mass change
        thk = thk + sub_dt * mb
        
        thk[thk <= 0] = 1e-9 # avoid negative thk and avoid divding by zero

        curr_t += sub_dt
    
    return thk, deform_vel, sliding_vel, large_dt_warning

@_speed_up
def _update_vel_numba_impl(topg, thk, dx, glen_n, ice_softness, rho_ice, g,
                           sliding_constant, weertman_m, deform_e, sliding_e):
    surf = topg + thk
    surf_grad = (surf[1:] - surf[:-1]) / dx
    
    # how to choose thk_staggered is important
    # 0.5 * (thk[1:] + thk[:-1]) will be the classic Mahaffy method,
    # but this can lead to mass conservation issue (see doi:10.5194/tc-7-229-2013)
    # center scheme
    #thk_staggered = 0.5 * (thk[1:] + thk[:-1])
    # upwind scheme
    dir = np.sign(surf_grad)
    dir[dir == 0] = 1
    thk_staggered = 0.5 * np.abs(dir) * (thk[1:] + thk[:-1]) \
        + 0.5 * dir * (thk[1:] - thk[:-1])

    # staggered
    deform_vel = np.zeros(len(thk)-1)
    deform_vel = -2 / (glen_n + 2) * ice_softness * \
        np.power((rho_ice * g), glen_n) * np.power(thk_staggered, glen_n+1) \
        * np.power(np.abs(surf_grad), glen_n-1) * surf_grad
    
    deform_vel *= deform_e

    # staggered
    sliding_vel = np.zeros(len(thk)-1)
    eff_pres_staggered = 0.8 * rho_ice * g * thk_staggered
    basal_shear_staggered = rho_ice * g * thk_staggered * surf_grad
    sliding_vel = -sliding_constant / eff_pres_staggered \
        * np.power(np.abs(basal_shear_staggered), weertman_m-1) \
        * basal_shear_staggered

    sliding_vel *= sliding_e

    return deform_vel, sliding_vel

@_speed_up
def _update_mass_balance_numba_impl(topg, thk, mass_balance_beta, ela,
                                    snowfall_rate, secperyr):
    surf = thk + topg
    mb = mass_balance_beta * (surf - ela)
    mb[mb > snowfall_rate] = snowfall_rate
    mb = mb / secperyr

    return mb