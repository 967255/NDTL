from utils_ode import *
import torch


'''
Define several odes
'''
# lorenz system
lorenz = ODE(name='lorenz', dimension=3, max_order=2, grid_limits=[[-25,25],[-25,25],[0,50]])
lorenz._sigma, lorenz._b, lorenz._r = 10.0, 8/3, 28.0
# dx = -\sigma*x+\sigma*y
lorenz.set_coef(0, 'x', -lorenz._sigma)
lorenz.set_coef(0, 'y', lorenz._sigma)
# dy = -xz + rx - y
lorenz.set_coef(1, 'xz', -1)
lorenz.set_coef(1, 'x', lorenz._r)
lorenz.set_coef(1, 'y', -1)
# dz = xy - bz
lorenz.set_coef(2, 'xy', 1)
lorenz.set_coef(2, 'z', -lorenz._b)
# velocity estimation
lorenz.estimate_system_velocity(method='grid', n_per_axis=48)


# chen system
chen = ODE(name='chen', dimension=3, max_order=2, grid_limits=[[-25,25],[-25,25],[0,50]])
chen._a, chen._b, chen._c = 35.0, 3.0, 28.0
# dx = -ax+ay
chen.set_coef(0, 'x', -chen._a)
chen.set_coef(0, 'y', chen._a)
# dy = -xz + (c-a)x + cy
chen.set_coef(1, 'xz', -1)
chen.set_coef(1, 'x', chen._c - chen._a)
chen.set_coef(1, 'y', chen._c)
# dz = xy - bz
chen.set_coef(2, 'xy', 1)
chen.set_coef(2, 'z', -chen._b)
# velocity estimation
chen.estimate_system_velocity(method='grid', n_per_axis=48)


# lv system
lv = ODE(name='lv', dimension=3, max_order=2, grid_limits=[[-25,25],[-25,25],[0,50]])
lv._a, lv._b, lv._c = 36, 3, 27.5
# dx = -ax + ay
lv.set_coef(0, 'x', -lv._a)
lv.set_coef(0, 'y', lv._a)
# dy = - xz + cy
lv.set_coef(1, 'xz', -1)
lv.set_coef(1, 'y', lv._c)
# dz = xy - bz
lv.set_coef(2, 'xy', 1)
lv.set_coef(2, 'z', -lv._b)
# velocity estimation
lv.estimate_system_velocity(method='grid', n_per_axis=48)



# rossler system
rossler = ODE(name='rossler', dimension=3, max_order=2, grid_limits=[[-10,10],[-10,10],[0,20]])
rossler._a, rossler._b, rossler._c = 0.2, 0.2, 5.7
# dx = -y - z
rossler.set_coef(0, 'y', -1)
rossler.set_coef(0, 'z', -1)
# dy = x + ay
rossler.set_coef(1, 'x', 1)
rossler.set_coef(1, 'y', rossler._a)
# dz = b + xz - cz
rossler.set_coef(2, '', rossler._b)
rossler.set_coef(2, 'xz', 1)
rossler.set_coef(2, 'z', -rossler._c)
rossler.estimate_system_velocity(method='grid', n_per_axis=48)


# burke_shaw system
burke_shaw = ODE(name='burke_shaw', dimension=3, max_order=2, grid_limits=[[-2,2],[-2,2],[-2,2]])
burke_shaw._s, burke_shaw._v = 10, 4.272
# dx = -s ( x + y )
burke_shaw.set_coef(0, 'x', -burke_shaw._s)
burke_shaw.set_coef(0, 'y', -burke_shaw._s)
# dy = - y - s x z
burke_shaw.set_coef(1, 'y', -1)
burke_shaw.set_coef(1, 'xz', -burke_shaw._s)
# dz = s x y + v
burke_shaw.set_coef(2, '', burke_shaw._v)
burke_shaw.set_coef(2, 'xy', burke_shaw._s)
# burke_shaw.estimate_system_velocity(method='grid', n_per_axis=48)


# dequan_li system
dequan_li = ODE(name='dequan_li', dimension=3, max_order=2, grid_limits=[[-150,150],[-150,150],[-50,250]])
dequan_li._a, dequan_li._c, dequan_li._d, dequan_li._e, dequan_li._k, dequan_li._f = 40, 1.833, 0.16, 0.65, 55, 20
# dx = a ( y - x ) + d x z
dequan_li.set_coef(0, 'x', -dequan_li._a)
dequan_li.set_coef(0, 'y', dequan_li._a)
dequan_li.set_coef(0, 'xz', dequan_li._d)
# dy = k x + f y - z x
dequan_li.set_coef(1, 'x', dequan_li._k)
dequan_li.set_coef(1, 'y', dequan_li._f)
dequan_li.set_coef(1, 'xz', -1)
# dz = c z + x y - e x^2
dequan_li.set_coef(2, 'z', dequan_li._c)
dequan_li.set_coef(2, 'xy', 1)
dequan_li.set_coef(2, 'xx', -dequan_li._e)
# dequan_li.estimate_system_velocity(method='grid', n_per_axis=48)


# genesio_tesi system
# init_point=(0.1,0.1,0.1)
genesio_tesi = ODE(name='genesio_tesi', dimension=3, max_order=2, grid_limits=[[-1,1],[-1,1],[-1,1]])
genesio_tesi._a, genesio_tesi._b, genesio_tesi._c = 0.44, 1.1, 1.0
# dx = y
genesio_tesi.set_coef(0, 'y', 1)
# dy = z
genesio_tesi.set_coef(1, 'z', 1)
# dz = -c x - b y - a z + x^2
genesio_tesi.set_coef(2, 'x', -genesio_tesi._c)
genesio_tesi.set_coef(2, 'y', -genesio_tesi._b)
genesio_tesi.set_coef(2, 'z', -genesio_tesi._a)
genesio_tesi.set_coef(2, 'xx', 1)
# genesio_tesi.estimate_system_velocity(method='grid', n_per_axis=48)


# hadley system
# init_point=(0.1, 0, 0)
hadley = ODE(name='hadley', dimension=3, max_order=2, grid_limits=[[-2,2],[-2,2],[-2,2]])
hadley._a, hadley._b, hadley._f, hadley._g = 0.2, 4, 8, 1
# dx = - y^2 - z^2 - a x + a f
hadley.set_coef(0, 'yy', -1)
hadley.set_coef(0, 'zz', -1)
hadley.set_coef(0, 'x', -hadley._a)
hadley.set_coef(0, '', hadley._a * hadley._f)
# dy = x y - b x z - y + g
hadley.set_coef(1, 'xy', 1)
hadley.set_coef(1, 'xz', -hadley._b)
hadley.set_coef(1, 'y', -1)
hadley.set_coef(1, '', -hadley._g)
# dz = b x y + x z - z
hadley.set_coef(2, 'xy', hadley._b)
hadley.set_coef(2, 'xz', 1)
hadley.set_coef(2, 'z', -1)
# hadley.estimate_system_velocity(method='grid', n_per_axis=48)


# halvorsen system
halvorsen = ODE(name='halvorsen', dimension=3, max_order=2, grid_limits=[[-15,7.5],[-15,7.5],[-15,7.5]])
halvorsen._a = 1.4
# dx = - a x - 4 y - 4 z - y^2
halvorsen.set_coef(0, 'x', -halvorsen._a)
halvorsen.set_coef(0, 'y', -4)
halvorsen.set_coef(0, 'z', -4)
halvorsen.set_coef(0, 'yy', -1)
# dy = - a y - 4 z - 4 x - z^2
halvorsen.set_coef(1, 'y', -halvorsen._a)
halvorsen.set_coef(1, 'z', -4)
halvorsen.set_coef(1, 'x', -4)
halvorsen.set_coef(1, 'zz', -1)
# dz = - a z - 4 x - 4 y - x^2
halvorsen.set_coef(2, 'z', -halvorsen._a)
halvorsen.set_coef(2, 'x', -4)
halvorsen.set_coef(2, 'y', -4)
halvorsen.set_coef(2, 'xx', -1)
# halvorsen.estimate_system_velocity(method='grid', n_per_axis=48)


# newton_leipnik system
newton_leipnik = ODE(name='newton_leipnik', dimension=3, max_order=2, grid_limits=[[-1,1],[-1,1],[-1,1]])
newton_leipnik._a, newton_leipnik._b = 0.4, 0.175
# dx = - a x + y + 10 y z
newton_leipnik.set_coef(0, 'x', -newton_leipnik._a)
newton_leipnik.set_coef(0, 'y', 1)
newton_leipnik.set_coef(0, 'yz', 10)
# dy = - x - 0.4 y + 5 x z 
newton_leipnik.set_coef(1, 'x', -1)
newton_leipnik.set_coef(1, 'y', -0.4)
newton_leipnik.set_coef(1, 'xz', 5)
# dz = - b z - 5 x y
newton_leipnik.set_coef(2, 'z', newton_leipnik._b)
newton_leipnik.set_coef(2, 'xy', -5)
# newton_leipnik.estimate_system_velocity(method='grid', n_per_axis=48)


# nose_hoover system
# optional initial point (1, 0, 0)
nose_hoover = ODE(name='nose_hoover', dimension=3, max_order=2, grid_limits=[[-4,4],[-4,4],[-4,4]])
nose_hoover._a = 1.5 
# dx = y
nose_hoover.set_coef(0, 'y', 1)
# dy = - x + y z
nose_hoover.set_coef(1, 'x', -1)
nose_hoover.set_coef(1, 'yz', 1)
# dz = a - y^2
nose_hoover.set_coef(2, '', nose_hoover._a)
nose_hoover.set_coef(2, 'yy', -1)
# nose_hoover.estimate_system_velocity(method='grid', n_per_axis=48)


# # rayleigh_benard system
# rayleigh_benard = ODE(name='rayleigh_benard', dimension=3, max_order=2, grid_limits=None)
# rayleigh_benard._a, rayleigh_benard._r, rayleigh_benard._b = 9, 12, 5
# # dx = - a x + a y
# rayleigh_benard.set_coef(0, 'x', -rayleigh_benard._a)
# rayleigh_benard.set_coef(0, 'y', rayleigh_benard._a)
# # dy = r x - y - x z
# rayleigh_benard.set_coef(1, 'x', rayleigh_benard._r)
# rayleigh_benard.set_coef(1, 'y', -1)
# rayleigh_benard.set_coef(1, 'xz', -1)
# # dz = x y - b z
# rayleigh_benard.set_coef(2, 'xy', 1)
# rayleigh_benard.set_coef(2, 'z', -rayleigh_benard._b)
# # rayleigh_benard.estimate_system_velocity(method='grid', n_per_axis=48)


# rucklidge system
rucklidge = ODE(name='rucklidge', dimension=3, max_order=2, grid_limits=[[-10,10],[-10,10],[0,20]])
rucklidge._k, rucklidge._a = 2, 6.7
# dx = - k x + a y - y z
rucklidge.set_coef(0, 'x', -rucklidge._k)
rucklidge.set_coef(0, 'y', rucklidge._a)
rucklidge.set_coef(0, 'yz', -1)
# dy = x
rucklidge.set_coef(1, 'x', 1)
# dz = -z + y^2
rucklidge.set_coef(2, 'z', -1)
rucklidge.set_coef(2, 'yy', 1)
# rucklidge.estimate_system_velocity(method='grid', n_per_axis=48)


# sakarya system
sakarya = ODE(name='sakarya', dimension=3, max_order=2, grid_limits=[[-20,20],[-20,20],[-20,20]])
sakarya._a, sakarya._b = 0.4, 0.3
# dx = - x + y + y z
sakarya.set_coef(0, 'x', -1)
sakarya.set_coef(0, 'y', 1)
sakarya.set_coef(0, 'yz', 1)
# dy = - x - y + a x z
sakarya.set_coef(1, 'x', -1)
sakarya.set_coef(1, 'y', -1)
sakarya.set_coef(1, 'xz', sakarya._a)
# dz = z - b x y
sakarya.set_coef(2, 'z', 1)
sakarya.set_coef(2, 'xy', -sakarya._b)
# sakarya.estimate_system_velocity(method='grid', n_per_axis=48)


# shimizu_morioka system
shimizu_morioka = ODE(name='shimizu_morioka', dimension=3, max_order=2, grid_limits=[[-1.5,1.5],[-1.5,1.5],[0,3]])
shimizu_morioka._a, shimizu_morioka._b = 0.75, 0.45
# dx = y
shimizu_morioka.set_coef(0, 'y', 1)
# dy = x - a y - x z
shimizu_morioka.set_coef(1, 'x', 1)
shimizu_morioka.set_coef(1, 'y', -shimizu_morioka._a)
shimizu_morioka.set_coef(1, 'xz', -1)
# dz =  x^2 - b 
shimizu_morioka.set_coef(2, 'xx', 1)
shimizu_morioka.set_coef(2, 'z', -shimizu_morioka._b)
# shimizu_morioka.estimate_system_velocity(method='grid', n_per_axis=48)


# #  system
#  = ODE(name='', dimension=3, max_order=2, grid_limits=None)
# ._a, ._b = 
# # dx = 
# .set_coef(0, 'x', )
# # dy = 
# .set_coef(1, 'x', )
# # dz = 
# .set_coef(2, 'x', )
# # .estimate_system_velocity(method='grid', n_per_axis=48)


class DataSampler(object):
    def __init__(self, grid_shape, grid_limits, turb_scale):
        self.odes = []

        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.turb_scale = turb_scale
        self.grids = calc_grids(grid_limits=grid_limits, grid_shape=grid_shape)

        self.max_order = 0 
        self.dimension = 0
        self.extra_term_dicts = {}
        
        self.X = np.array([x for x in product(*self.grids)]).reshape((-1, len(self.grids)))


    def get_batch(self, batch_size, indexes=None, turb=True, turb_scale=None):
        vecs = []
        if indexes is None:
            indexes = np.random.randint(len(self.odes), size=(batch_size, ))
        elif isinstance(indexes, (int, float)):
            indexes = np.array([indexes]) 

        if turb_scale is None:
            turb_scale = self.turb_scale

        for index in indexes:
            # if turb:
            #     self.odes[index].random_coef()
            # else:
            #     self.odes[index].reset_coef()

            vec = ode2vec(ode=self.odes[index], grids=self.grids)
            if turb:
                vec += turb_scale * np.random.randn(*vec.shape)

            vecs.append(vec)
        
        # self.odes[index].reset_coef()
        vecs = np.array(vecs)
        vecs = vecs.transpose(0, 4, 1, 2, 3)  # (batch_size, channels, D, H, W)
        return torch.tensor(vecs, dtype=torch.float32), [self.odes[index].name for index in indexes]

    def add_odes(self, odes):
        for ode in odes:
            self.odes.append(ode)
            if self.max_order < ode.max_order:
                self.max_order = ode.max_order
            if self.dimension < ode.dimension:
                self.dimension = ode.dimension   
            self.extra_term_dicts.update(ode.extra_term_dicts)
            print(f'ODE {ode.name} added to Data Sampler!')
        
    def remove_all_odes(self):
        self.odes = []
        print('All ODEs removed')


if __name__ == '__main__':
    # print(shimizu_morioka.get_info())
    # shimizu_morioka.plot(pic_save_dir='./shimizu_morioka')
    # # newton_leipnik.plot(init_point=(0.349, 0, -0.16), pic_save_dir='./newton_leipnik')
    print('ode bank!')
