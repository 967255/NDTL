from utils_ode import *


'''
Def ode(lorenz, chen, lv)
'''
# 这组init_point 有问题, plot不出来
init_point = np.random.normal(loc=0.5, scale=0.1, size=(1, 3))
# init_point = np.array([-5.76,  2.27,  32.82])


# lorenz system
lorenz = ODE(name='lorenz', dimension=3, max_order=2)
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

turb_mode = 'rate'
turb_args = np.zeros(lorenz.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(lorenz.coef_init)
turb_args[:, :, 1] = np.ones(lorenz.coef_shape) * 0.1

lorenz.set_turb(turb_mode, turb_args)



# chen system
chen = ODE(name='chen')
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

turb_mode = 'rate'
turb_args = np.zeros(chen.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(chen.coef_init)
turb_args[:, :, 1] = np.ones(chen.coef_shape) * 0.1

chen.set_turb(turb_mode, turb_args)


# lv system
lv = ODE(name='lv')
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

turb_mode = 'rate'
turb_args = np.zeros(lv.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(lv.coef_init)
turb_args[:, :, 1] = np.ones(lv.coef_shape) * 0.1

lv.set_turb(turb_mode, turb_args)




'''
Def ode(seirs)
'''
seirs = ODE(name='seirs', variables=['S', 'E', 'I'], max_order=2,
          extra_term_dicts={'R': R})


seirs._beta = 0.9  # 感染率，即一个感染者在单位时间内感染易感者的概率
seirs._sigma = 0.1  # 潜伏期倒数，即一个暴露者转变为感染者的速率
seirs._gamma = 0.2  # 康复率，即一个感染者在单位时间内康复并获得免疫力的概率
seirs._xi = 0.15  # 失去免疫率，即一个康复者在单位时间内失去免疫力的概率
seirs._N = 1  # 初始总人数

seirs.set_coef(0, 'SI', -seirs._beta/seirs._N)
seirs.set_coef(0, 'R', seirs._xi)

seirs.set_coef(1, 'E', -seirs._sigma)
seirs.set_coef(1, 'SI', seirs._beta/seirs._N)

seirs.set_coef(2, 'I', -seirs._gamma) 
seirs.set_coef(2, 'E', seirs._sigma)

turb_mode = 'uniform'
turb_args = np.zeros(seirs.coef_shape + (2, ))
for i in range(seirs.coef_shape[0]):
    for j in range(seirs.coef_shape[1]):
        if np.abs(seirs.coef_init[i, j]) > EPSILON:
            # turb_args[i, j, 0] = 0.0
            if seirs.coef_init[i, j] > 0:
                turb_args[i, j, 1] = 1.0
            elif seirs.coef_init[i, j] < 0:
                turb_args[i, j, 1] = -1.0
                
seirs.set_turb(turb_mode, turb_args)



'''
Def ode(rossler)
'''
# rossler system
rossler = ODE(name='rossler', dimension=3, max_order=2)
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

turb_mode = 'rate'
turb_args = np.zeros(rossler.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(rossler.coef_init)
turb_args[:, :, 1] = np.ones(rossler.coef_shape) * 0.1

rossler.set_turb(turb_mode, turb_args)

# print(rossler.info)
# rossler.plot(num_points=20000)



'''
Def ode(prs)
'''
# prs system
prs = ODE(name='prs', dimension=3, max_order=2)
# dx = xy - xz
prs.set_coef(0, 'xy', 1)
prs.set_coef(0, 'xz', -1)
# dy = yz - xy
prs.set_coef(1, 'yz', 1)
prs.set_coef(1, 'xy', -1)
# dz = xz - yz
prs.set_coef(2, 'xz', 1)
prs.set_coef(2, 'yz', -1)

turb_mode = 'rate'
turb_args = np.zeros(prs.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(prs.coef_init)
turb_args[:, :, 1] = np.ones(prs.coef_shape) * 0.1

prs.set_turb(turb_mode, turb_args)

# print(prs.info)
# prs.plot(num_points=5000)


'''
Create ode_list = [lorenz, chen, lv]
'''
ode_list = [lorenz, chen, lv]



'''
Normalize ode to target grid: Create ode_shifted_list = [lorenz_shifted, chen_shifted, lv_shifted, seris, rossler_shifted, prs]
'''
grid_limits = [[0, 1], [0, 1], [0, 1]]
grid_limits_origin = [[-25, 25], [-25, 25], [0, 50]]
grid_shape = (48, 48, 48)


ode_shifted_list = []
for i_ode in [0, 1, 2]:
    ode_shifted = ode_change_grid(ode_origin=ode_list[i_ode], grid_limits_origin=grid_limits_origin, grid_limits_target=grid_limits, grid_shape=grid_shape)
    ode_shifted.name += '_shifted'
    ode_shifted_list.append(ode_shifted)

ode_shifted_list.append(seirs)

### normalize rossler and add rossler and prs
grid_limits = [[0, 1], [0, 1], [0, 1]]
grid_limits_origin = [[-12, -12], [-12, 12], [0, 24]]
grid_shape = (48, 48, 48)

rossler_shifted = ode_change_grid(ode_origin=rossler, grid_limits_origin=grid_limits_origin, grid_limits_target=grid_limits, grid_shape=grid_shape)
rossler_shifted.name += '_shifted'

ode_shifted_list.append(rossler_shifted)

ode_shifted_list.append(prs)


'''
SHOW
'''
# lorenz.plot(num_points=5000)
# lorenz.show()
# chen.plot(num_points=5000)
# chen.show()
# lv.plot(num_points=5000)
# lv.show()

'''
CHECK
'''

if __name__ == "__main__":
    for ode in ode_shifted_list:
        print(ode.name)