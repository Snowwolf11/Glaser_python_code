
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from mpl_toolkits.mplot3d import Axes3D

#create plot of Bloch sphere
b1 = Bloch()

psi = (3*basis(2, 0) + ( 1j-1)*basis(2, 1)).unit() 
psi_proj = (np.sqrt(2)*basis(2, 0) + ( 1j-1)*basis(2, 1)).unit()/1.1

b1.add_states([psi, psi_proj])

b1.add_arc((basis(2,0)+basis(2,1)).unit()/1.5, psi_proj.unit()/1.5, color='black')
b1.add_arc((basis(2,0)).unit()/1.5, psi.unit()/1.5, color='black')

b1.font_size = 18
b1.vector_width = 2.5
b1.frame_alpha = 0.1
b1.frame_width = 0.8
b1.vector_color = ['#FF0000','#999999']

b1.xlabel = [r'x, $\left|+\right>$', r'$\left|-\right>$']
b1.ylabel = [r'$y, \left|+i\right>$', r'$\left|-i\right>$']
b1.zlabel = [r'$z, \left|0\right>$', r'$\left|1\right>$']

b1.show()
plt.show()

#create Bloch for B1 inhomogentities and offresonance
b2 = Bloch()
b3 = Bloch()
b4 = Bloch()

b2.font_size = 18
b2.vector_width = 2.5
b2.frame_alpha = 0.1
b2.frame_width = 0.8
b2.vector_color = ['red', 'blue', 'green']
#b2.point_color = ['blue', 'red', 'green']
b2.xlabel = [r'x, $\left|+\right>$', r'$\left|-\right>$']
b2.ylabel = [r'$y, \left|+i\right>$', r'$\left|-i\right>$']
b2.zlabel = [r'$z, \left|0\right>$', r'$\left|1\right>$']

phi = np.pi/4
Omega = 0.667
B1_inho = 1.5
offs = 0.5

H0 = Omega*(np.cos(phi)*sigmax()/2+np.sin(phi)*sigmay()/2)
H_B1_inho = H0 * B1_inho
H_offs = H0 + offs*sigmaz()/2

psi0 = basis(2, 0).unit()

times = np.linspace(0.0, 4, 25)

result_H0 = mesolve(H0, psi0, times, e_ops=[sigmax(), sigmay(), sigmaz()])
result_H_B1_inho = mesolve(H_B1_inho, psi0, times, e_ops=[sigmax(), sigmay(), sigmaz()])
result_H_offs = mesolve(H_offs, psi0, times, e_ops=[sigmax(), sigmay(), sigmaz()])

b2.add_states([H_B1_inho, H0, H_offs])

b2.add_points([result_H0.expect[0],result_H0.expect[1],result_H0.expect[2]])
b2.add_points([result_H_B1_inho.expect[0],result_H_B1_inho.expect[1],result_H_B1_inho.expect[2]])
b2.add_points([result_H_offs.expect[0],result_H_offs.expect[1],result_H_offs.expect[2]])

b2.show()
plt.show()

#create plot for rotations added up
b3 = Bloch()

b3.font_size = 18
b3.vector_width = 2.5
b3.frame_alpha = 0.1
b3.frame_width = 0.8
b3.vector_color = ['blue', 'gray', 'gray', 'red', 'red', 'gray', 'gray', 'green', 'green', 'gray', 'gray', 'purple']
#b3.point_color = ['blue', 'red', 'green']
b3.xlabel = [r'x, $\left|+\right>$', r'$\left|-\right>$']
b3.ylabel = [r'$y, \left|+i\right>$', r'$\left|-i\right>$']
b3.zlabel = [r'$z, \left|0\right>$', r'$\left|1\right>$']

H1 = sigmax()
H2 = sigmaz()
H3 = sigmay()

psi0 = basis(2,0).unit()

times = np.linspace(0,np.pi/4, 4)

result_H1 = mesolve(H1, psi0, times)
result_H2 = mesolve(H2, result_H1.states[-1], times)
result_H3 = mesolve(H3, result_H2.states[-1], times)

b3.add_states(result_H1.states)
b3.add_states(result_H2.states)
b3.add_states(result_H3.states)

b3.add_arc(result_H1.states[0], result_H1.states[-1], color='green')
b3.add_arc(result_H2.states[0], result_H2.states[-1], color='blue')
b3.add_arc(result_H3.states[0], result_H3.states[-1], color='red')

b3.show()
plt.show()

result_H1_vec = mesolve(H1, psi0, times, e_ops=[sigmax(), sigmay(), sigmaz()])
result_H2_vec = mesolve(H2, result_H1.states[-1], times, e_ops=[sigmax(), sigmay(), sigmaz()])
result_H3_vec = mesolve(H3, result_H2.states[-1], times, e_ops=[sigmax(), sigmay(), sigmaz()])


x_vals = np.cumsum(np.concatenate((np.array([0]),result_H1_vec.expect[0],result_H2_vec.expect[0][1:],result_H3_vec.expect[0][1:])))
y_vals = np.cumsum(np.concatenate((np.array([0]),result_H1_vec.expect[1],result_H2_vec.expect[1][1:],result_H3_vec.expect[1][1:])))
z_vals = np.cumsum(np.concatenate((np.array([0]),result_H1_vec.expect[2],result_H2_vec.expect[2][1:],result_H3_vec.expect[2][1:])))

segment_colors = ['blue', 'gray', 'gray', 'red', 'gray', 'gray', 'green', 'gray', 'gray', 'purple']  # Example colors for each segment

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x_vals) - 1):
    ax.plot(x_vals[i:i+2], y_vals[i:i+2], z_vals[i:i+2], color=segment_colors[i % len(segment_colors)], marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


#do it correct

b4 = Bloch()

b4.font_size = 18
b4.vector_width = 2.5
b4.frame_alpha = 0.1
b4.frame_width = 0.8
b4.vector_color = ['blue', 'gray', 'gray', 'red', 'gray', 'gray', 'green', 'gray', 'gray', 'purple']
#b4.point_color = ['blue', 'red', 'green']
b4.xlabel = [r'x, $\left|+\right>$', r'$\left|-\right>$']
b4.ylabel = [r'$y, \left|+i\right>$', r'$\left|-i\right>$']
b4.zlabel = [r'$z, \left|0\right>$', r'$\left|1\right>$']

H1 = sigmax()
H2 = sigmay()
H3 = sigmax()

times = np.linspace(0,np.pi/4, 4)

U0_1 = propagator(H1, times)
U0_2 = propagator(H2, times)
U0_3 = propagator(H3, times)

U0 = U0_1[0]
U1 = U0_1[1]
U2 = U0_1[2]
U3 = U0_1[3]
U4 = U3*U0_2[1]
U5 = U3*U0_2[2]
U6 = U3*U0_2[3]
U7 = U6*U0_3[1]
U8 = U6*U0_3[2]
U9 = U6*U0_3[3]

psi0 = U0*basis(2,0).unit()
psi1 = U1*psi0
psi2 = U2*psi0
psi3 = U3*psi0
psi4 = U4*psi0
psi5 = U5*psi0
psi6 = U6*psi0
psi7 = U7*psi0
psi8 = U8*psi0
psi9 = U9*psi0

curve_states = [psi0.unit(),psi1.unit(),psi2.unit(),psi3.unit(),psi4.unit(),psi5.unit(),psi6.unit(),psi7.unit(),psi8.unit(),psi9.unit()]

print(curve_states)

#b4.add_states(curve_states)
b4.add_states(psi0.unit())
b4.add_states(psi1.unit())
b4.add_states(psi2.unit())
b4.add_states(psi3.unit())
b4.add_states(psi4.unit())
b4.add_states(psi5.unit())
b4.add_states(psi6.unit())
b4.add_states(psi7.unit())
b4.add_states(psi8.unit())
b4.add_states(psi9.unit())

b4.add_arc(psi0.unit(), psi3.unit(), color='green')
b4.add_arc(psi3.unit(), psi6.unit(), color='blue')
b4.add_arc(psi6.unit(), psi9.unit(), color='red')

b4.show()
plt.show()

x_vals = np.cumsum(expect(sigmax(), curve_states))
y_vals = np.cumsum(expect(sigmay(), curve_states))
z_vals = np.cumsum(expect(sigmaz(), curve_states))

segment_colors = ['blue', 'gray', 'gray', 'red', 'gray', 'gray', 'green', 'gray', 'gray', 'purple']  # Example colors for each segment

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x_vals) - 1):
    ax.plot(x_vals[i:i+2], y_vals[i:i+2], z_vals[i:i+2], color=segment_colors[i % len(segment_colors)], marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()