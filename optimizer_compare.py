import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

#--------Build tensorflow for equation 0.5*x^2+2.5*y^2
#        Now is 20+x^2+y^2−10 cos(2πx)−10 cos(2πy) Hehe 20+x**2+y**2-10*np.cos(2*np.pi*x)-10*np.cos(2*np.pi*y)
#        Now is (x^2+y-11)^2 + (x+y^2-7)^2 --> (x**2+y-11)**2,(x+y**2-7)**2
#        We got (2*x**2-1.05*x**4),((x**6)/6+x*y+y**2)
#        also np.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))
x = tf.Variable(0.0)
y = tf.Variable(-3.0)
output =tf.experimental.numpy.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

#-------Chosing different optimizer and different learning rate
train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate = 0.02, momentum=0.9).minimize(output)

#-------Session start
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#-------only do 10 step training
epochs = 40
epoch = 0
array_x = -7
xtt=[0.0]
ytt=[-3.0]
while epoch < epochs:
    sess.run(train_step)
    if epoch % 1 == 0:
        array_x = x.eval(sess)
        array_y = y.eval(sess)
        xtt.append(array_x)
        ytt.append(array_y)
    epoch+=1


#        Now is (x^2+y-11)^2 + (x+y^2-7)^2
x = tf.Variable(0.0)
y = tf.Variable(-3.0)
output = tf.experimental.numpy.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

#-------Chosing different optimizer and different learning rate
train_step = tf.compat.v1.train.AdagradOptimizer(learning_rate = 0.5).minimize(output)


#-------Session start
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#-------only do 10 step training
epochs = 40
epoch = 0
array_x = -7
xtadag=[0.0]
ytadag=[-3.0]
while epoch < epochs:
    sess.run(train_step)
    if epoch % 1 == 0:
        array_x = x.eval(sess)
        array_y = y.eval(sess)
        xtadag.append(array_x)
        ytadag.append(array_y)
    epoch+=1

#        Now is (x^2+y-11)^2 + (x+y^2-7)^2
x = tf.Variable(0.0)
y = tf.Variable(-3.0)
output = tf.experimental.numpy.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

#-------Chosing different optimizer and different learning rate
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.02).minimize(output)

#-------Session start
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#-------only do 10 step training
epochs = 40
epoch = 0
array_x = -7
xtsgd=[0.0]
ytsgd=[-3.0]
while epoch < epochs:
    sess.run(train_step)
    if epoch % 1 == 0:
        array_x = x.eval(sess)
        array_y = y.eval(sess)
        xtsgd.append(array_x)
        ytsgd.append(array_y)
    epoch+=1


#        Now is (x^2+y-11)^2 + (x+y^2-7)^2
x = tf.Variable(0.0)
y = tf.Variable(-3.0)
output = tf.experimental.numpy.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

#-------Chosing different optimizer and different learning rate
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.1).minimize(output)

#-------Session start
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#-------only do 10 step training
epochs = 40
epoch = 0
array_x = -7
xtadam=[0.0]
ytadam=[-3.0]
while epoch < epochs:
    sess.run(train_step)
    if epoch % 1 == 0:
        array_x = x.eval(sess)
        array_y = y.eval(sess)
        xtadam.append(array_x)
        ytadam.append(array_y)
    epoch+=1
#        Now is (x^2+y-11)^2 + (x+y^2-7)^2
x = tf.Variable(0.0)
y = tf.Variable(-3.0)
output = tf.experimental.numpy.log10( tf.multiply(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2),30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

#-------Chosing different optimizer and different learning rate
train_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.1).minimize(output)

#-------Session start
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#-------only do 10 step training
epochs = 40
epoch = 0
array_x = -7
xtrms=[0.0]
ytrms=[-3.0]
while epoch < epochs:
    sess.run(train_step)
    if epoch % 1 == 0:
        array_x = x.eval(sess)
        array_y = y.eval(sess)
        xtrms.append(array_x)
        ytrms.append(array_y)
    epoch+=1

# Himmelblau 函數
def obj_fun(x, y):
    return np.log10((1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))

# 設定繪圖範圍
x = np.linspace(-2, 2, 400)
y = np.linspace(-3, 1, 400)
X, Y = np.meshgrid(x, y)
Z = obj_fun(X, Y)

# 假設不同優化器的收斂路徑
mmt_path = np.array([[xtt[t],ytt[t]] for t in range(0,len(xtt))])  # Momentum 的軌跡
adagrad_path = np.array([[xtadag[t],ytadag[t]] for t in range(0,len(xtadag))])  # adagrad 的軌跡
adam_path = np.array([[xtadam[t],ytadam[t]] for t in range(0,len(xtadam))])  # Adam 的軌跡
sgd_path = np.array([[xtsgd[t],ytsgd[t]] for t in range(0,len(xtsgd))])  #  SGD 的軌跡
rms_path = np.array([[xtrms[t],ytrms[t]] for t in range(0,len(xtrms))]) # RMS 的軌跡的軌跡
# 繪製等高線
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap="plasma")
plt.colorbar(label="Function Value")
plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)

# 繪製不同優化器的收斂軌跡
plt.plot(mmt_path[:, 0], mmt_path[:, 1], 'r-x', label="Momentum Path")
plt.plot(adagrad_path[:, 0], adagrad_path[:, 1], 'C1-x', label="Adagrad Path")
plt.plot(adam_path[:, 0], adam_path[:, 1], 'C2-x', label="Adam Path")
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'y-x', label="SGD Path")
plt.plot(rms_path[:, 0], rms_path[:, 1], 'C9-x', label="RMSprop Path")

# 標記局部最小值
minima = np.array([[0.0, -1.0]])
plt.scatter(minima[:, 0], minima[:, 1], color='yellow', edgecolors='black', s=100, label="Local Minimum")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Optimization Paths on Goldstein-Price function Function")
plt.legend(loc=1)
plt.show()
