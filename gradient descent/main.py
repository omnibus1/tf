import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import numpy as np
import tensorflow as tf
import random

def f1(x):
    return (x-5.0)**2

def f2(x):
    return 60*tf.sin(x)

def f3(x):
    return 0.6*f1(x)+0.5*f2(x)

def f4(x,y):
    return 1.0*tf.sin(0.5*x**2-0.25*y**2+3)*tf.cos(2*x+1+2**y)


def main():
    
    x=np.linspace(-3,3,100)
    y=np.linspace(-3,3,100)
    current_x=tf.Variable(random.uniform(-3, 3))
    current_y=tf.Variable(random.uniform(-3, 3))
    X,Y=np.meshgrid(x,y)
    Z=f4(X,Y)
    fig,ax=plt.subplots()
    optimiser=tf.keras.optimizers.SGD(learning_rate=0.2,momentum=0.8)

    def step(i):
        nonlocal current_x
        nonlocal current_y
        with tf.GradientTape() as tape:
            current_z=f4(current_x,current_y)
        ax.clear()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.scatter(current_x.numpy(),current_y.numpy())
        gradients=tape.gradient(current_z,[current_x,current_y])
        print(f'current value: {f4(current_x,current_y).numpy()}')
        optimiser.apply_gradients(zip(gradients,[current_x,current_y]))

    animation=mpl_animation.FuncAnimation(fig,step,interval=50)
    plt.show()

if __name__=='__main__':
    main()
    

