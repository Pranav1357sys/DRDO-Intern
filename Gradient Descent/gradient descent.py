# Not a working exact code for gradient descent, but a simple example of how it works. The code defines a simple logistic function and an error function, and then uses gradient descent to find the optimal parameters w and b that minimize the error. The grab_w and grab_b functions calculate the gradients for w and b, respectively, which are then used to update the parameters in the do_gradient_descent function.
import numpy as np

X=[0.5,2.5]
Y=[0.2,0.9]

def f(w,x,b):
    return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,x,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,x,b)
        err += 0.5*(fx-y)**2
    return err

def grab_b(w,x,b,y):
    fx = f(w,x,b)
    return (fx-y) * fx * (1-fx)

def grab_w(w,x,b,y):
    fx = f(w,x,b)
    return (fx-y) * fx * (1-fx) * x

def do_gradient_descent():
    w,b,eta,max_epochs = -2,-2,1.0,1000
    for i in range(max_epochs):
        dw,db = 0,0
        for x,y in zip(X,Y):
            dw += grab_w(w,x,b,y)
            db += grab_b(w,x,b,y)
        w=w-eta*dw
        b=b-eta*db
    return w,b