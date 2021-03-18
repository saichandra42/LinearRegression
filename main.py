import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class LinearRegression:
  # class attribute
  epochs = 8000

  def __init__(self,x,y):
    self.x = x
    self.y = y
    self.theta0 = 0
    self.theta1 = 0
    self.theta = [self.theta0, self.theta1]
    self.lr = 0.0001
    self.n_examples = len(x)
    self.loss_array = []

  def __str__(self):
    return 'Class initialised for linear regression'

  def calculateCost(self):
    theta = np.array(self.theta)
    height = np.array(self.x)
    ones = np.ones(self.n_examples)
    height = np.c_[height,ones]
    prediction = np.dot(height,theta)
    loss_array = np.square(prediction - self.y)
    loss = np.sum(loss_array)/(2*self.n_examples)
    return loss

  def epoch(self):
    height = np.array(self.x)
    ones = np.ones(self.n_examples)
    height = np.c_[height,ones]
    prediction = np.dot(height,self.theta)
    partial_derivative_theta0 = (1/self.n_examples) * np.sum((prediction - self.y))
    partial_derivative_theta1 = (1/self.n_examples) * np.sum(np.dot(self.x,(prediction - self.y)))
    self.theta0 = self.theta0 - (self.lr*partial_derivative_theta0)
    self.theta1 = self.theta1 - (self.lr*partial_derivative_theta1)
    self.theta = [self.theta0, self.theta1]
    loss = self.calculateCost()
    self.loss_array.append(loss)



  def optimise(self):
    for i in range(0,self.epochs):
      self.epoch()

  def showOutput(self):
    theta = np.array([self.theta0, self.theta1])
    prediction = np.dot(self.x,theta)
    plt.scatter(self.x,self.y,color='blue',label='original')
    plt.plot(self.x,prediction,color='red',label='prediction')
    plt.legend()
    plt.show()

data = pd.read_csv('data.csv')

lr1 = LinearRegression(data['Height'],data['Weight'])
lr1.optimise()
lr1.showOutput() 



