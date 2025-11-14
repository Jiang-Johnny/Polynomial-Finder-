import numpy as np
import matplotlib.pyplot as plt
import argparse

class polyhunt:

  def __init__(self):

    pass

  def main(self):

    parser = argparse.ArgumentParser(description="polyhunt")
    parser.add_argument('--input', type=str, help="Please input the data file")
    arg = parser.parse_args()

    data = np.loadtxt(arg.input)

    X = data[:, 0]
    y = data[:, 1]

    #Create training and testing split in 70/30 form for the rest of the methods
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    ##This is how u can manually use methods
    ##print(self.lrSolver(X_train, y_train, 5, 0.001))


  def closedFormSolver(self,X,y,deg,lamb):
    #lamb is lambda (the regularization constant)
    x = np.array(X)
    t = np.array(y)

    #Below is where we create the VanDerMonde Matrix
    matrix = np.column_stack([x**i for i in range(deg+1)])

    I = np.identity(deg+1)

    A=(matrix.T @ matrix + lamb * I)

    W = np.linalg.solve(A,matrix.T @ t)

    return W

  #This will be for stopping condition 1
  def gradientDescentSolver1(self,X,y,deg,lamb,lr):

    x = np.array(X)
    t = np.array(y)

    #initialize the weight vector first
    W = np.zeros(deg+1)

    matrix = np.column_stack([x**i for i in range(deg+1)])

    epochs = 100

    #Stopping condition 1 gradient magnitude
    #norm of the loss_change squared is less than minimum threshold = 0.0001
    for  i in range(epochs):

      loss_change = matrix.T @ (matrix @ W - t) + lamb* W
      
      loss_change = np.clip(loss_change, -1e5, 1e5)

      W = W - lr * loss_change
      
      W = np.clip(W, -1e5, 1e5)

      if np.linalg.norm(loss_change)**2 < 0.0001:

        break

    return W

   #This will be for stopping condition 2
  def gradientDescentSolver2(self,X,y,deg,lamb,lr):

     x = np.array(X)
     t = np.array(y)

     prev_loss = 1e5

     W = np.zeros(deg+1)

     matrix = np.column_stack([x**i for i in range(deg+1)])


     epochs = 100

    #Stopping condition 2 gradient magnitude
    #norm of the loss_change squared is less than minimum threshold = 0.0001
     for i in range(epochs):

      loss_change = matrix.T @ (matrix @ W - t) + lamb* W

      #Added loss for stopping condition 2
      loss = np.linalg.norm(matrix @ W - t)**2 + lamb* np.linalg.norm(W)**2

      loss_change = np.clip(loss_change, -1e5, 1e5)

      W = W - lr * loss_change
      
      W = np.clip(W, -1e5, 1e5)

      #Stopping condition 2
      if np.abs(loss - prev_loss) < 0.0001:

        break

      prev_loss = loss

     return W

  def modelSelection(self,X,y,lr,solver):

    lambdas = [0.001,0.01,0.1,1,10]

    #stores best lambda per degree
    #index 0 represent best lambda for degree 1
    current_lamb = 0.001
    best_lamb = []

    #creating 70/30 splits for comparing results
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(X))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    d = 1

    x = np.array(X_train)
    t = np.array(y_train)
    test_x = np.array(X_test)
    test_t = np.array(y_test)

    low = 1e5
    low_lamb_err = 1e5

    for deg in range(1,11):

      current_lamb = 0.001

      low_lamb_err = 1e5

      for l in lambdas:

        if solver == self.closedFormSolver:
          W = self.closedFormSolver(X_train,y_train,deg,l)
        else:
          W = solver(X_train,y_train,deg,l,lr)

          matrix = np.column_stack([test_x**i for i in range(deg+1)])

          error = test_t - matrix @ W
          
          error = np.clip(error, -1e5, 1e5)

          error_square = error**2

          mse = np.mean(error_square)

          if mse < low_lamb_err:

            low_lamb_err = mse

            current_lamb = l

      best_lamb.append(current_lamb)

      if solver == self.closedFormSolver:
        weight = self.closedFormSolver(X_train,y_train,deg,current_lamb)
      else:
        weight = solver(X_train,y_train,deg,current_lamb,lr)

      matrix = np.column_stack([test_x**i for i in range(deg+1)])

      error = test_t - matrix @ weight
      
      error = np.clip(error, -1e5, 1e5)

      error_square = error**2

      mse = np.mean(error_square)

      if mse < low:

        low = mse

        d = deg

    return d

  def modelSelectionLamb(self,X,y,lr,solver):

    lambdas = [0.001,0.01,0.1,1,10]

    #stores best lambda per degree
    #index 0 represent best lambda for degree 1
    current_lamb = 0.001
    best_lamb = []

    #creating 70/30 splits for comparing results
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(X))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    d = 1

    x = np.array(X_train)
    t = np.array(y_train)
    test_x = np.array(X_test)
    test_t = np.array(y_test)

    low = 1e5
    low_lamb_err = 1e5

    for deg in range(1,11):

      current_lamb = 0.001

      low_lamb_err = 1e5

      for l in lambdas:

        if solver == self.closedFormSolver:
          W = self.closedFormSolver(X_train,y_train,deg,l)
        else:
          W = solver(X_train,y_train,deg,l,lr)

          matrix = np.column_stack([test_x**i for i in range(deg+1)])

          error = test_t - matrix @ W
          
          error = np.clip(error, -1e5, 1e5)

          error_square = error**2

          mse = np.mean(error_square)

          if mse < low_lamb_err:

            low_lamb_err = mse

            current_lamb = l

      best_lamb.append(current_lamb)

      if solver == self.closedFormSolver:
        weight = self.closedFormSolver(X_train,y_train,deg,current_lamb)
      else:
        weight = solver(X_train,y_train,deg,current_lamb,lr)

      matrix = np.column_stack([test_x**i for i in range(deg+1)])

      error = test_t - matrix @ weight
      
      error = np.clip(error, -1e5, 1e5)

      error_square = error**2

      mse = np.mean(error_square)

      if mse < low:

        low = mse

        d = deg

    return best_lamb[d-1]

  def lrSolver(self,X,y,deg,lamb):

    x = np.array(X)
    t = np.array(y)

    percent_loss = 1e5
    epochs = 10000
    lr = [0.0000001,0.00001,0.0001,0.001,0.01,0.1,1,10]

    best_lr = lr[0]   
    best_loss = 1e5 

    W = np.zeros(deg+1)
    weight = self.closedFormSolver(X,y,deg,lamb)

    for l in lr:

      for i in range(epochs):

        matrix = np.column_stack([x**j for j in range(deg+1)])
        gradient = matrix.T @ (matrix @ W - t) + lamb* W
        gradient = np.clip(gradient, -1e5, 1e5)

        W = W - l * gradient
        W = np.clip(W, -1e5, 1e5)

        close_error = t - matrix @ weight
        close_error = np.clip(close_error, -1e5, 1e5)
        close_mse = np.mean(close_error**2)

        error = t - matrix @ W
        error = np.clip(error, -1e5, 1e5)
        mse = np.mean(error**2)

        percent_loss = np.abs((mse - close_mse)/close_mse)

        if percent_loss < best_loss:

            best_loss = percent_loss
            best_lr = l

        if percent_loss < 100:

            W = np.zeros(deg+1)

            break

    return best_lr

  def lrSolverEpoch(self,X,y,deg,lamb):

    x = np.array(X)
    t = np.array(y)

    percent_loss = 1e5
    epochs = 10000
    lr = [0.00001,0.0001,0.001,0.01,0.1,1,10]

    best_lr = 0.001
    best_epoch = epochs   

    low = 1e5

    W = np.zeros(deg+1)
    weight = self.closedFormSolver(X,y,deg,lamb)

    for l in lr:

      for i in range(epochs):

        matrix = np.column_stack([x**j for j in range(deg+1)])
        gradient = matrix.T @ (matrix @ W - t) + lamb* W
        gradient = np.clip(gradient, -1e5, 1e5)

        W = W - l * gradient
        W = np.clip(W, -1e5, 1e5)

        close_error = t - matrix @ weight
        close_error = np.clip(close_error, -1e5, 1e5)
        close_error_square = close_error**2
        close_mse = np.mean(close_error_square)

        error = t - matrix @ W
        error = np.clip(error, -1e5, 1e5)
        error_square = error**2
        mse = np.mean(error_square)

        percent_loss = np.abs((mse - close_mse)/close_mse)

        if percent_loss < 100:

          if i < best_epoch:

            best_epoch = i
            best_lr = l

          W = np.zeros(deg+1)

          break

    return best_epoch


if __name__ == "__main__":

  polyhunt().main()
