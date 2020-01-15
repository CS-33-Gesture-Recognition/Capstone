from sklearn.neural_network import MLPClassifier;
import datacollection as dc;
import numpy as np;


def main():
    x = dc.collectTrainingX();
    y = dc.collectTrainingY();

    print('y size: ', len(y));
    print('x dims: ', x.shape);

    print('y: ', y);
    print("x: ", x);

    ex_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    ex_y = [np.array([4.]), np.array([2.])]


    print('ex_y: ', ex_y);
    print("ex_x: ", ex_x);


if __name__ == "__main__":
    main();
