from sklearn.neural_network import MLPClassifier;
import datacollection as dc;


def main():
    x = dc.collectTrainingX();
    y = dc.collectTrainingY();

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1);

    print('training on dataset');
    clf.fit(x,y);
    print('training on dataset complete');
    
    print('testing on same dataset');
    for x_i in x:
        print(clf.predict([x_i]));
    


if __name__ == "__main__":
    main();
    