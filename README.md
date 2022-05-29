There should be four python files in this project: images.py, data.py, network.py, and main.py.

Follow these 3 steps.

1)
    To get the best test accuracy for class 0 vs class 6, run the following in the terminal:

        python3 main.py --batch-size 4 --learning-rate 0.0001

    To get the best test accuracy for class 2 vs class 6, run the following in the terminal:

        python3 main.py --batch-size 4 --learning-rate 0.0001

    To get the best test accuracy for all classes, run the following in the terminal:
    
        python3 main.py --batch-size 1 --learning-rate 0.0003

2)
    After running one of these commands, the following will be displayed on the terminal:

        batch size =  [BATCH-SIZE]
        epochs =  [EPOCHS]
        learning rate =  [LEARNING-RATE]
        k-folds =  [K-FOLDS]

        Enter 1 for class 0 and class 6 binary classification
        Enter 2 for class 2 and class 6 binary classification
        Enter 3 for all class multiclass classification
        Enter:

    Please enter 1, 2, or 3 depending on what you want to classify, and also depending on what arguments you passed when you ran the program in step 1. 

    If you passed:

        python3 main.py --batch-size 4 --learning-rate 0.0001

    Then enter 1 or 2 since the best hyperparameters for both these classifiers are the same.

    If you passed:

        python3 main.py --batch-size 1 --learning-rate 0.0003
    
    Then enter 3 for the multiclass classifier.

    After this is done, the program will start running.

3)
    After everything has been trained, validated, and tested, the terminal should display the test loss and the test accuracy. Finally, a plot containing the training and validation losses will also be generated and will pop up.
