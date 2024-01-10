"""
AMLS 1 Final Assessment - main.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""


from A.TaskA import TaskA
from B.TaskB import TaskB


import sys


if __name__ == "__main__":

    #Only Accepts 2 Arguments after main.py
    if len(sys.argv) != 3:
        print("Usage: python main.py <Task_A_Function> <Task_B_Function>")

    if len(sys.argv) >= 3:

        task_a = sys.argv[1].lower()
        task_b = sys.argv[2].lower()
    else:
        task_a = "final"
        task_b = "final"

    #TASK A - Binary Classifcation
    task_a_instance = TaskA()

    if task_a == "svm":
        task_a_instance.run_svm()
    elif task_a == "svmhog":
        task_a_instance.run_svm_hog()
    elif task_a == "rf":
        task_a_instance.run_rf()
    elif task_a == "adaboost":
        task_a_instance.run_adaboost()
    elif task_a == "cnn" or task_a == "final" or task_a == "":
        task_a_instance.run_cnn()
    else:
        print(f"Unsupported input for Task A: {task_a}")

    #TASK B - Multi Class Classifcation
    task_b_instance = TaskB()

    if task_b == "taska":
        task_b_instance.run_nn(model_name="taska")
    elif task_b == "deep":
        task_b_instance.run_nn(model_name="deep")
    elif task_b == "resnet" or task_b == "final":
        task_b_instance.run_nn(model_name="resnet")

    else:
        print(f"Unsupported input for Task B: {task_b}")
