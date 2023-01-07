from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


class Evaluator(object):

    def eval(self, y_true, y_pred, ind2label):
        self.ind2label = ind2label
        # y_true and y_pred should be ints
        true_labels = set(y_true)
        for ind in ind2label:
            assert ind in true_labels # this evaluation only works when the dataset contains all the labels! would go wrong when 

        #individual precision and recall
        pres, recs, f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels = list(sorted(ind2label.keys())), average = None);

        #total weighted precision recall
        pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels = list(sorted(ind2label.keys())), average = 'macro');

        total_eval = [pre,rec, f1]
        individual_eval = [pres, recs, f1s]


        return individual_eval, total_eval 


    def print_eval(self, eval, ind2label, best_eval = None):
        
        individual_eval, total_eval = eval

        individual_pre, individual_rec, individual_f1 = individual_eval

        if best_eval is not None:
            best_individual_eval, best_total_eval = best_eval

        to_print = []

        score_dict = {"tot":{"f1":"0", "pre":"0", "rec":"0"}}
        for i in ind2label:
            score_dict[ind2label[i]] = {"f1":"0", "pre":"0", "rec":"0"}

        for i in range(len(individual_pre)):
            class_name = ind2label[i]
            pre = individual_pre[i]
            pre_str = f"{pre*100:.2f}"
            rec = individual_rec[i]
            rec_str = f"{rec*100:.2f}"
            f1 = individual_f1[i]
            f1_str = f"{f1*100:.2f}"

            score_dict[class_name]["pre"] = pre_str
            score_dict[class_name]["rec"] = rec_str
            score_dict[class_name]["f1"] = f1_str


            if best_eval is not None:
                best_individual_pre, best_individual_rec, best_individual_f1 = best_individual_eval

                best_pre = best_individual_pre[i]
                best_pre_str = f"{best_pre*100:.2f}"
                best_rec = best_individual_rec[i]
                best_rec_str = f"{best_rec*100:.2f}"
                best_f1 = best_individual_f1[i]
                best_f1_str = f"{best_f1*100:.2f}"
                
                to_print.append([class_name, pre_str, rec_str, f1_str, best_pre_str, best_rec_str, best_f1_str])
            else:
                to_print.append([class_name, pre_str, rec_str, f1_str])

        pre, rec, f1 = total_eval
        pre_str = f"{pre*100:.2f}"
        rec_str = f"{rec*100:.2f}"
        f1_str = f"{f1*100:.2f}"

        score_dict["tot"]["pre"] = pre_str
        score_dict["tot"]["rec"] = rec_str
        score_dict["tot"]["f1"] = f1_str

        if best_eval is not None:
            best_pre_str = f"{best_total_eval[0]*100:.2f}"
            best_rec_str = f"{best_total_eval[1]*100:.2f}"
            best_f1_str = f"{best_total_eval[2]*100:.2f}"
            to_print.append(["tot", pre_str, rec_str, f1_str, best_pre_str, best_rec_str, best_f1_str])
        else:
            to_print.append(["tot", pre_str, rec_str, f1_str])
        
        print("\n")

        if best_eval is not None: 
            print_str = f"""{tabulate(to_print, headers = ["", "PRE", "REC", "F1", "BEST_PRE", "BEST_REC", "BEST_F1"])}"""
           
        else:
            print_str = f"""{tabulate(to_print, headers = ["", "PRE", "REC", "F1"])}"""
        print(print_str) # which is the table str

        general_performance = f"""{",".join([score_dict["tot"]["f1"], score_dict["pos"]["f1"], score_dict["neg"]["f1"],score_dict["neu"]["f1"] ,score_dict["pos"]["pre"], score_dict["pos"]["rec"], score_dict["neg"]["pre"], score_dict["neg"]["rec"], score_dict["neu"]["pre"], score_dict["neu"]["rec"]])}"""

        return print_str, general_performance





    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



    def img_confusion_matrix(self, temp_y_true, temp_y_pred, label2name):
        y_true = list(temp_y_true);
        y_pred = list(temp_y_pred);

        classnames = set();
        for l in label2name:
            classnames.add(label2name[l]);
        classnames = list(classnames);

        for i in range(len(y_true)):
            y_true[i] = label2name[y_true[i]];
            y_pred[i] = label2name[y_pred[i]];

        cnf_matrix = confusion_matrix(y_true, y_pred,labels =  classnames);
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classnames,title=title + " without normalization")
        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classnames, normalize=True,
                              title=title + ' Normalized confusion matrix')
        plt.show()

    def text_confusion_matrix(self, temp_y_true, temp_y_pred, ind2label):
        print("Confusion Matrix: ")
        labels = [ind2label[i] for i in range(len(ind2label))]

        y_true = []
        y_pred = []
        for i in range(len(temp_y_true)):
            y_true.append(ind2label[temp_y_true[i]]);
            y_pred.append(ind2label[temp_y_pred[i]]);

        cnf_matrix = [list(k) for k in confusion_matrix(y_true, y_pred, labels =  labels)]
        np.set_printoptions(precision=2)

        headers = [" "] + labels
        for i in range(len(cnf_matrix)):
            cnf_matrix[i] = [labels[i]] + cnf_matrix[i]
        print(tabulate(cnf_matrix, headers = headers))
        return str(tabulate(cnf_matrix, headers = headers))





