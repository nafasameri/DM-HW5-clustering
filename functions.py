from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


def read_data():
    iris = load_iris()
    X = iris.data
    return X


def normalize(X):
    # نرمال‌سازی داده‌ها با استفاده از روش Min-Max
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # ساخت ماتریس شباهت با استفاده از فاصله اقلیدسی
    similarity_matrix = pairwise_distances(X_normalized, metric='euclidean')

    # نمایش ماتریس شباهت
    print(similarity_matrix)


def kmeans(X):
    # ساخت مدل k-means با تعداد خوشه‌های مورد نظر (اینجا سه خوشه)
    kmeans = KMeans(n_clusters=3, random_state=42)

    # آموزش مدل بر روی داده‌ها
    kmeans.fit(X)

    # برچسب‌های خوشه‌ها برای نمونه‌ها
    labels = kmeans.labels_

    # نمایش برچسب‌ها
    print(labels)


def dbscan(X):
    # اجرای الگوریتم DBSCAN با پارامترهای مناسب
    dbscan = DBSCAN(eps=0.3, min_samples=5)

    # خوشه‌بندی داده‌ها
    dbscan.fit(X)

    # برچسب‌های خوشه‌ها برای نمونه‌ها (-1 برای نمونه‌هایی که به هیچ خوشه‌ای تعلق نمی‌گیرند)
    labels = dbscan.labels_

    # نمایش برچسب‌ها
    print(labels)


def test_data(clf, X_Test, Y_Test, method):
    y_pred = clf.predict(X_Test)
    # print("prediction labels:", y_pred)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(Y_Test, y_pred)
    AUC_ROC = roc_auc_score(Y_Test, y_pred)
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig('results/' + method + "-ROC.png")
    # plt.show()

    precision, recall, thresholds = precision_recall_curve(Y_Test, y_pred)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("Area under Precision-Recall curve:", AUC_prec_rec)
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig('results/' + method + "-Precision_recall.png")
    # plt.show()

    confusion = confusion_matrix(Y_Test, y_pred)
    print("confusion_matrix: ", confusion)
    tn, fp, fn, tp = confusion.ravel()

    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(tp + tn) / float(np.sum(confusion))
    print("Accuracy: " + str(accuracy))

    specificity = 0
    if float(tn + fp) != 0:
        specificity = float(tn) / float(tn + fp)
    print("Specificity: " + str(specificity))

    sensitivity = 0
    if float(tp + fn) != 0:
        sensitivity = float(tp) / float(tp + fn)
    print("Sensitivity: " + str(sensitivity))

    precision = 0
    if float(tp + fp) != 0:
        precision = float(tp) / float(tp + fp)
    print("Precision: " + str(precision))

    NPV = 0
    if float(tn + fn) != 0:
        NPV = float(tn) / float(tn + fn)
    print("NPV: " + str(NPV))

    f1score = 0
    if float(tp + fp + fn) != 0:
        f1score = float((2. * tp)) / float((2. * tp) + fp + fn)
    print("F1-Score: " + str(f1score))

    error_rate = 0
    if float(np.sum(confusion)) != 0:
        error_rate = float(fp + fn) / float(np.sum(confusion))
    print("Error Rate: " + str(error_rate))

    jaccard_index = jaccard_score(Y_Test, y_pred, average='weighted')
    print("Jaccard similarity score: " + str(jaccard_index))

    corrcoef = matthews_corrcoef(Y_Test, y_pred)
    print("The Matthews correlation coefficient: " + str(corrcoef))

    file_perf = open('results/' + method + '-performances.txt', 'w')
    file_perf.write("Jaccard similarity score: " + str(jaccard_index)
                    + "\nConfusion matrix: " + str({"Real Pos": {"tp": tp, "fn": fn}, "Real Neg": {"fp": fp, "tn": tn}})
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    + "\nNPV: " + str(NPV)
                    + "\nError Rate: " + str(error_rate)
                    + "\nThe Matthews correlation coefficient: " + str(corrcoef)
                    + "\nF1 score: " + str(f1score))
    file_perf.close()

    rep = classification_report(Y_Test, y_pred)
    print(rep)
    file_perf = open('results/' + method + '-classification_report.txt', 'w')
    file_perf.write(rep)
    file_perf.close()

    return precision