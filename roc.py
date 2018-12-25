import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_test = [1,1,0,0,1,1,0,0,1,0]
preds = [0.61,0.03,0.68,0.31,0.45,0.09,0.38,0.05,0.01,0.04]

fpr, tpr, threshold = sk.metrics.roc_curve(y_test,preds)
roc_auc = sk.metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()