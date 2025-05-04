from sklearn.metrics import f1_score, accuracy_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

# Metrics
def evaluate_metrics(y_true, y_pred):
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'hamming': hamming_loss(y_true, y_pred),
        'cohen': cohen_kappa_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)),
        'mcc': matthews_corrcoef(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    }
