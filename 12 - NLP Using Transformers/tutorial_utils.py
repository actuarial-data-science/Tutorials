import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import torch
import plotly.express as px


# apply the NLP model to tokenized inputs, extract last hidden states, and return one tensor per input sample
def extract_sequence_encoding(batch, model, normalize=False):
    input_ids = torch.tensor(batch["input_ids"]).to(model.device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)
    with torch.no_grad():
        # apply transformer model and extract last hidden state
        model_output = model(input_ids, attention_mask)
        last_hidden_state = model_output.last_hidden_state

        # extract the tensor corresponding to the CLS token, i.e. the first element in the encoded sequence
        v = last_hidden_state[:,0,:].cpu().numpy()
        if normalize:
            v = v / np.linalg.norm(v, axis=-1)[:, None]
        batch["cls_hidden_state"] = v

        # mean pooling: take average over input sequence, but mask sequence elements corresponding to the PAD token
        last_hidden_state = last_hidden_state.cpu().numpy()
        lhs_shape = last_hidden_state.shape
        boolean_mask = ~np.array(batch["attention_mask"]).astype(bool)
        boolean_mask = np.repeat(boolean_mask, lhs_shape[-1], axis=-1)
        boolean_mask = boolean_mask.reshape(lhs_shape)
        masked_mean = np.ma.array(last_hidden_state, mask=boolean_mask).mean(axis=1)
        v = masked_mean.data
        if normalize:
            v = v / np.linalg.norm(v, axis=-1)[:, None]
        batch["mean_hidden_state"] = v
    return batch


# get features and label corresponding to the train and test split of a dataset
def get_xy(data, features, label):
    x_train = np.array(data["train"][features])
    y_train = np.array(data["train"][label])
    x_test  = np.array(data["test"][features])
    y_test  = np.array(data["test"][label])
    return x_train, y_train, x_test, y_test


# fit a logistic regression classifier
def logistic_regression_classifier(x, y, c=1, class_weight=None):
    clf = LogisticRegression(n_jobs=-1, penalty="l2", C=c, solver="newton-cg", max_iter=500, class_weight=class_weight)
    clf.fit(x, y)
    return clf


# fit a dummy classifier, which always predicts the most frequent class
# and returns probabilities according to empirical frequency
def dummy_classifier(x, y):
    return DummyClassifier(strategy="prior").fit(x, y)


# calculate Brier loss for multinomial case
def brier_multi(y_true, p_pred):
    y_true_ = np.zeros_like(p_pred)
    y_true_[range(p_pred.shape[0]), y_true] = 1
    return np.mean(np.sum((p_pred - y_true_)**2, axis=1))


# calculate and display performance metrics
def evaluate_classifier(y_true, y_pred, p_pred, target_names, title, file_name='new_plot'):
    if y_pred is None:
        y_pred = np.argmax(p_pred, -1)  # calculate vector of predicted classes from probability matrix

    # calculate scores
    score_accuracy = accuracy_score(y_true, y_pred)
    score_log = log_loss(y_true, p_pred) if p_pred is not None else np.nan
    score_brier = brier_multi(y_true, p_pred) if p_pred is not None else np.nan
    cm = confusion_matrix(y_true, y_pred, normalize=None)

    # print scores and classification report
    print(f"{title}")
    print(f"accuracy score = {score_accuracy:.1%},  log loss = {score_log:.3f},  Brier loss = {score_brier:.3f}")
    print("classification report\n", classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # display confusion matrix
    font_size = 14 if len(target_names) >=6 else 20
    x = [" <b>"+i+"</b> " for i in target_names]
    y = [" <b>"+i+"</b> " for i in target_names]
    fig = px.imshow(cm, text_auto=True, x=x, y=y, width=600, title=title)
    fig.update_layout(coloraxis={"colorscale": "blues", "showscale": False}, font={"size": font_size})
    fig.update_xaxes(title_text="<b>predicted class</b>")
    fig.update_yaxes(title_text="<b>actual class</b>")
    fig.show(config={"toImageButtonOptions": {"format": "svg", "filename": file_name}})

    # export image
    if file_name is not None:
        if not os.path.exists("./results"):
            os.mkdir("./results")
        fig.write_image("./results/" + file_name + ".svg")

    return score_accuracy, score_log, score_brier, cm, fig
