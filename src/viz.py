"""Plotting helpers: training curves, confusion matrices, ROC/PR, Grad-CAM.

Extracted verbatim from the monolith's "4. Resultados" / "5. Explicabilidade
Visual" sections. avaliar_modelo_multiclasse (renamed
plot_train_test_confusion) used to read `class_names` from the notebook's
global scope; it is now an explicit parameter like the other plot functions
already had -- a signature change required to make it importable, not a
behavioural one.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize


def plot_training_history(history):
    """Plota as curvas de Loss e Accuracy de treino e validação ao longo das
    épocas. Permite identificar overfitting (divergência entre treino e
    validação) e monitorizar a convergência do modelo."""
    train_loss     = history.history["loss"]
    val_loss       = history.history["val_loss"]
    train_accuracy = history.history["accuracy"]
    val_accuracy   = history.history["val_accuracy"]

    epochs = range(1, len(train_loss) + 1)

    print("Última train accuracy:", train_accuracy[-1])
    print("Última val accuracy:  ", val_accuracy[-1])
    print("Melhor train accuracy:", max(train_accuracy))
    print("Melhor val accuracy:  ", max(val_accuracy))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "bo-", label="Training Loss")
    plt.plot(epochs, val_loss,   "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_accuracy,   "ro-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_train_test_confusion(y_train_pred_probs, y_train, y_test_pred_probs, y_test,
                               class_names, nome_modelo="Modelo"):
    """Matriz de confusão e classification report lado a lado (treino vs
    teste) -- o diagnóstico de overfitting de cada modelo, executado no
    notebook produtor logo a seguir ao treino (usa proba_treino, que não é
    um artefacto committed)."""
    y_train_pred = np.argmax(y_train_pred_probs, axis=1)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
    disp_train.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{nome_modelo} - Treino")

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    disp_test.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title(f"{nome_modelo} - Teste")

    plt.tight_layout()
    plt.show()

    print(f"=== {nome_modelo} | TREINO ===")
    print(classification_report(y_train, y_train_pred))

    print(f"=== {nome_modelo} | TESTE ===")
    print(classification_report(y_test, y_test_pred))


def plot_test_confusion(proba_test, y_test, class_names, nome_modelo="Modelo"):
    """Matriz de confusão e classification report no conjunto de teste.

    Test-only variant for 05_clinical_evaluation, which only has committed
    test predictions available (no train-set proba is committed).
    """
    y_pred = np.argmax(proba_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{nome_modelo} - Teste")
    plt.tight_layout()
    plt.show()

    print(f"=== {nome_modelo} | TESTE ===")
    print(classification_report(y_test, y_pred, target_names=class_names))


def plot_roc_pr_curves(y_pred_probs, y_t, class_names, nome_modelo):
    """Plota lado a lado a Curva ROC e a Curva Precision-Recall para cada
    classe usando a estratégia One-vs-Rest (OvR)."""
    y_test_bin = label_binarize(y_t, classes=list(range(len(class_names))))
    colors = ["blue", "red", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (cls, cor) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=cor, label=f"{cls} (AUC={roc_auc:.2f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title(f"Curva ROC — {nome_modelo}")
    axes[0].set_xlabel("1 - Especificidade")
    axes[0].set_ylabel("Sensibilidade")
    axes[0].legend()

    for i, (cls, cor) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_pred_probs[:, i])
        axes[1].plot(rec, prec, color=cor, label=f"{cls} (AP={ap:.2f})")
    axes[1].set_title(f"Curva Precision-Recall — {nome_modelo}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_sensitivity_specificity_threshold(y_pred_probs, y_t, class_names, nome_modelo):
    """Para cada classe, plota Sensibilidade e Especificidade em função do
    threshold de decisão."""
    y_test_bin = label_binarize(y_t, classes=list(range(len(class_names))))
    colors = ["blue", "red", "green"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (cls, cor, ax) in enumerate(zip(class_names, colors, axes)):
        fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])

        sensibilidade = tpr
        especificidade = 1 - fpr

        ax.plot(thresholds, sensibilidade, color="blue", label="Sensibilidade")
        ax.plot(thresholds, especificidade, color="red", label="Especificidade")
        ax.set_title(f"{cls} — {nome_modelo}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.suptitle(f"Sensibilidade vs Especificidade por Threshold — {nome_modelo}")
    plt.tight_layout()
    plt.show()


def show_grad_cam(model, image, class_names, last_conv_layer_name, base_model=None):
    """Gera e visualiza o Grad-CAM para um modelo Sequential (CNNs) ou
    Functional (TL).
    - base_model=None  -> modo Sequential (CNN treinada de raiz)
    - base_model=<obj> -> modo Functional (EfficientNet, ResNet)
    """
    img_array = np.expand_dims(image.astype("float32"), axis=0)
    import tensorflow as tf

    if base_model is None:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_index = model.layers.index(last_conv_layer)
        conv_model = tf.keras.Sequential(model.layers[:last_conv_index + 1])
        head_layers = model.layers[last_conv_index + 1:]

        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            tape.watch(conv_outputs)
            x = conv_outputs
            for layer in head_layers:
                x = layer(x, training=False)
            predictions = x
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]
    else:
        lambda_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Lambda)]
        preprocess_layer = lambda_layers[0] if lambda_layers else None

        feat_extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
        base_idx = next(i for i, l in enumerate(model.layers) if l.name == base_model.name)
        head_layers = model.layers[base_idx + 1:]

        with tf.GradientTape() as tape:
            preprocessed = preprocess_layer(img_array) if preprocess_layer else img_array
            conv_outputs, base_out = feat_extractor(preprocessed)
            tape.watch(conv_outputs)
            x = base_out
            for layer in head_layers:
                x = layer(x, training=False)
            predictions = x
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (image.shape[0], image.shape[1])
    ).numpy().squeeze()

    preds = model.predict(img_array, verbose=0)
    final_pred_idx = np.argmax(preds[0])
    final_pred_prob = preds[0][final_pred_idx]

    plt.figure(figsize=(12, 4))
    img_disp = image.astype("uint8") if np.max(image) > 1 else image
    plt.subplot(1, 3, 1); plt.imshow(img_disp); plt.title("Imagem Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(heatmap, cmap="jet"); plt.title("Grad-CAM"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(img_disp)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.title(f"Overlay | {class_names[final_pred_idx]}: {final_pred_prob:.2f}"); plt.axis("off")
    plt.tight_layout()
    plt.show()
