"""Image loading, patient/label derivation, report join, split and results.

Extracted from the original monolith notebook: id_paciente was duplicated
(EDA cell and split cell), the CSV join used raw Windows backslash paths, and
the patient-grouped split lived inline. No behavioural change versus the
original inline code -- same values, now importable and shared.
"""
import os
import re
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize

CLASS_FOLDERS = [
    (0, "cataract"),
    (1, "diabetic_retinopathy"),
    (2, "normal"),
]
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Normal"]


def id_paciente(nome):
    """Patient id from filename: '1115_left.jpg' / '1115_right.jpg' -> '1115'."""
    base = nome.rsplit(".", 1)[0]
    base = re.sub(r'_(left|right)$', '', base, flags=re.I)
    return base


def load_images_from_folder(folder, label):
    """Load every image in `folder`, padded to 224x224, tagged with `label`."""
    images = []
    labels = []
    nomes = []
  
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")

        
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0))

        images.append(np.array(img))
        labels.append(label)
        nomes.append(filename)
    return images, labels, nomes


def get_class_folders(data_dir):
    """label -> (class_name, folder_path), the three ocular/ subfolders."""
    data_dir = Path(data_dir)
    return {
        label: (name, str(data_dir / "ocular" / name))
        for label, name in CLASS_FOLDERS
    }


def load_dataset(data_dir):
    """Load all three classes, label derived from the subfolder.

    Same concatenation order the monolith used (cataract, diabetic_retinopathy,
    normal), which is what makes groups/X_original alignment structural.
    Returns X_original, y, nomes_ficheiros, labels_ficheiros, folders.
    """
    folders = get_class_folders(data_dir)

    all_images, all_labels, all_files = [], [], []
    counts_by_label = []
    for label, (_name, folder) in folders.items():
        images, labels, files = load_images_from_folder(folder, label)
        all_images += images
        all_labels += labels
        all_files += files
        counts_by_label.append((label, len(files)))

    X_original = np.array(all_images)
    y = np.array(all_labels)
    nomes_ficheiros = np.array(all_files)
    labels_ficheiros = np.concatenate(
        [np.full(count, label) for label, count in counts_by_label]
    )

    return X_original, y, nomes_ficheiros, labels_ficheiros, folders


def build_groups(nomes_ficheiros):
    """Patient id per file, built from the SAME filename sequence that
    produced X_original -- keeps groups aligned by construction rather than
    by re-listing folders (which could reorder independently)."""
    return np.array([id_paciente(f) for f in nomes_ficheiros])


def split_patient_grouped(X, y, groups, seed, n_splits_test=5, n_splits_val=4):
    """Two-stage patient-grouped, class-stratified split (~60/20/20).

    A patient's images never end up split across train/val/test. All
    returned indices are ABSOLUTE positions into X/y/groups, so callers in
    any notebook can slice X_original directly (X_original[train_idx],
    etc.) without an intermediate trainval-relative step.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(X, y, groups))

    groups_trainval = groups[trainval_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_rel_idx, val_rel_idx = next(
        sgkf2.split(X[trainval_idx], y[trainval_idx], groups_trainval)
    )

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    return {
        "trainval_idx": trainval_idx,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def save_split_indices(path, split):
    np.savez(
        path,
        trainval_idx=split["trainval_idx"],
        test_idx=split["test_idx"],
        train_idx=split["train_idx"],
        val_idx=split["val_idx"],
    )

def guardar_modelo(nome, models_dir, modelo):
    destino = Path(models_dir) / f"{slug(nome)}.keras"
    destino.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(destino)
    return destino

def load_split_indices(path):
    npz = np.load(path)
    return {k: npz[k] for k in ("trainval_idx", "test_idx", "train_idx", "val_idx")}


def slug(nome):
    """Model name -> stable filename slug, e.g. for artifacts/preds/{slug}.npy."""
    s = nome.lower()
    for ch in " ()-/":
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def load_reports(csv_path):
    """Load ocular_reports.csv, normalising Windows-style paths to forward
    slashes -- the CSV was produced on Windows and image<->report joins
    would silently fail on the Colab Linux runtime otherwise."""
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    df["filename"] = df["image_path"].apply(lambda p: p.split("/")[-1])
    return df


def join_images_reports(nomes_ficheiros, df_reports):
    """Align reports to the image loading order.

    Asserts the join is complete in both directions: no image left without a
    report, and no report left over that doesn't match an image.
    """
    df_aligned = pd.DataFrame({"filename": list(nomes_ficheiros)})
    df_aligned = df_aligned.merge(df_reports, on="filename", how="left")

    assert len(df_aligned) == len(nomes_ficheiros), \
        f"aligned reports ({len(df_aligned)}) != images ({len(nomes_ficheiros)})"

    missing_reports = df_aligned["report"].isna().sum()
    assert missing_reports == 0, \
        f"{missing_reports} images have no matching report after the join"

    image_filenames = set(nomes_ficheiros)
    report_filenames = set(df_reports["filename"])
    orphan_reports = report_filenames - image_filenames
    assert not orphan_reports, \
        f"{len(orphan_reports)} report rows match no image: {sorted(orphan_reports)[:5]}"

    return df_aligned


def compute_metrics_row(nome, proba_test, y_test, class_names):
    """Test-set metrics for one model -- one row of the comparative table."""
    y_pred = np.argmax(proba_test, axis=1)
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    model_auc = roc_auc_score(y_bin, proba_test, average="macro", multi_class="ovr")
    recalls = recall_score(y_test, y_pred, average=None)

    row = {"Modelo": nome, "Accuracy": acc, "F1-macro": f1, "AUC-ROC macro": model_auc}
    for cls_name, rec in zip(class_names, recalls):
        row[f"Recall {cls_name}"] = rec
    return row


def append_results_row(row, results_path):
    """Read-modify-write artifacts/results.csv: upsert this model's row.

    02/03/04 run as separate Colab sessions, so the comparative table can
    only be assembled incrementally on disk -- there is no in-memory
    registry that survives across notebooks.
    """
    results_path = Path(results_path)
    if results_path.exists():
        df = pd.read_csv(results_path)
        df = df[df["Modelo"] != row["Modelo"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_path, index=False)
    return df

def adicionar_modelo(nome, modelo, history, X_teste, X_treino=None, y_treino=None):
    global modelos_treinados
    X_tr = X_treino if X_treino is not None else X_train
    y_tr = y_treino if y_treino is not None else y_train

    # Congelar as previsões AGORA, enquanto o modelo está no estado que este
    # nome descreve. Depois disto, mutações do objeto já não afectam a tabela.
    proba_teste = modelo.predict(X_teste, verbose=0)
    proba_treino = modelo.predict(X_tr, verbose=0)

    # Guarda contra aliasing: dois modelos distintos não produzem previsões
    # bit-a-bit idênticas em 507x3 floats. Se isto disparar, duas entradas
    # estão a partilhar o mesmo objeto.
    for outro_nome, _, _, _, _, _, outra_proba, _ in modelos_treinados:
        if outra_proba.shape == proba_teste.shape and np.array_equal(outra_proba, proba_teste):
            raise RuntimeError(
                f"ALIASING: '{nome}' produz previsões idênticas a '{outro_nome}'. "
                "Provavelmente é o mesmo objeto de modelo registado duas vezes "
                "(ex.: fine-tuning aplicado in place ao modelo de feature extraction). "
                "Cada condição precisa da sua própria instância."
            )

    caminho = PREDS_DIR / f"{data.slug(nome)}_proba.npy"
    np.save(caminho, proba_teste)
    np.save(PREDS_DIR / f"{data.slug(nome)}_train_proba.npy", proba_treino)

    with open(HISTORY_DIR / f"{data.slug(nome)}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    guardar_modelo(nome, MODELS_DIR, modelo)

    # remover entrada anterior com o mesmo nome (evita duplicados ao re-correr)
    modelos_treinados = [m for m in modelos_treinados if m[0] != nome]
    modelos_treinados.append((
        nome, modelo, history, X_teste, X_tr, y_tr, proba_teste, proba_treino,
    ))
    print(f"Adicionado: {nome}  (total: {len(modelos_treinados)})  -> {caminho.name}")


def fine_tuning_grid_search(model, base_model, nome_modelo, phase1_weights):
    """
    Grid search sobre learning_rate e n_layers_unfrozen para fine-tuning.
    Para cada combinação, parte sempre dos pesos da fase 1 (feature extraction)
    para garantir comparação justa. Seleciona com base no F1-macro na validação.
    """
    learning_rates = [1e-4, 1e-5, 1e-6]
    n_layers_list  = [10, 30, 50]

    resultados = []

    for lr, n_layers in product(learning_rates, n_layers_list):
        tf.keras.utils.set_random_seed(SEED)
        print(f"=== {nome_modelo} | LR={lr}  layers={n_layers} ===")

        # restaurar pesos da fase 1 para partir do mesmo ponto em cada iteração
        model.set_weights(phase1_weights)

        # descongelar apenas as últimas n_layers da base convolucional
        base_model.trainable = True
        for layer in base_model.layers[:-n_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        model.fit(
            X_train_tl, y_train_tl,
            validation_data=(X_val_tl, y_val_tl),
            epochs=10,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[es],
            verbose=0
        )

        # avaliar na validação — critério de selecção: F1-macro
        y_pred  = np.argmax(model.predict(X_val_tl, verbose=0), axis=1)
        f1m     = f1_score(y_val_tl, y_pred, average="macro")
        rec_cat = recall_score(y_val_tl, y_pred, labels=[0], average="macro")

        resultados.append({
            "lr": lr, "n_layers": n_layers,
            "f1_macro": float(f1m), "recall_catarata": float(rec_cat)
        })
        print(f"   F1-macro={f1m:.4f}  recall_catarata={rec_cat:.4f}")

        # congelar novamente antes da próxima iteração
        base_model.trainable = False

    # ordenar por F1-macro e mostrar ranking completo
    resultados.sort(key=lambda r: r["f1_macro"], reverse=True)
    print(f"\n===== RANKING — {nome_modelo} =====")
    for r in resultados:
        print(f"F1={r['f1_macro']:.4f}  rec_cat={r['recall_catarata']:.4f}  <-  LR={r['lr']}  layers={r['n_layers']}")

    return resultados[0]  # devolver os melhores hiperparâmetros



def aplicar_melhor_fine_tuning(model, base_model, best_params, nome_modelo, phase1_weights):
    """
    Treina o modelo final com os melhores hiperparâmetros encontrados no grid search.
    Restaura os pesos da fase 1, descongela as últimas n_layers e treina com LR reduzido.
    """
    # restaurar pesos da feature extraction para partir do mesmo ponto
    model.set_weights(phase1_weights)

    # descongelar apenas as últimas n_layers da base
    base_model.trainable = True
    for layer in base_model.layers[:-best_params["n_layers"]]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    nome_final = f"{nome_modelo} (fine-tuned)"
    checkpoint = ModelCheckpoint(
        filepath=str(MODELS_DIR / f"{data.slug(nome_final)}.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    history = model.fit(
        X_train_tl, y_train_tl,
        validation_data=(X_val_tl, y_val_tl),
        epochs=20,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[es, checkpoint],
        verbose=2
    )

    adicionar_modelo(nome_final, model, history, X_test_tl, X_train_tl, y_train_tl)
    return history