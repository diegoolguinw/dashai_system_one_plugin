#!/usr/bin/env python3
"""End-to-end verification script testing DashAI pipeline with routing_sample.csv."""

import csv
import pathlib
import sys

import numpy as np

# Ensure src/ is on python path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.models.base_dashai_system_one import (
    DashAISystemOneClassifier,
)


class PipelineTestClassifier(DashAISystemOneClassifier):
    MODEL_NAME = "convaiinnovations/laya-multilingual"
    BACKEND_TYPE = "fake"

    def _create_backend(self):
        return FakeBackend(simulate_probabilities=True)


def main():
    print("=" * 60)
    print("Prueba de Integración: Pipeline de DashAI con routing_sample.csv")
    print("=" * 60)

    # 1. Cargar dataset sintético de routing
    csv_path = pathlib.Path(__file__).resolve().parent / "routing_sample.csv"
    texts = []
    labels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["category"])

    print(f"\n1. Cargadas {len(texts)} muestras de '{csv_path.name}'.")
    categories = sorted(set(labels))
    print(f"   Categorías ({len(categories)}): {categories}")

    # 2. Inicializar modelo
    print("\n2. Inicializando clasificador System One...")
    model = PipelineTestClassifier(batch_size=4)

    # 3. Entrenar / inicializar etiquetas zero-shot (train)
    print("3. Ejecutando train() zero-shot con mapeo de clases...")
    model.train(texts, labels)
    print(f"   Clases registradas en el modelo: {model.class_names}")

    # 4. Inferencia (predict)
    print("\n4. Ejecutando predict() sobre las muestras de prueba...")
    test_queries = [
        "No puedo ingresar a mi plataforma y necesito ayuda urgente",
        "Tengo un error en el cobro de mi tarjeta este mes",
        "El botón de exportar reportes arroja un pantallazo de error",
    ]
    prob_matrix = model.predict(test_queries)

    print(f"   Matriz de probabilidades de salida: forma {prob_matrix.shape}")
    assert prob_matrix.shape == (len(test_queries), len(categories))
    assert np.allclose(prob_matrix.sum(axis=1), np.ones(len(test_queries)), atol=1e-2)

    # 5. Simular process_predictions de DashAI ClassificationTask
    best_class_indices = np.argmax(prob_matrix, axis=1)
    predicted_labels = [model.class_names[idx] for idx in best_class_indices]

    print("\n5. Resultados de decisión (simulando ClassificationTask.process_predictions):")
    for q, pred_label, p_row in zip(test_queries, predicted_labels, prob_matrix):
        top_prob = p_row.max()
        print(f"   • Consulta: \"{q}\"")
        print(f"     -> Predicción: [{pred_label}] (Confianza: {top_prob:.2%})\n")

    print("=" * 60)
    print("¡Integración DashAI verificada exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
