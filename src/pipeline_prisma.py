from __future__ import annotations

try:
    from .prisma_core import BASE_DIR, MODEL_PATH, train_and_save_model
except ImportError:
    from prisma_core import BASE_DIR, MODEL_PATH, train_and_save_model


def main() -> None:
    print("Treinando o modelo PRISMA para gestao de risco FIDC...")
    bundle = train_and_save_model(BASE_DIR)
    print(f"Modelo salvo em: {MODEL_PATH}")
    print("Metricas de validacao:")
    for key, value in bundle["metrics"].items():
        print(f"  - {key}: {value:.4f}")


if __name__ == "__main__":
    main()
