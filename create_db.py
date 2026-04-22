from app import init_db, train_model, MODEL_PATH_S1, MODEL_PATH_S2, THRESHOLD_S1, THRESHOLD_S2

if __name__ == "__main__":
    init_db()
    train_model(MODEL_PATH_S1, THRESHOLD_S1)
    train_model(MODEL_PATH_S2, THRESHOLD_S2)
    print("Database initialized and models trained successfully.")