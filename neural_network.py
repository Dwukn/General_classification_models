from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
def baseline_model(input_shape):
    model = Sequential([
        Conv2D(16, (4, 4), activation='relu', input_shape=input_shape),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    return model

def residual_model(input_shape):
    model = Sequential([
        Conv2D(16, (2, 2), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Conv2D(92, (3, 3), activation='relu'),
        Conv2D(92, (3, 3), activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    return model
