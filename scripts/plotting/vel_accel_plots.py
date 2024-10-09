def load_all_npy_files(directory, max_seq_len=100):
    X = []
    y = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                label = root.split("/")[-1]  # Assuming the label is in the folder name
                file_path = os.path.join(root, file)
                data = np.load(file_path)

                if data.size == 0:
                    print(f"Empty data in file: {file_path}")
                    continue

                # Iterate over each data sample and combine the features
                for i in range(min(len(data), max_seq_len)):
                    try:
                        landmarks, velocity, acceleration = data[i]
                        combined_features = np.concatenate([landmarks, velocity, acceleration], axis=-1)
                        X.append(combined_features)
                    except ValueError:
                        print(f"Data format issue in file: {file_path}")
                        continue

                # Append labels (adjust this based on your labeling logic)
                if label == "2_4_60bpm":
                    y.append(0)
                elif label == "3_4_60bpm":
                    y.append(1)
                elif label == "4_4_60bpm":
                    y.append(2)

    X = np.array(X)
    y = np.array(y)
    
    # Check if X and y have the expected shapes
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y
