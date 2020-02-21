import h5py
base_path = "./MFN/data/"
# base_path = "./"


def load_saved_data(base_path):
    h5f = h5py.File(base_path+'X_train.h5','r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(base_path+'y_train.h5','r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(base_path+'X_valid.h5','r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(base_path+'y_valid.h5','r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(base_path+'X_test.h5','r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(base_path+'y_test.h5','r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data(base_path)
