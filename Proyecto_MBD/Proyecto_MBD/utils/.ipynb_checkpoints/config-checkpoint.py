DATA_PATH = "Proyecto_MBD/data/raw/train_test_network.csv"
PROCESSED_PATH = "Proyecto_MBD/data/processed/"
ARTIFACTS_PATH = "Proyecto_MBD/data/artifacts/"

DROP_COLUMNS_BASE = ["src_ip", "dst_ip"]
DROP_COLUMNS_OPTIONAL = ["src_port", "dst_port"]

BINARY_TARGET = "label"
MULTICLASS_TARGET = "type"

TEST_SIZE = 0.2
RANDOM_STATE = 42