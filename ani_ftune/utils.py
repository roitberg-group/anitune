from enum import Enum


class TrainKind(Enum):
    TRAIN = "train"
    FTUNE = "ftune"


DATA_ELEMENTS = {
    "ANI1x": ("H", "C", "N", "O"),
    "ANIExCorr": ("H", "C", "N", "O", "S"),
    "ANI1ccx": ("H", "C", "N", "O"),
    "TestData": ("H", "C", "N", "O"),
    "ANI2x": ("H", "C", "N", "O", "F", "S", "Cl"),
}
