from src.utilities.dataloader import PASTIS


data_set = PASTIS(
    path_to_pastis='src/data/PASTIS/', 
    data_files='DATA_S2', 
    label_files='ANNOTATIONS',
    rgb_only=True,
    multi_temporal=True
    )

data_set.__iter__().__next__()