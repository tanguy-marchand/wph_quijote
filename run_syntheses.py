import os
from pipeline.syntheses_pipeline import SynthesesPipeline


kwargs_pipeline = {
    "J": 6,
    "L": 8,
    "delta_j": 5,
    "delta_l": 4,
    "delta_n": 2,
    "nb_chunks": 40,
    "nb_batches_of_syntheses": 1,
    "nb_iter": 100,
    "factr": 1e7,
    "number_synthesis": 2,
    'result_path': os.path.join('result', 'first_test', '1'),
    'filepaths': [os.path.join('data', 'quijote_fiducial_log_256', str(k), 'df_z=0.npy') for k in range(3)],
    "scaling_function_moments": [0, 1, 2, 3],
}

pipeline = SynthesesPipeline(**kwargs_pipeline)
pipeline.run()
