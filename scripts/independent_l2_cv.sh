python train.py breast_cancer gaussian 4.0 --output_path ./models/independent_cv/breast_cancer_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py breast_cancer ./models/independent_cv/breast_cancer_l2_0/stump_ensemble.pkl ./experiments/independent_cv//breast_cancer_l2_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/independent_cv/breast_cancer_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py breast_cancer ./models/independent_cv/breast_cancer_l2_1/stump_ensemble.pkl ./experiments/independent_cv//breast_cancer_l2_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/independent_cv/breast_cancer_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py breast_cancer ./models/independent_cv/breast_cancer_l2_2/stump_ensemble.pkl ./experiments/independent_cv//breast_cancer_l2_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/independent_cv/breast_cancer_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py breast_cancer ./models/independent_cv/breast_cancer_l2_3/stump_ensemble.pkl ./experiments/independent_cv//breast_cancer_l2_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/independent_cv/breast_cancer_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py breast_cancer ./models/independent_cv/breast_cancer_l2_4/stump_ensemble.pkl ./experiments/independent_cv//breast_cancer_l2_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/breast_cancer_l2/table 5 ./experiments/independent_cv/breast_cancer_l2_0/certified_radii ./experiments/independent_cv/breast_cancer_l2_1/certified_radii ./experiments/independent_cv/breast_cancer_l2_2/certified_radii ./experiments/independent_cv/breast_cancer_l2_3/certified_radii ./experiments/independent_cv/breast_cancer_l2_4/certified_radii; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent_cv/diabetes_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py diabetes ./models/independent_cv/diabetes_l2_0/stump_ensemble.pkl ./experiments/independent_cv//diabetes_l2_0 certified_radii gaussian 0.25 --device cuda --current_fold 0 --n_splits 5; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent_cv/diabetes_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py diabetes ./models/independent_cv/diabetes_l2_1/stump_ensemble.pkl ./experiments/independent_cv//diabetes_l2_1 certified_radii gaussian 0.25 --device cuda --current_fold 1 --n_splits 5; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent_cv/diabetes_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py diabetes ./models/independent_cv/diabetes_l2_2/stump_ensemble.pkl ./experiments/independent_cv//diabetes_l2_2 certified_radii gaussian 0.25 --device cuda --current_fold 2 --n_splits 5; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent_cv/diabetes_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py diabetes ./models/independent_cv/diabetes_l2_3/stump_ensemble.pkl ./experiments/independent_cv//diabetes_l2_3 certified_radii gaussian 0.25 --device cuda --current_fold 3 --n_splits 5; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent_cv/diabetes_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py diabetes ./models/independent_cv/diabetes_l2_4/stump_ensemble.pkl ./experiments/independent_cv//diabetes_l2_4 certified_radii gaussian 0.25 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/diabetes_l2/table 5 ./experiments/independent_cv/diabetes_l2_0/certified_radii ./experiments/independent_cv/diabetes_l2_1/certified_radii ./experiments/independent_cv/diabetes_l2_2/certified_radii ./experiments/independent_cv/diabetes_l2_3/certified_radii ./experiments/independent_cv/diabetes_l2_4/certified_radii; \
python train.py spambase gaussian 0.25 --output_path ./models/independent_cv/spambase_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py spambase ./models/independent_cv/spambase_l2_0/stump_ensemble.pkl ./experiments/independent_cv//spambase_l2_0 certified_radii gaussian 0.25 --device cuda --current_fold 0 --n_splits 5; \
python train.py spambase gaussian 0.25 --output_path ./models/independent_cv/spambase_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py spambase ./models/independent_cv/spambase_l2_1/stump_ensemble.pkl ./experiments/independent_cv//spambase_l2_1 certified_radii gaussian 0.25 --device cuda --current_fold 1 --n_splits 5; \
python train.py spambase gaussian 0.25 --output_path ./models/independent_cv/spambase_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py spambase ./models/independent_cv/spambase_l2_2/stump_ensemble.pkl ./experiments/independent_cv//spambase_l2_2 certified_radii gaussian 0.25 --device cuda --current_fold 2 --n_splits 5; \
python train.py spambase gaussian 0.25 --output_path ./models/independent_cv/spambase_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py spambase ./models/independent_cv/spambase_l2_3/stump_ensemble.pkl ./experiments/independent_cv//spambase_l2_3 certified_radii gaussian 0.25 --device cuda --current_fold 3 --n_splits 5; \
python train.py spambase gaussian 0.25 --output_path ./models/independent_cv/spambase_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py spambase ./models/independent_cv/spambase_l2_4/stump_ensemble.pkl ./experiments/independent_cv//spambase_l2_4 certified_radii gaussian 0.25 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/spambase_l2/table 5 ./experiments/independent_cv/spambase_l2_0/certified_radii ./experiments/independent_cv/spambase_l2_1/certified_radii ./experiments/independent_cv/spambase_l2_2/certified_radii ./experiments/independent_cv/spambase_l2_3/certified_radii ./experiments/independent_cv/spambase_l2_4/certified_radii; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent_cv/fmnist_sandal_sneaker_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py fmnist_sandal_sneaker ./models/independent_cv/fmnist_sandal_sneaker_l2_0/stump_ensemble.pkl ./experiments/independent_cv//fmnist_sandal_sneaker_l2_0 certified_radii gaussian 0.25 --device cuda --current_fold 0 --n_splits 5; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent_cv/fmnist_sandal_sneaker_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py fmnist_sandal_sneaker ./models/independent_cv/fmnist_sandal_sneaker_l2_1/stump_ensemble.pkl ./experiments/independent_cv//fmnist_sandal_sneaker_l2_1 certified_radii gaussian 0.25 --device cuda --current_fold 1 --n_splits 5; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent_cv/fmnist_sandal_sneaker_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py fmnist_sandal_sneaker ./models/independent_cv/fmnist_sandal_sneaker_l2_2/stump_ensemble.pkl ./experiments/independent_cv//fmnist_sandal_sneaker_l2_2 certified_radii gaussian 0.25 --device cuda --current_fold 2 --n_splits 5; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent_cv/fmnist_sandal_sneaker_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py fmnist_sandal_sneaker ./models/independent_cv/fmnist_sandal_sneaker_l2_3/stump_ensemble.pkl ./experiments/independent_cv//fmnist_sandal_sneaker_l2_3 certified_radii gaussian 0.25 --device cuda --current_fold 3 --n_splits 5; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent_cv/fmnist_sandal_sneaker_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py fmnist_sandal_sneaker ./models/independent_cv/fmnist_sandal_sneaker_l2_4/stump_ensemble.pkl ./experiments/independent_cv//fmnist_sandal_sneaker_l2_4 certified_radii gaussian 0.25 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/fmnist_sandal_sneaker_l2/table 5 ./experiments/independent_cv/fmnist_sandal_sneaker_l2_0/certified_radii ./experiments/independent_cv/fmnist_sandal_sneaker_l2_1/certified_radii ./experiments/independent_cv/fmnist_sandal_sneaker_l2_2/certified_radii ./experiments/independent_cv/fmnist_sandal_sneaker_l2_3/certified_radii ./experiments/independent_cv/fmnist_sandal_sneaker_l2_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent_cv/mnist_1_5_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py mnist_1_5 ./models/independent_cv/mnist_1_5_l2_0/stump_ensemble.pkl ./experiments/independent_cv//mnist_1_5_l2_0 certified_radii gaussian 0.25 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent_cv/mnist_1_5_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py mnist_1_5 ./models/independent_cv/mnist_1_5_l2_1/stump_ensemble.pkl ./experiments/independent_cv//mnist_1_5_l2_1 certified_radii gaussian 0.25 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent_cv/mnist_1_5_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py mnist_1_5 ./models/independent_cv/mnist_1_5_l2_2/stump_ensemble.pkl ./experiments/independent_cv//mnist_1_5_l2_2 certified_radii gaussian 0.25 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent_cv/mnist_1_5_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py mnist_1_5 ./models/independent_cv/mnist_1_5_l2_3/stump_ensemble.pkl ./experiments/independent_cv//mnist_1_5_l2_3 certified_radii gaussian 0.25 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent_cv/mnist_1_5_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py mnist_1_5 ./models/independent_cv/mnist_1_5_l2_4/stump_ensemble.pkl ./experiments/independent_cv//mnist_1_5_l2_4 certified_radii gaussian 0.25 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/mnist_1_5_l2/table 5 ./experiments/independent_cv/mnist_1_5_l2_0/certified_radii ./experiments/independent_cv/mnist_1_5_l2_1/certified_radii ./experiments/independent_cv/mnist_1_5_l2_2/certified_radii ./experiments/independent_cv/mnist_1_5_l2_3/certified_radii ./experiments/independent_cv/mnist_1_5_l2_4/certified_radii; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent_cv/mnist_2_6_l2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5; \
python certify.py mnist_2_6 ./models/independent_cv/mnist_2_6_l2_0/stump_ensemble.pkl ./experiments/independent_cv//mnist_2_6_l2_0 certified_radii gaussian 0.25 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent_cv/mnist_2_6_l2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5; \
python certify.py mnist_2_6 ./models/independent_cv/mnist_2_6_l2_1/stump_ensemble.pkl ./experiments/independent_cv//mnist_2_6_l2_1 certified_radii gaussian 0.25 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent_cv/mnist_2_6_l2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5; \
python certify.py mnist_2_6 ./models/independent_cv/mnist_2_6_l2_2/stump_ensemble.pkl ./experiments/independent_cv//mnist_2_6_l2_2 certified_radii gaussian 0.25 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent_cv/mnist_2_6_l2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5; \
python certify.py mnist_2_6 ./models/independent_cv/mnist_2_6_l2_3/stump_ensemble.pkl ./experiments/independent_cv//mnist_2_6_l2_3 certified_radii gaussian 0.25 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent_cv/mnist_2_6_l2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5; \
python certify.py mnist_2_6 ./models/independent_cv/mnist_2_6_l2_4/stump_ensemble.pkl ./experiments/independent_cv//mnist_2_6_l2_4 certified_radii gaussian 0.25 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/independent_cv/mnist_2_6_l2/table 5 ./experiments/independent_cv/mnist_2_6_l2_0/certified_radii ./experiments/independent_cv/mnist_2_6_l2_1/certified_radii ./experiments/independent_cv/mnist_2_6_l2_2/certified_radii ./experiments/independent_cv/mnist_2_6_l2_3/certified_radii ./experiments/independent_cv/mnist_2_6_l2_4/certified_radii; \
