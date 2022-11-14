python train.py breast_cancer gaussian 4.00 --output_path ./models/independent/breast_cancer_l2 --model_name stump_ensemble.pkl; \
python certify.py breast_cancer ./models/independent/breast_cancer_l2/stump_ensemble.pkl ./experiments/independent/breast_cancer_l2 certified_radii gaussian 4.00 --device cuda; \
python analyze.py ./experiments/independent/breast_cancer_l2/table 0 ./experiments/independent/breast_cancer_l2/certified_radii; \
python train.py diabetes gaussian 0.25 --output_path ./models/independent/diabetes_l2 --model_name stump_ensemble.pkl; \
python certify.py diabetes ./models/independent/diabetes_l2/stump_ensemble.pkl ./experiments/independent/diabetes_l2 certified_radii gaussian 0.25 --device cuda; \
python analyze.py ./experiments/independent/diabetes_l2/table 0 ./experiments/independent/diabetes_l2/certified_radii; \
python train.py fmnist_sandal_sneaker gaussian 0.25 --output_path ./models/independent/fmnist_sandal_sneaker_l2 --model_name stump_ensemble.pkl; \
python certify.py fmnist_sandal_sneaker ./models/independent/fmnist_sandal_sneaker_l2/stump_ensemble.pkl ./experiments/independent/fmnist_sandal_sneaker_l2 certified_radii gaussian 0.25 --device cuda; \
python analyze.py ./experiments/independent/fmnist_sandal_sneaker_l2/table 0 ./experiments/independent/fmnist_sandal_sneaker_l2/certified_radii; \
python train.py mnist_1_5 gaussian 0.25 --output_path ./models/independent/mnist_1_5_l2 --model_name stump_ensemble.pkl; \
python certify.py mnist_1_5 ./models/independent/mnist_1_5_l2/stump_ensemble.pkl ./experiments/independent/mnist_1_5_l2 certified_radii gaussian 0.25 --device cuda; \
python analyze.py ./experiments/independent/mnist_1_5_l2/table 0 ./experiments/independent/mnist_1_5_l2/certified_radii; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/independent/mnist_2_6_l2 --model_name stump_ensemble.pkl; \
python certify.py mnist_2_6 ./models/independent/mnist_2_6_l2/stump_ensemble.pkl ./experiments/independent/mnist_2_6_l2 certified_radii gaussian 0.25 --device cuda; \
python analyze.py ./experiments/independent/mnist_2_6_l2/table 0 ./experiments/independent/mnist_2_6_l2/certified_radii;
