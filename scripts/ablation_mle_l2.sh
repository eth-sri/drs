python train.py mnist_2_6 default 0.25 --output_path ./models/ablation_mle/mnist_2_6_l2_0.25 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_0.25/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l2_0.25 certified_radii_default gaussian 0.25 --device cuda; \
python train.py mnist_2_6 default 0.25 --output_path ./models/ablation_mle/mnist_2_6_l2_0.25 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 1; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_0.25/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l2_0.25 certified_radii_sampling gaussian 0.25 --device cuda; \
python train.py mnist_2_6 gaussian 0.25 --output_path ./models/ablation_mle/mnist_2_6_l2_0.25 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_0.25/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l2_0.25 certified_radii_mle gaussian 0.25 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l2_0.25/table 2 ./experiments/ablation_mle/mnist_2_6_l2_0.25/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l2_0.25/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l2_0.25/certified_radii_mle; \
python train.py mnist_2_6 default 1.00 --output_path ./models/ablation_mle/mnist_2_6_l2_1.00 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_1.00/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l2_1.00 certified_radii_default gaussian 1.00 --device cuda; \
python train.py mnist_2_6 default 1.00 --output_path ./models/ablation_mle/mnist_2_6_l2_1.00 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 1; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_1.00/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l2_1.00 certified_radii_sampling gaussian 1.00 --device cuda; \
python train.py mnist_2_6 gaussian 1.00 --output_path ./models/ablation_mle/mnist_2_6_l2_1.00 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_1.00/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l2_1.00 certified_radii_mle gaussian 1.00 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l2_1.00/table 2 ./experiments/ablation_mle/mnist_2_6_l2_1.00/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l2_1.00/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l2_1.00/certified_radii_mle; \
python train.py mnist_2_6 default 4.00 --output_path ./models/ablation_mle/mnist_2_6_l2_4.00 --model_name stump_ensemble_default.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_4.00/stump_ensemble_default.pkl ./experiments/ablation_mle/mnist_2_6_l2_4.00 certified_radii_default gaussian 4.00 --device cuda; \
python train.py mnist_2_6 default 4.00 --output_path ./models/ablation_mle/mnist_2_6_l2_4.00 --model_name stump_ensemble_sampling.pkl --use_noisy_samples 1; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_4.00/stump_ensemble_sampling.pkl ./experiments/ablation_mle/mnist_2_6_l2_4.00 certified_radii_sampling gaussian 4.00 --device cuda; \
python train.py mnist_2_6 gaussian 4.00 --output_path ./models/ablation_mle/mnist_2_6_l2_4.00 --model_name stump_ensemble_mle.pkl; \
python certify.py mnist_2_6 ./models/ablation_mle/mnist_2_6_l2_4.00/stump_ensemble_mle.pkl ./experiments/ablation_mle/mnist_2_6_l2_4.00 certified_radii_mle gaussian 4.00 --device cuda; \
python analyze.py ./experiments/ablation_mle/mnist_2_6_l2_4.00/table 2 ./experiments/ablation_mle/mnist_2_6_l2_4.00/certified_radii_default ./experiments/ablation_mle/mnist_2_6_l2_4.00/certified_radii_sampling ./experiments/ablation_mle/mnist_2_6_l2_4.00/certified_radii_mle;
