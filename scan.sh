mkdir -p logs
python3 -m rebar.scripts.train_mnist \
    --epochs 100 \
    --log logs/rebar.npz
python3 -m rebar.scripts.train_mnist \
    --epochs 100 \
    --init_eta 0.000001 \
    --cv_lr 0 \
    --log logs/reinforce.npz
python3 -m rebar.scripts.train_mnist \
    --epochs 100 \
    --init_eta 0.000001 \
    --cv_lr 0 \
    --sample_baseline \
    --log logs/baseline.npz
