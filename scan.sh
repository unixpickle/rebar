mkdir -p logs
epochs=150
for n_latent in 32 64 128 256; do
    python3 -m rebar.scripts.train_mnist \
        --epochs $epochs \
        --n_latent $n_latent \
        --log logs/rebar_${n_latent}.npz
    # python3 -m rebar.scripts.train_mnist \
    #     --epochs 100 \
    #     --n_latent $n_latent \
    #     --init_eta 0.000001 \
    #     --cv_lr 0 \
    #     --log logs/reinforce_${n_latent}.npz
    python3 -m rebar.scripts.train_mnist \
        --epochs $epochs \
        --n_latent $n_latent \
        --init_eta 0.000001 \
        --cv_lr 0 \
        --sample_baseline \
        --log logs/baseline_${n_latent}.npz
done
