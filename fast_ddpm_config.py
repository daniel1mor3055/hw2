cifar10_cfg = {
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1, 2, 2, 2),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.1,
    "var_type": "fixedlarge"
}

diffusion_config = {
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "T": 200,
}
