"""Pairmixer - Efficient biomolecular structure prediction without attention.

This module provides programmatic access to Pairmixer models for use as a library.
Pairmixer removes triangle attention while preserving geometric reasoning through
triangle multiplication.

Example usage with hooks:

    import pairmixer

    # Load a model
    model = pairmixer.load_model("pairmixer", device="cuda")

    # Store captured activations
    activations = {}

    # Define a hook to capture triangle multiplication outputs
    def capture_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = tuple(
                    o.detach().cpu() if hasattr(o, 'detach') else o
                    for o in output
                )
            elif hasattr(output, 'detach'):
                activations[name] = output.detach().cpu()
        return hook

    # Register hooks on modules of interest
    model.pairformer_module.register_forward_hook(capture_hook("pairformer"))
    model.structure_module.register_forward_hook(capture_hook("structure"))

    # Run prediction
    results = pairmixer.predict(
        model,
        "protein.yaml",
        use_msa_server=True,
        diffusion_samples=1,
    )

    # Access captured pair representations
    print("Pairformer output shapes:")
    s, z = activations["pairformer"]
    print(f"  Sequence embedding (s): {s.shape}")
    print(f"  Pair embedding (z): {z.shape}")
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pairmixer")
except PackageNotFoundError:
    __version__ = "unknown"

# Lazy imports
Boltz1 = None
Manifest = None
Record = None


def _ensure_imports():
    """Lazily import model classes when first needed."""
    global Boltz1, Manifest, Record
    if Boltz1 is None:
        from pairmixer.model.models.boltz1 import Boltz1 as _Boltz1
        from pairmixer.data.types import Manifest as _Manifest, Record as _Record
        Boltz1 = _Boltz1
        Manifest = _Manifest
        Record = _Record


__all__ = [
    "Boltz1",
    "Manifest",
    "Record",
    "load_model",
    "predict",
    "__version__",
]


def load_model(
    model_name: str = "pairmixer",
    checkpoint: str = None,
    device: str = "cpu",
    use_kernels: bool = True,
    cache_dir: str = None,
    **kwargs,
):
    """Load a Pairmixer model for programmatic use.

    Pairmixer is a simplified architecture that removes triangle attention
    while preserving geometric reasoning through triangle multiplication.

    Parameters
    ----------
    model_name : str, optional
        Model to load. Only "pairmixer" is supported. Default is "pairmixer".
    checkpoint : str, optional
        Path to a custom checkpoint file. If None, downloads the default weights.
    device : str, optional
        Device to load the model on ("cpu", "cuda", "cuda:0", etc.). Default is "cpu".
    use_kernels : bool, optional
        Whether to use optimized CUDA kernels. Default is True.
    cache_dir : str, optional
        Directory to cache downloaded weights. Defaults to ~/.boltz or $BOLTZ_CACHE.
    **kwargs
        Additional arguments passed to the model's load_from_checkpoint method.

    Returns
    -------
    Boltz1
        The loaded Pairmixer model in eval mode (uses Boltz1 architecture without attention).

    Examples
    --------
    Basic usage:

        >>> import pairmixer
        >>> model = pairmixer.load_model("pairmixer")
        >>> model.eval()

    Register hooks for triangle multiplication analysis:

        >>> activations = {}
        >>> def capture_trimul(name):
        ...     def hook(module, input, output):
        ...         activations[name] = output.detach().cpu()
        ...     return hook
        >>>
        >>> # Hook triangle multiplication layers
        >>> for name, module in model.named_modules():
        ...     if "triangle" in name.lower():
        ...         module.register_forward_hook(capture_trimul(name))
    """
    import os
    import urllib.request
    from dataclasses import asdict, dataclass
    from pathlib import Path

    import torch

    # Ensure model classes are imported
    _ensure_imports()

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get("BOLTZ_CACHE", os.path.expanduser("~/.boltz"))
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # URLs for model weights
    PAIRMIXER_URLS = [
        "https://huggingface.co/genesisml/pairmixer/resolve/main/pairmixer.ckpt",
    ]

    # Only support pairmixer
    if model_name.lower() != "pairmixer":
        raise ValueError(f"Unknown model: {model_name}. Only 'pairmixer' is supported.")

    model_cls = Boltz1  # Pairmixer uses Boltz1 architecture
    default_ckpt = cache / "pairmixer.ckpt"
    urls = PAIRMIXER_URLS

    # Determine checkpoint path
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    else:
        ckpt_path = default_ckpt
        # Download if needed
        if not ckpt_path.exists():
            print(f"Downloading Pairmixer weights to {ckpt_path}...")
            for i, url in enumerate(urls):
                try:
                    urllib.request.urlretrieve(url, str(ckpt_path))
                    break
                except Exception as e:
                    if i == len(urls) - 1:
                        raise RuntimeError(
                            f"Failed to download model from all URLs. Last error: {e}"
                        ) from e
                    continue

    # Download CCD if needed
    ccd_path = cache / "ccd.pkl"
    if not ccd_path.exists():
        print(f"Downloading CCD dictionary to {ccd_path}...")
        CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
        urllib.request.urlretrieve(CCD_URL, str(ccd_path))

    # Default prediction arguments
    predict_args = {
        "recycling_steps": kwargs.pop("recycling_steps", 3),
        "sampling_steps": kwargs.pop("sampling_steps", 200),
        "diffusion_samples": kwargs.pop("diffusion_samples", 1),
        "max_parallel_samples": kwargs.pop("max_parallel_samples", 1),
        "write_confidence_summary": kwargs.pop("write_confidence_summary", True),
        "write_full_pae": kwargs.pop("write_full_pae", True),
        "write_full_pde": kwargs.pop("write_full_pde", False),
    }

    # Pairmixer uses Boltz1-style diffusion parameters
    @dataclass
    class DiffusionParams:
        gamma_0: float = 0.605
        gamma_min: float = 1.107
        noise_scale: float = 0.901
        rho: float = 8
        step_scale: float = 1.638
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True
        use_inference_model_cache: bool = True

    @dataclass
    class PairformerArgs:
        num_blocks: int = 48
        num_heads: int = 16
        dropout: float = 0.0
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        v2: bool = False

    @dataclass
    class MSAModuleArgs:
        msa_s: int = 64
        msa_blocks: int = 4
        msa_dropout: float = 0.0
        z_dropout: float = 0.0
        use_paired_feature: bool = False
        pairwise_head_width: int = 32
        pairwise_num_heads: int = 4
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        subsample_msa: bool = False
        num_subsampled_msa: int = 1024

    @dataclass
    class SteeringArgs:
        fk_steering: bool = False
        num_particles: int = 3
        fk_lambda: float = 4.0
        fk_resampling_interval: int = 3
        physical_guidance_update: bool = False
        contact_guidance_update: bool = True
        num_gd_steps: int = 20

    diffusion_params = DiffusionParams()
    pairformer_args = PairformerArgs()
    msa_args = MSAModuleArgs()
    steering_args = SteeringArgs()

    # Load model with pairmixer=True flag
    model = model_cls.load_from_checkpoint(
        str(ckpt_path),
        strict=True,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        pairmixer=True,  # KEY: This enables Pairmixer mode (no attention)
        **kwargs,
    )
    model.eval()

    return model


def predict(
    model,
    input_path: str,
    out_dir: str = None,
    use_msa_server: bool = False,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    device: str = None,
    cache_dir: str = None,
    num_workers: int = 4,
):
    """Run structure prediction with a loaded Pairmixer model.

    Parameters
    ----------
    model : Boltz1
        A loaded Pairmixer model (from `load_model()`).
    input_path : str
        Path to a YAML input file or directory of YAML files.
    out_dir : str, optional
        Output directory for predictions. Defaults to current directory.
    use_msa_server : bool, optional
        Whether to use the MSA server for automatic MSA generation.
    recycling_steps : int, optional
        Number of recycling steps. Default is 3.
    sampling_steps : int, optional
        Number of diffusion sampling steps. Default is 200.
    diffusion_samples : int, optional
        Number of structure samples to generate. Default is 1.
    device : str, optional
        Device to run on. Defaults to model's current device.
    cache_dir : str, optional
        Cache directory for downloaded data. Defaults to ~/.boltz.
    num_workers : int, optional
        Number of data loading workers. Default is 4.

    Returns
    -------
    list[dict]
        List of prediction results, one dict per input.
    """
    import os
    from pathlib import Path

    import torch

    # Ensure model classes are imported
    _ensure_imports()

    # Determine paths
    input_path = Path(input_path)
    if out_dir is None:
        out_dir = Path.cwd() / input_path.stem
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = os.environ.get("BOLTZ_CACHE", os.path.expanduser("~/.boltz"))
    cache = Path(cache_dir)

    # Collect input files
    if input_path.is_dir():
        input_files = list(input_path.glob("*.yaml")) + list(input_path.glob("*.yml"))
        input_files += list(input_path.glob("*.fasta")) + list(input_path.glob("*.fa"))
    else:
        input_files = [input_path]

    if not input_files:
        raise ValueError(f"No input files found at {input_path}")

    # Use the existing process_inputs function from main.py
    from pairmixer.main import process_inputs, download_pairmixer

    # Ensure required data is downloaded
    download_pairmixer(cache)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols" if (cache / "mols").exists() else cache

    # Process inputs
    process_inputs(
        data=input_files,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
        boltz2=False,  # Pairmixer is based on Boltz1 architecture
    )

    # Load the manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Set up data module (Pairmixer uses Boltz1 data module)
    from pairmixer.data.module.inference import BoltzInferenceDataModule
    data_module = BoltzInferenceDataModule(
        manifest=manifest,
        target_dir=out_dir / "processed" / "structures",
        msa_dir=out_dir / "processed" / "msa",
        num_workers=num_workers,
        constraints_dir=out_dir / "processed" / "constraints",
    )

    # Set up data module
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # Determine device
    if device is not None:
        model_device = torch.device(device)
    elif next(model.parameters()).is_cuda:
        model_device = next(model.parameters()).device
    else:
        model_device = torch.device("cpu")

    model = model.to(model_device)

    # Run inference
    all_results = []
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Run forward pass - hooks will fire here
            output = model(
                batch,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                max_parallel_samples=diffusion_samples,
                run_confidence_sequentially=True,
            )

            # Collect results
            result = {
                "coords": output["sample_atom_coords"].cpu(),
                "s": output["s"].cpu(),
                "z": output["z"].cpu(),
                "pdistogram": output["pdistogram"].cpu(),
            }

            # Add confidence outputs if available
            if "plddt" in output:
                result["plddt"] = output["plddt"].cpu()
            if "pae" in output:
                result["pae"] = output["pae"].cpu()
            if "pde" in output:
                result["pde"] = output["pde"].cpu()
            if "ptm" in output:
                result["ptm"] = output["ptm"].cpu()
            if "iptm" in output:
                result["iptm"] = output["iptm"].cpu()
            if "complex_plddt" in output:
                result["complex_plddt"] = output["complex_plddt"].cpu()

            all_results.append(result)

    return all_results