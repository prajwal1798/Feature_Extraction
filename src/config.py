# =============================================================================
# CONFIG: EVERY SETTING OF THE PROJECT IN ONE PLACE
#
# One Settings object is created by run.py and passed to every routine as
# "cfg". Values below are the defaults of the published 500-case study;
# run.py overrides some of them from the command line.
#
# Units: SI throughout (m, K, W). Paths: relative paths are resolved from the
# project folder (the parent of src/), never from the current directory, so
# the code behaves the same on a laptop and on Sunbird.
# =============================================================================

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    # =========================================================================
    # DATASET SELECTION
    #   dataset_family        : "circle", "rect" or "combined" (half / half),
    #                           or "scaling" for the data-size study
    #   selected_dataset_size : number of FE cases; also names the folders
    #                           (for "scaling": the size of the training pool)
    # =========================================================================
    dataset_family: str = "combined"
    selected_dataset_size: int = 500

    # =========================================================================
    # DATA-SIZE STUDY (data/scaling.py)
    #   training pool : Halton sequence, nested (first N cases = training set N)
    #   validation and test sets: fixed, Latin hypercube, the same for every N
    #   train_size    : use only the first train_size training cases (0 = all)
    #   scaling_sizes : training-set sizes of the study (doubling -> log2 axis)
    # =========================================================================
    scaling_val_cases: int = 200
    scaling_test_cases: int = 400
    train_size: int = 0
    scaling_sizes: tuple = (200, 400, 800, 1600, 3200)

    # =========================================================================
    # STORAGE ROOTS (relative to the project folder, or absolute)
    # =========================================================================
    data_root: str = "data"
    results_root: str = "outputs/study_results"
    checkpoints_root: str = "outputs/checkpoints"
    logs_root: str = "outputs/logs"
    figures_root: str = "outputs/figures"
    dataset_name: str = "dataset.npz"
    model_name: str = "best_model.pt"

    # =========================================================================
    # PLATE GEOMETRY                                                     [m]
    # =========================================================================
    Lx: float = 0.036
    Ly: float = 0.021

    # =========================================================================
    # CAVITY SIZE RANGES FOR TRAINING DATA                               [m]
    # =========================================================================
    D_min: float = 0.008
    D_max: float = 0.014
    W_min: float = 0.008
    W_max: float = 0.014
    H_min: float = 0.006
    H_max: float = 0.012

    # =========================================================================
    # EXTERNAL TEST SET: ROUNDED RECTANGLES (never used in training)
    # =========================================================================
    rounded_test_n_cases: int = 10
    rounded_test_case_number: int = 8
    rounded_test_seed: int = 777
    rounded_test_dataset_name: str = "rounded_rect_test_dataset.npz"
    rounded_test_folder_name: str = "rounded_rect_external_test"
    rounded_W_min: float = 0.008
    rounded_W_max: float = 0.014
    rounded_H_min: float = 0.006
    rounded_H_max: float = 0.012
    rounded_R_min: float = 0.0005
    rounded_R_max: float = 0.003
    rounded_margin: float = 0.003
    rounded_corner_segments: int = 12

    # =========================================================================
    # PHYSICS
    #   kappa  : thermal conductivity            [W/(m K)]
    #   h_hole : heat transfer coefficient       [W/(m^2 K)]
    #   T_inf  : coolant temperature             [K]
    #   q_top  : heat flux on the top edge       [W/m^2]
    # =========================================================================
    kappa: float = 28.3
    h_hole: float = 4000.0
    T_inf: float = 300.0
    q_top: float = 0.5e6

    # =========================================================================
    # FINITE ELEMENTS AND LEARNING GRID
    #   order : polynomial order of the H1 elements
    #   maxh  : maximum element size                                     [m]
    #   Nx,Ny : learning-grid points along x and y
    # =========================================================================
    order: int = 2
    maxh: float = 0.0008
    Nx: int = 128
    Ny: int = 96

    # =========================================================================
    # DATA GENERATION
    #   n_cases must equal selected_dataset_size (checked in generate.py)
    # =========================================================================
    n_cases: int = 500
    seed: int = 42
    force_generation: bool = False

    # PREVIEW CASE (data/generate.py run_preview, analysis/mesh_check.py)
    preview_shape: str = "circle"
    preview_D: float = 0.012
    preview_W: float = 0.012
    preview_H: float = 0.008
    preview_cx: float = 0.018
    preview_cy: float = 0.0105

    # =========================================================================
    # FROZEN SPLIT (data/protocol.py)
    # =========================================================================
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    split_seed: int = 123

    # =========================================================================
    # OPERATOR AND ARCHITECTURE
    #   operator : "deeponet" or "fno"
    #   variant  : named architecture from models/build.py, which fills the
    #              fields below; "custom" keeps the fields as they are.
    #
    # DeepONet  branch : flattened SDF (Nx*Ny) -> branch_hidden -> latent_dim
    #           trunk  : (x/Lx, y/Ly, SDF)     -> trunk_hidden  -> latent_dim
    # FNO       lift (3 -> width), fno_blocks Fourier blocks keeping
    #           fno_modes_y x fno_modes_x modes, projection (width -> 64 -> 1)
    # =========================================================================
    operator: str = "deeponet"
    variant: str = "baseline"

    branch_hidden: tuple = (256, 256)
    trunk_hidden: tuple = (128, 128)
    latent_dim: int = 128
    deeponet_activation: str = "tanh"

    fno_modes_y: int = 12
    fno_modes_x: int = 16
    fno_width: int = 32
    fno_blocks: int = 4
    fno_padding: int = 8
    fno_activation: str = "gelu"

    # =========================================================================
    # TRAINING
    #   lr                  : initial Adam learning rate
    #   early_stop_patience : stop after this many epochs without a new best
    #                         validation loss
    #   scheduler_*         : ReduceLROnPlateau on the validation loss
    #   grad_clip           : maximum gradient 2-norm (0 = off)
    #   point_chunk         : DeepONet grid points per forward/backward piece
    #                         (memory only; the gradient is the full one)
    # =========================================================================
    epochs: int = 5000
    batch_size: int = 8
    lr: float = 1e-3
    early_stop_patience: int = 800
    scheduler_patience: int = 200
    scheduler_factor: float = 0.5
    grad_clip: float = 1.0
    point_chunk: int = 2048
    checkpoint_every: int = 50

    # =========================================================================
    # RUNTIME
    # =========================================================================
    device: str = "cuda"
    precision: str = "fp32"
    cpu_threads: int = 4
    train_seed: int = 123
    run_name: str = "deeponet_baseline_seed123"
    resume: bool = False

    # =========================================================================
    # EVALUATION AND FIGURES
    # =========================================================================
    test_case_number: int = 0
    show_figures: bool = False
    export_vtk: bool = False

    # =========================================================================
    # PATH HELPERS
    # =========================================================================
    def _root(self, value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def case_tag(self):
        return "%s_%04d" % (self.dataset_family, self.selected_dataset_size)

    @property
    def outdir(self):
        return str(self._root(self.results_root) / self.case_tag / self.run_name)

    @property
    def fe_outdir(self):
        return str(self._root(self.data_root) / "raw" / self.case_tag / "protocol_v2" / "fe_cases")

    @property
    def dataset_path(self):
        return str(self._root(self.data_root) / "processed" / self.case_tag / "protocol_v2"
                   / self.dataset_name)

    @property
    def checkpoint_outdir(self):
        return str(self._root(self.checkpoints_root) / self.case_tag / self.run_name)

    @property
    def log_outdir(self):
        return str(self._root(self.logs_root) / self.case_tag / self.run_name)

    @property
    def figure_outdir(self):
        return str(self._root(self.figures_root) / self.case_tag / self.run_name)

    @property
    def model_path(self):
        return str(Path(self.checkpoint_outdir) / self.model_name)

    @property
    def study_dir(self):
        # results shared by many runs: tuning tables, selections, seed summaries
        return self._root(self.results_root) / self.case_tag

    @property
    def rounded_test_outdir(self):
        return str(Path(self.outdir) / self.rounded_test_folder_name)

    @property
    def rounded_test_fe_outdir(self):
        return str(self._root(self.data_root) / "external" / self.case_tag / "protocol_v2"
                   / self.rounded_test_folder_name / "fe_cases")

    @property
    def rounded_test_dataset_path(self):
        return str(self._root(self.data_root) / "external" / self.case_tag / "protocol_v2"
                   / self.rounded_test_folder_name / self.rounded_test_dataset_name)

    def ensure_directories(self):
        directories = [
            self.outdir,
            self.fe_outdir,
            str(Path(self.dataset_path).parent),
            self.checkpoint_outdir,
            self.log_outdir,
            self.figure_outdir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    # PATHS STORED IN DATA FILES ARE RELATIVE TO THE PROJECT FOLDER
    def relative_project_path(self, value):
        path = Path(value).resolve()
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    # FIND AN FE CASE FILE FROM A STORED (POSSIBLY OLD WINDOWS) PATH
    def resolve_fe_path(self, value):
        path = Path(str(value))
        file_name = PureWindowsPath(str(value)).name

        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(PROJECT_ROOT / path)
        candidates.append(Path(self.dataset_path).parent / path)
        candidates.append(Path(self.fe_outdir) / file_name)
        candidates.append(Path(self.rounded_test_fe_outdir) / file_name)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        raise FileNotFoundError("FE case %r not found. Regenerate the dataset or restore its FE files."
                                % str(value))
